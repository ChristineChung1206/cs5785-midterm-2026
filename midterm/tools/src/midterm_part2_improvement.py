"""Controlled outcome-model improvements, fixed selection, and submission validation."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import re
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from midterm_augmentation import logistic_parameters
from midterm_evaluation import PHASES, evaluate_part2, file_sha256
from midterm_part1_finalization import BroadPathwayEncoder
from midterm_part2_baseline import TEXT_COLUMNS, build_trial_text, normalized_json, validate_outcome_rows

BASELINE_RUN = 'part2_tfidf_lr_by_phase_seed42_v1'
STRUCTURED_COLUMNS = ('start_year', 'icdcodes', 'drug_pathways', 'matched_drugs')


def icd_categories(value):
    """Extract unique three-character ICD categories from nested list strings."""
    if pd.isna(value):
        return set()
    codes = re.findall(r'\b[A-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?\b', str(value).upper())
    return {code[:3] for code in codes}


class OutcomeMetadataEncoder:
    """Fit ICD vocabulary, broad pathways, and bounded year values on one training phase."""
    def __init__(self, min_df=5):
        self.min_df = min_df

    def fit(self, frame, catalog):
        if not set(STRUCTURED_COLUMNS).issubset(frame):
            raise ValueError('Missing declared structured inputs.')
        counts = Counter(token for value in frame['icdcodes'] for token in icd_categories(value))
        self.icd_vocabulary_ = sorted(token for token, count in counts.items() if count >= self.min_df)
        self.icd_df_ = {token: counts[token] for token in self.icd_vocabulary_}
        self.pathways_ = BroadPathwayEncoder(min_df=self.min_df).fit(frame, catalog)
        years = pd.to_numeric(frame['start_year'], errors='raise').to_numpy(dtype=float)
        if np.isinf(years).any() or np.isnan(years).all():
            raise ValueError('Training years require finite observed values.')
        self.year_median_ = float(np.nanmedian(years))
        self.year_min_ = float(np.nanmin(years))
        self.year_max_ = float(np.nanmax(years))
        self.year_scaler_ = StandardScaler().fit(np.where(np.isnan(years), self.year_median_, years).reshape(-1, 1))
        self.feature_names_ = ([f'icd__{token}' for token in self.icd_vocabulary_] + ['has_any_icd'] +
            self.pathways_.feature_names_ + ['year_standardized', 'year_missing'])
        self.fit_trial_ids_ = frame['trial_id'].tolist()
        return self

    def transform(self, frame):
        if not hasattr(self, 'feature_names_'):
            raise ValueError('Fit metadata preprocessing before transform.')
        categories = frame['icdcodes'].map(icd_categories)
        columns = [categories.map(lambda tokens: token in tokens).to_numpy(dtype=float)
                   for token in self.icd_vocabulary_]
        columns.append(categories.map(bool).to_numpy(dtype=float))
        icd = np.column_stack(columns)
        years = pd.to_numeric(frame['start_year'], errors='raise').to_numpy(dtype=float)
        if np.isinf(years).any():
            raise ValueError('Years must be finite or missing.')
        missing = np.isnan(years)
        imputed = np.where(missing, self.year_median_, years)
        bounded = np.clip(imputed, self.year_min_, self.year_max_)
        year = self.year_scaler_.transform(bounded.reshape(-1, 1))
        return sparse.csr_matrix(np.column_stack([icd, self.pathways_.transform(frame), year, missing.astype(float)]))

    def audit_record(self):
        return {'icd_vocabulary': self.icd_vocabulary_, 'icd_training_document_frequencies': self.icd_df_,
                'pathway_vocabulary': self.pathways_.vocabulary_, 'feature_names': self.feature_names_,
                'year_median': self.year_median_, 'year_min': self.year_min_, 'year_max': self.year_max_,
                'year_scaler_mean': self.year_scaler_.mean_.tolist(),
                'year_scaler_scale': self.year_scaler_.scale_.tolist(), 'min_df': self.min_df}


def candidate_run_id(structured, C):
    tag = {0.1: '01', 1.0: '1', 10.0: '10'}[float(C)]
    return BASELINE_RUN if not structured and C == 1 else f'part2_s7_{"metadata" if structured else "text"}_lr_c{tag}_seed42_v1'


def candidate_settings(structured, C, catalog_sha):
    return {'baseline_vocabulary_run': BASELINE_RUN, 'C': float(C), 'structured_features': bool(structured),
        'metadata': {'columns': list(STRUCTURED_COLUMNS), 'icd_representation': 'binary_three_character_categories',
            'min_training_df': 5, 'pathway_level': 0, 'binary_features_unscaled': True,
            'year_imputation': 'training_phase_median', 'year_scaling': 'training_phase_StandardScaler',
            'year_clipping': 'training_phase_observed_min_max', 'missing_year_indicator': True,
            'block_weight': 1.0, 'catalog_sha256': catalog_sha} if structured else None,
        'classifier': {'solver': 'liblinear', 'max_iter': 500, 'tol': 1e-4, 'effective_regularization': 'L2',
                       'class_weight': None, 'fit_intercept': True, 'random_state': 42},
        'threshold': 0.5, 'fit_split': 'training_only', 'reason_features_used': False}


def fit_candidate(train, baseline, catalog, *, structured, C):
    """Reuse the training-fitted text vocabularies, and vary only declared factors."""
    validate_outcome_rows(train)
    models, diagnostics, fit_scope = {}, [], {}
    for phase in PHASES:
        rows = train.loc[train['phase'].eq(phase)]
        vectorizer = baseline['pipelines'][phase].named_steps['tfidf']
        text_matrix = vectorizer.transform(build_trial_text(rows))
        encoder = OutcomeMetadataEncoder().fit(rows, catalog) if structured else None
        matrix = sparse.hstack([text_matrix, encoder.transform(rows)], format='csr') if encoder else text_matrix
        classifier = LogisticRegression(**{**logistic_parameters(42), 'C': float(C)})
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            classifier.fit(matrix, rows['label'].to_numpy(dtype=int))
        if any(issubclass(w.category, ConvergenceWarning) for w in caught) or classifier.n_iter_.max() >= 500:
            raise RuntimeError(f'Phase {phase} did not converge.')
        if not np.array_equal(classifier.classes_, [0, 1]):
            raise RuntimeError('Expected success=1 as the positive class.')
        models[phase] = {'vectorizer': vectorizer, 'metadata_encoder': encoder, 'classifier': classifier}
        for component in ['frozen_tfidf', 'classifier'] + (['metadata_encoder'] if structured else []):
            fit_scope[f'{component}_phase_{phase}'] = rows['trial_id'].tolist()
        diagnostics.append({'phase': phase, 'training_rows': len(rows), 'text_features': text_matrix.shape[1],
            'metadata_features': matrix.shape[1]-text_matrix.shape[1], 'total_features': matrix.shape[1],
            'classifier_iterations': int(classifier.n_iter_.max()), 'converged': True,
            'year_median': encoder.year_median_ if encoder else np.nan,
            'year_min': encoder.year_min_ if encoder else np.nan, 'year_max': encoder.year_max_ if encoder else np.nan})
    return {'models': models, 'text_columns': list(TEXT_COLUMNS), 'structured': bool(structured),
            'C': float(C), 'positive_class': 1, 'reason_features_used': False}, pd.DataFrame(diagnostics), fit_scope


def predict_candidate(bundle, frame):
    validate_outcome_rows(frame, labels_required=False)
    probabilities = np.full(len(frame), np.nan)
    for phase in PHASES:
        positions = np.flatnonzero(frame['phase'].to_numpy() == phase)
        if not len(positions):
            continue
        rows = frame.iloc[positions]
        model = bundle['models'][phase]
        matrix = model['vectorizer'].transform(build_trial_text(rows))
        if model['metadata_encoder'] is not None:
            matrix = sparse.hstack([matrix, model['metadata_encoder'].transform(rows)], format='csr')
        probabilities[positions] = model['classifier'].predict_proba(matrix)[:, 1]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Expected finite success probabilities in [0, 1].')
    return pd.DataFrame({'ID': frame['ID'].to_numpy(), 'p_success': probabilities})


def load_candidate(directory, settings, validation, input_files):
    """Check frozen run settings, inputs, artifacts, and reproduced predictions."""
    directory = Path(directory)
    config = json.loads((directory/'config.json').read_text())
    if config['hyperparameters'] != normalized_json(settings) or config['input_files'] != input_files:
        raise ValueError('Run settings or training/validation inputs changed; use a new run ID.')
    if config['task'] != 'part2' or config['thresholds'] != dict.fromkeys(PHASES, .5):
        raise ValueError('Unexpected evaluation convention.')
    manifest = json.loads((directory/'artifact_checksums.json').read_text())
    for name, digest in manifest.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or file_sha256(directory/name) != digest:
            raise ValueError(f'Immutable run artifact changed: {name}')
    bundle = joblib.load(directory/'model.joblib')
    result = evaluate_part2(validation, predict_candidate(bundle, validation), .5)
    saved = pd.read_csv(directory/'validation_predictions.csv').set_index('ID').loc[validation['ID']]
    if not np.allclose(result.predictions['p_success'], saved['p_success'], atol=1e-12, rtol=0):
        raise ValueError('Saved model predictions do not reproduce.')
    if not np.array_equal(result.predictions['prediction'], saved['prediction']):
        raise ValueError('Saved binary predictions do not reproduce.')
    return bundle, result


def validate_submission(frame, sample, part):
    """Require exact sample IDs/order/schema and the competition-specific output domain."""
    expected = ['ID', 'Predicted'] if part == 1 else ['ID', 'label'] if part == 2 else None
    if expected is None or list(frame) != expected or list(sample) != expected:
        raise ValueError('Incorrect submission columns.')
    if not frame['ID'].is_unique or frame['ID'].isna().any() or frame['ID'].tolist() != sample['ID'].tolist():
        raise ValueError('Submission IDs/order must match the supplied sample exactly.')
    if part == 2:
        scores = pd.to_numeric(frame['label'], errors='raise').to_numpy(dtype=float)
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError('Outcome submissions require probabilities in [0, 1].')
    else:
        allowed = {'Safety','Funding','Efficacy','Recruitment','Other'}
        if frame['Predicted'].isna().any():
            raise ValueError('Reason submissions cannot contain null values.')
        for value in frame['Predicted']:
            if not isinstance(value, str) or not value.strip():
                raise ValueError('Reason submissions require at least one reason per row.')
            tokens = value.split()
            if len(tokens) != len(set(tokens)) or not set(tokens).issubset(allowed):
                raise ValueError('Invalid or repeated reason labels.')
    return {'part': part, 'rows': len(frame), 'unique_ids': frame['ID'].nunique(), 'columns': expected, 'status': 'PASS'}
