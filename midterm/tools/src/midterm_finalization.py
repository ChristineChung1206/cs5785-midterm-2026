"""Predict historical/final outcome bundles and dispatch the final closeout."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from midterm_evaluation import file_sha256
from midterm_part2_baseline import predict_outcome_probabilities
from midterm_refinement import predict_field_outcomes

WORD_RUN = 'part2_equal_probability_blend_seed42_v1'
FINAL_RUN = 'part2_word_character_blend_seed42_v1'
FIELD_RUN = 'field_tfidf_lr_c1_v1'


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def check_run(directory, checksum_name):
    directory=Path(directory)
    for name,digest in json.loads((directory/checksum_name).read_text()).items():
        if Path(name).is_absolute() or '..' in Path(name).parts or file_sha256(directory/name)!=digest:
            raise AssertionError(f'Immutable experiment changed: {directory.name}/{name}')


def blend_probabilities(reference, components, weights):
    """Align components by ID before combining success probabilities."""
    w=np.asarray(weights,dtype=float)
    if len(components)!=len(w) or not np.isfinite(w).all() or (w<0).any() or not np.isclose(w.sum(),1):
        raise ValueError('Blend weights must be finite, nonnegative, and sum to one.')
    if reference.ID.isna().any() or not reference.ID.is_unique:raise ValueError('Reference IDs must be unique.')
    scores=[]
    for component in components:
        if component.ID.isna().any() or not component.ID.is_unique or set(component.ID)!=set(reference.ID):
            raise ValueError('Component ID sets must match the reference exactly.')
        p=component.set_index('ID').loc[reference.ID,'p_success'].to_numpy(dtype=float)
        if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid component probabilities.')
        scores.append(p)
    return pd.DataFrame({'ID':reference.ID.to_numpy(),'p_success':np.sum(w[:,None]*np.array(scores),axis=0)})


def predict_final_outcomes(bundle,frame):
    """Preserve old word-only inference and support the promoted character blend."""
    if bundle.get('positive_class')!=1 or bundle.get('reason_features_used') is not False:
        raise ValueError('Unexpected final outcome bundle.')
    if bundle.get('weights')!=[.5,.5]:
        raise ValueError('The selected nested blend uses fixed equal weights.')
    kind=bundle.get('kind')
    if kind=='equal_probability_blend':
        parts=[predict_outcome_probabilities(bundle['baseline'],frame),predict_field_outcomes(bundle['field'],frame)]
    elif kind=='word_character_probability_blend':
        if bundle.get('flat_weights')!={'concatenated_word':.25,'separate_field_word':.25,'character':.5}:
            raise ValueError('The promoted blend has fixed 25/25/50 component weights.')
        parts=[predict_final_outcomes(bundle['word_blend'],frame),predict_outcome_probabilities(bundle['character'],frame)]
    else:
        raise ValueError('Unexpected final outcome bundle kind.')
    return blend_probabilities(frame,parts,bundle['weights'])


def integrate_final_models(root):
    raise RuntimeError(
        "The historical fifteen-notebook integration is stored in "
        "../archive/midterm_before_cleanup_2026-11-02.zip. "
        "Use python reproduce.py for this curated project."
    )
