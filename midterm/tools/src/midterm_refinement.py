"""Bounded follow-up experiments after the first public leaderboard submission."""
from __future__ import annotations

from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from midterm_augmentation import estimate_cluster_reason_rates,logistic_parameters,observed_label_mask,predict_reason_probabilities
from midterm_evaluation import REASONS,PHASES,dataset_path,evaluate_part1,evaluate_part2,file_sha256
from midterm_part1_finalization import binary_decisions,build_reason_lookup,build_submission
from midterm_part2_baseline import TEXT_COLUMNS,analyze_failure_reasons,predict_outcome_probabilities
from midterm_part2_improvement import validate_submission

EMBEDDINGS=[f'emb_{i}' for i in range(150)]
P1_SETTINGS=[{'method':m,'clusters':k,'C':c} for m,k in
             [('supervised',0),('hard',10),('soft',10),('soft',30)] for c in [.1,1.]]
TFIDF={'ngram_range':(1,2),'min_df':3,'lowercase':True,'strip_accents':'unicode',
       'sublinear_tf':True,'smooth_idf':True,'use_idf':True,'norm':'l2','stop_words':None}


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def setting_id(setting):
    return f'p1ref_{setting["method"]}_k{setting["clusters"]}_c{str(setting["C"]).replace(".","")}_v1'


def fit_binary_targets(gold_x,gold_y,pseudo_x,pseudo_q,*,C,pseudo_weight=.1):
    """Soft targets use two weighted binary rows; their total weight stays fixed.

    For q in [0,1], weighted positive and negative rows contribute exactly
    -w*(q*log(p)+(1-q)*log(1-p)) to the logistic loss.
    """
    models=[]
    for j in range(len(REASONS)):
        if pseudo_q is None:
            x=gold_x;y=gold_y[:,j];weights=np.ones(len(y))
        else:
            q=pseudo_q[:,j]
            if not np.isfinite(q).all() or ((q<0)|(q>1)).any():raise ValueError('Invalid pseudo-target probabilities.')
            x=np.vstack([gold_x,pseudo_x,pseudo_x])
            y=np.concatenate([gold_y[:,j],np.ones(len(q)),np.zeros(len(q))])
            weights=np.concatenate([np.ones(len(gold_y)),pseudo_weight*q,pseudo_weight*(1-q)])
        model=LogisticRegression(**{**logistic_parameters(42),'C':C})
        with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(x,y,sample_weight=weights)
        if any(issubclass(w.category,ConvergenceWarning) for w in caught) or model.n_iter_.max()>=500:
            raise RuntimeError('A refinement classifier failed to converge.')
        models.append(model)
    return models


def prepare_reason_context(frame,clusters,seed=42):
    """Exclude held-out rows before fitting either scaling or clustering."""
    observed=observed_label_mask(frame[list(REASONS)].to_numpy())
    scaler=StandardScaler().fit(frame[EMBEDDINGS]);x=scaler.transform(frame[EMBEDDINGS])
    context={'scaler':scaler,'x':x,'observed':observed,'fit_ids':frame.ID.tolist(),
             'seed_ids':frame.loc[observed,'ID'].tolist()}
    if clusters:
        clusterer=KMeans(n_clusters=clusters,init='k-means++',n_init=20,max_iter=500,tol=1e-4,algorithm='lloyd',random_state=seed)
        with threadpool_limits(limits=1):clusterer.fit(x)
        memberships=np.eye(clusters)[clusterer.labels_]
        mapping=estimate_cluster_reason_rates(memberships,frame[list(REASONS)].to_numpy(),2.)
        context.update(clusterer=clusterer,mapping=mapping,pseudo_q=(memberships@mapping['rates'])[~observed])
    return context


def fit_reason_setting(frame,context,setting):
    observed=context['observed'];x=context['x']
    q=context.get('pseudo_q')
    if setting['method']=='hard':q=(q>=.5).astype(float)
    if setting['method']=='supervised':q=None
    models=fit_binary_targets(x[observed],frame.loc[observed,list(REASONS)].to_numpy(dtype=int),x[~observed],q,C=setting['C'])
    return {'setting':setting,'scaler':context['scaler'],'models':models,
            'threshold':.35,'empty_set_policy':'argmax','fit_ids':context['fit_ids'],'seed_ids':context['seed_ids'],
            'clusterer':context.get('clusterer'),'mapping':context.get('mapping'),'pseudo_q':q}


def predict_reasons(bundle,frame):
    scores=predict_reason_probabilities(bundle['models'],bundle['scaler'].transform(frame[EMBEDDINGS]))
    result=pd.DataFrame(scores,columns=REASONS);result.insert(0,'ID',frame.ID.to_numpy());return result


def save_run(directory,config,bundle,evaluation,extras=None):
    directory=Path(directory)
    if directory.exists():
        if json.loads((directory/'config.json').read_text())!=config:raise ValueError('Existing refinement run has different settings.')
        for name,digest in json.loads((directory/'checksums.json').read_text()).items():
            if file_sha256(directory/name)!=digest:raise ValueError(f'Changed immutable artifact: {name}')
        return
    directory.mkdir(parents=True)
    write_json(directory/'config.json',config);evaluation.save(directory)
    joblib.dump(bundle,directory/'model.joblib',compress=3)
    if extras is not None:write_json(directory/'fit_metadata.json',extras)
    write_json(directory/'checksums.json',{p.name:file_sha256(p) for p in directory.iterdir() if p.is_file()})


def fit_field_outcomes(train):
    """Give criteria, disease and drug text separate training-fitted L2 TF-IDF blocks."""
    models={};diagnostics=[]
    for phase in PHASES:
        rows=train.loc[train.phase.eq(phase)];vectors={};blocks=[]
        for field,limit in zip(TEXT_COLUMNS,[20000,5000,5000]):
            vector=TfidfVectorizer(**TFIDF,max_features=limit)
            matrix=vector.fit_transform(rows[field].fillna('').astype(str))
            vectors[field]=vector;blocks.append(matrix)
        x=sparse.hstack(blocks,format='csr')/np.sqrt(3.)
        classifier=LogisticRegression(**logistic_parameters(42))
        with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');classifier.fit(x,rows.label.to_numpy(dtype=int))
        if any(issubclass(w.category,ConvergenceWarning) for w in caught) or classifier.n_iter_.max()>=500:
            raise RuntimeError('Field-text classifier did not converge.')
        models[phase]={'vectorizers':vectors,'classifier':classifier,'fit_trial_ids':rows.trial_id.tolist()}
        diagnostics.append({'phase':phase,'iterations':int(classifier.n_iter_.max()),'rows':len(rows),
                            **{f'{field}_features':len(v.vocabulary_) for field,v in vectors.items()}})
    return {'models':models,'reason_features_used':False,'scale':float(1/np.sqrt(3.))},pd.DataFrame(diagnostics)


def predict_field_outcomes(bundle,frame):
    scores=np.full(len(frame),np.nan)
    for phase in PHASES:
        positions=np.flatnonzero(frame.phase.to_numpy()==phase)
        if not len(positions):continue
        rows=frame.iloc[positions];model=bundle['models'][phase]
        blocks=[model['vectorizers'][field].transform(rows[field].fillna('').astype(str)) for field in TEXT_COLUMNS]
        x=sparse.hstack(blocks,format='csr')*bundle['scale']
        scores[positions]=model['classifier'].predict_proba(x)[:,1]
    if not np.isfinite(scores).all():raise ValueError('Missing phase-routed predictions.')
    return pd.DataFrame({'ID':frame.ID.to_numpy(),'p_success':scores})


def run_refinement(root):
    root=Path(root);folder=root/'refinement';folder.mkdir(exist_ok=True)
    inputs={f'{task}_{split}':file_sha256(dataset_path(root,task,split)) for task in ['part1','part2'] for split in ['train','val']}
    plan={'seed':42,'part1_settings':P1_SETTINGS,'part1_cv_folds':4,'part1_threshold':.35,
        'part1_empty_policy':'argmax','pseudo_weight':.1,'smoothing_strength':2.,
        'cv_strategy':'StratifiedKFold on first-positive reason; held-out rows excluded from all fitted transforms and seeds',
        'official_validation_policy':'Evaluate one supervised and one augmented winner selected by training CV',
        'part2_settings':['baseline','separate_field_tfidf_lr_c1','equal_probability_blend'],
        'part2_field_weights':[float(1/np.sqrt(3.))]*3,'part2_threshold':.5,
        'selection_uses_public_scores':False,'original_final_outputs_modified':False,'input_sha256':inputs}
    pp=folder/'experiment_plan.json'
    if pp.exists():
        if json.loads(pp.read_text())!=plan:raise ValueError('Refinement plan changed; use a new study folder.')
    else:write_json(pp,plan)
    if (folder/'completed_checksums.json').exists():
        for name,digest in json.loads((folder/'completed_checksums.json').read_text()).items():
            if file_sha256(folder/name)!=digest:raise ValueError(f'Changed completed study output: {name}')
        return json.loads((folder/'summary.json').read_text())
    p1=folder/'part1';p2=folder/'part2';submissions=folder/'submissions'
    for path in [p1,p2,submissions]:path.mkdir(exist_ok=True)
    train=pd.read_csv(dataset_path(root,'part1','train'));mask=observed_label_mask(train[list(REASONS)].to_numpy())
    observed_positions=np.flatnonzero(mask);strata=train.loc[mask,list(REASONS)].to_numpy().argmax(axis=1)
    if min(np.bincount(strata))<4:raise ValueError('First-positive reason stratification requires four examples per stratum.')
    cv=StratifiedKFold(n_splits=4,shuffle=True,random_state=42)
    fold_rows=[];fold_provenance=[];oof={setting_id(s):[] for s in P1_SETTINGS}
    for fold,(seed_local,hold_local) in enumerate(cv.split(observed_positions,strata)):
        held=observed_positions[hold_local];fit=train.drop(index=train.index[held]).reset_index(drop=True)
        reference=train.iloc[held].copy()
        assert set(fit.ID).isdisjoint(reference.ID)
        contexts={k:prepare_reason_context(fit,k) for k in [0,10,30]}
        fold_provenance.append({'fold':fold,'fit_ids':fit.ID.tolist(),'held_out_ids':reference.ID.tolist(),
            'label_seed_ids':fit.loc[observed_label_mask(fit[list(REASONS)].to_numpy()),'ID'].tolist()})
        for setting in P1_SETTINGS:
            bundle=fit_reason_setting(fit,contexts[setting['clusters']],setting)
            probabilities=predict_reasons(bundle,reference)
            evaluation=evaluate_part1(reference,probabilities,.35,empty_set_policy='argmax')
            fold_rows.append({'setting_id':setting_id(setting),'fold':fold,**setting,**evaluation.summary})
            oof[setting_id(setting)].append(evaluation.predictions.assign(fold=fold))
        print(f'Completed training-only reason CV fold {fold+1}/4',flush=True)
    folds=pd.DataFrame(fold_rows);folds.to_csv(p1/'cv_fold_metrics.csv',index=False)
    table=folds.groupby(['setting_id','method','clusters','C'],as_index=False).agg(
        mean_f1_samples=('f1_samples','mean'),std_f1_samples=('f1_samples','std'),
        mean_f1_macro=('f1_macro','mean'),cv_rows=('n_rows','sum'))
    table=table.sort_values(['mean_f1_samples','setting_id'],ascending=[False,True]).reset_index(drop=True)
    table.to_csv(p1/'cv_comparison.csv',index=False);write_json(p1/'cv_fit_scope.json',fold_provenance)
    for key,frames in oof.items():pd.concat(frames).to_csv(p1/f'oof_{key}.csv',index=False)
    supervised_id=table.loc[table.method.eq('supervised'),'setting_id'].iloc[0]
    augmented_id=table.loc[table.method.ne('supervised'),'setting_id'].iloc[0]
    write_json(p1/'cv_selection_locked.json',{'supervised_id':supervised_id,'augmented_id':augmented_id,
        'metric':'mean_fold_sample_f1','validation_labels_used':False,'test_used':False})
    validation=pd.read_csv(dataset_path(root,'part1','val'))
    current_probs=pd.read_csv(root/'models/part1/02_kmeans_lr/evidence/selected/selected_validation/validation_predictions.csv')
    current_scores=current_probs[['ID',*[f'prob_{reason}' for reason in REASONS]]].rename(columns={f'prob_{reason}':reason for reason in REASONS})
    current_eval=evaluate_part1(validation,current_scores,.35,empty_set_policy='argmax')
    models={};evaluations={'current_submission':current_eval};summary_rows=[{'setting_id':'current_submission',**current_eval.summary}]
    for key in [supervised_id,augmented_id]:
        setting=next(s for s in P1_SETTINGS if setting_id(s)==key)
        context=prepare_reason_context(train,setting['clusters']);bundle=fit_reason_setting(train,context,setting)
        probabilities=predict_reasons(bundle,validation);evaluation=evaluate_part1(validation,probabilities,.35,empty_set_policy='argmax')
        config={'task':'part1','setting':setting,'threshold':.35,'decision_policy':'argmax_if_empty','selection':'training_CV_winner',
                'input_sha256':{k:v for k,v in inputs.items() if k.startswith('part1_')}}
        save_run(p1/'runs'/key,config,bundle,evaluation,{'fit_ids':bundle['fit_ids'],'seed_ids':bundle['seed_ids'],
            'kmeans_iterations':int(context['clusterer'].n_iter_) if setting['clusters'] else None})
        models[key]=bundle;evaluations[key]=evaluation;summary_rows.append({'setting_id':key,**evaluation.summary})
        if setting['clusters']:
            q=context['pseudo_q'];rows=train.loc[~mask,['ID']].copy()
            for j,reason in enumerate(REASONS):rows[f'cluster_probability_{reason}']=q[:,j]
            rows.to_csv(p1/'augmented_cluster_probabilities.csv',index=False)
            target_summary=pd.DataFrame({'reason':REASONS,'masked_rows':len(q),'mean_cluster_probability':q.mean(axis=0),
                'hard_positive_rows_at_05':(q>=.5).sum(axis=0),'effective_positive_pseudo_weight':(.1*q).sum(axis=0)})
            target_summary.to_csv(p1/'pseudo_target_summary.csv',index=False)
    validation_comparison=pd.DataFrame(summary_rows);validation_comparison.to_csv(p1/'validation_comparison.csv',index=False)
    pd.concat([e.by_group.assign(setting_id=k) for k,e in evaluations.items()]).to_csv(p1/'validation_by_reason.csv',index=False)
    candidate_eval=evaluations[augmented_id]
    better=candidate_eval.summary['f1_samples']>current_eval.summary['f1_samples']+1e-12
    chosen_id=augmented_id if better else 'current_submission'
    p1_selection={'chosen_candidate':chosen_id,'augmented_cv_winner':augmented_id,'supervised_cv_winner':supervised_id,
        'improved_official_validation':better,'source_current_score':current_eval.summary['f1_samples'],
        'augmented_candidate_score':candidate_eval.summary['f1_samples'],'test_used_for_selection':False}
    write_json(p1/'submission_selection_locked.json',p1_selection)
    # Paired resampling describes validation uncertainty, without a significance claim.
    def row_f1(e):
        y=e.predictions[[f'true_{v}' for v in REASONS]].to_numpy();d=e.predictions[[f'pred_{v}' for v in REASONS]].to_numpy()
        return 2*(y*d).sum(axis=1)/(y.sum(axis=1)+d.sum(axis=1))
    delta=row_f1(candidate_eval)-row_f1(current_eval)
    rng=np.random.default_rng(42);boot=delta[rng.integers(0,len(delta),size=(5000,len(delta)))].mean(axis=1)
    write_json(p1/'paired_validation_uncertainty.json',{'delta_sample_f1':float(delta.mean()),
        'descriptive_95_percentile_interval':np.quantile(boot,[.025,.975]).tolist(),
        'warning':'Does not account for previous validation selection or nonrandom labeled seeds.'})
    if better:
        test=pd.read_csv(dataset_path(root,'part1','test'));bundle=models[augmented_id]
        probs=predict_reasons(bundle,test);decisions=binary_decisions(probs[list(REASONS)],dict.fromkeys(REASONS,.35),empty_set_policy='argmax')
        pred=pd.DataFrame(decisions,columns=REASONS);pred.insert(0,'ID',test.ID.to_numpy())
        sample=pd.read_csv(dataset_path(root,'part1','test').parent/'part1_sample_submission.csv')
        submission=build_submission(sample,pred);probs.to_csv(submissions/'part1_candidate_probabilities.csv',index=False)
        submission.to_csv(submissions/'part1_candidate.csv',index=False);validate_submission(pd.read_csv(submissions/'part1_candidate.csv'),sample,1)
        frames={'train':train,'val':validation,'test':test}
        lookup=build_reason_lookup(frames,{s:predict_reasons(bundle,f)[list(REASONS)].to_numpy() for s,f in frames.items()},
            dict.fromkeys(REASONS,.35),augmented_id,empty_set_policy='argmax')
        lookup.to_csv(p1/'reason_lookup_candidate.csv',index=False)
    # Field-balanced outcome model and an a priori equal-probability blend.
    outcome_train=pd.read_csv(dataset_path(root,'part2','train'));outcome_val=pd.read_csv(dataset_path(root,'part2','val'))
    base=joblib.load(root/'models/part2/01_tfidf_lr/model.joblib')
    base_probs=predict_outcome_probabilities(base,outcome_val)
    field,diagnostics=fit_field_outcomes(outcome_train);field_probs=predict_field_outcomes(field,outcome_val)
    blend_probs=base_probs.copy();blend_probs['p_success']=.5*base_probs.p_success+.5*field_probs.p_success
    p2_probabilities={'baseline':base_probs,'field_tfidf_lr_c1':field_probs,'blend_equal':blend_probs}
    p2_evaluations={k:evaluate_part2(outcome_val,v,.5) for k,v in p2_probabilities.items()}
    save_run(p2/'runs/field_tfidf_lr_c1_v1',{'task':'part2','C':1.,'fields':list(TEXT_COLUMNS),'field_scale':float(1/np.sqrt(3.)),
        'tfidf':{**TFIDF,'ngram_range':[1,2]},'max_features_by_field':[20000,5000,5000],
        'input_sha256':{k:v for k,v in inputs.items() if k.startswith('part2_')}},field,p2_evaluations['field_tfidf_lr_c1'])
    diagnostics.to_csv(p2/'field_diagnostics.csv',index=False)
    p2_comparison=pd.DataFrame([{'setting_id':k,**v.summary} for k,v in p2_evaluations.items()]).sort_values('log_loss_equal_phase')
    p2_comparison.to_csv(p2/'validation_comparison.csv',index=False)
    pd.concat([v.by_group.assign(setting_id=k) for k,v in p2_evaluations.items()]).to_csv(p2/'validation_by_phase.csv',index=False)
    for key,result in p2_evaluations.items():result.save(p2/key)
    best_p2=p2_comparison.iloc[0].setting_id;p2_better=best_p2!='baseline'
    write_json(p2/'submission_selection_locked.json',{'chosen_candidate':best_p2,'metric':'log_loss_equal_phase',
        'score':p2_evaluations[best_p2].summary['log_loss_equal_phase'],'test_used':False})
    if p2_better:
        test=pd.read_csv(dataset_path(root,'part2','test'));probs=predict_field_outcomes(field,test)
        if best_p2=='blend_equal':probs['p_success']=.5*probs.p_success+.5*predict_outcome_probabilities(base,test).p_success
        sample=pd.read_csv(dataset_path(root,'part2','test').parent/'part2_sample_submission.csv')
        submission=sample[['ID']].merge(probs,on='ID',validate='one_to_one',sort=False).rename(columns={'p_success':'label'})
        validate_submission(submission,sample,2);submission.to_csv(submissions/'part2_candidate.csv',index=False)
    frozen_lookup=pd.read_csv(root/'models/part1/02_kmeans_lr/evidence/selected/reason_lookup.csv')
    before=analyze_failure_reasons(p2_evaluations['baseline'].predictions,frozen_lookup)
    after=analyze_failure_reasons(p2_evaluations[best_p2].predictions,frozen_lookup)
    rates=before['reason_rates'].merge(after['reason_rates'],on=['phase','reason_group'],suffixes=('_baseline','_candidate'))
    rates.to_csv(p2/'reason_error_comparison.csv',index=False)
    summary={'part1':p1_selection,'part1_validation':validation_comparison.to_dict('records'),
        'part2':{'chosen_candidate':best_p2,'improved_official_validation':p2_better,
            'baseline_log_loss':p2_evaluations['baseline'].summary['log_loss_equal_phase'],
            'candidate_log_loss':p2_evaluations[best_p2].summary['log_loss_equal_phase']},
        'new_public_scores_known':False,'public_scores_used_to_select_settings':False,
        'original_final_outputs_modified':False,'model_source_sha256':file_sha256(Path(__file__))}
    write_json(folder/'summary.json',summary)
    checksums={str(p.relative_to(folder)):file_sha256(p) for p in folder.rglob('*') if p.is_file()}
    write_json(folder/'completed_checksums.json',checksums)
    return summary


if __name__ == "__main__":
    raise SystemExit("Historical study runner archived. Use python tools/src/train_core.py --help for the six fixed recipes.")
