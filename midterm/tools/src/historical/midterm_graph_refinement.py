"""Local-geometry comparison selected by training-only cross-validation."""

from _historical_support import prepare_output
from pathlib import Path
import json,warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
from threadpoolctl import threadpool_limits

from midterm_augmentation import observed_label_mask,predict_reason_probabilities
from midterm_evaluation import REASONS,dataset_path,evaluate_part1,file_sha256
from midterm_part1_finalization import binary_decisions,build_submission,build_reason_lookup
from midterm_part2_improvement import validate_submission
from midterm_refinement import EMBEDDINGS,fit_binary_targets,save_run,write_json

SETTINGS=[{'method':m,'neighbors':k} for m in ['knn','graph','graph_lr'] for k in [5,15]]

def key(setting):return f'p1geo_{setting["method"]}_k{setting["neighbors"]}_v1'


def safe_positive_probability(probabilities,classes,prior):
    values=np.asarray(probabilities,dtype=float)
    valid=np.isfinite(values).all(axis=1)&(np.nan_to_num(values).sum(axis=1)>0)
    result=np.full(len(values),prior,dtype=float)
    result[valid]=values[valid,list(classes).index(1)]
    return result,~valid


def fit_geometry(frame,setting):
    observed=observed_label_mask(frame[list(REASONS)].to_numpy())
    scaler=StandardScaler().fit(frame[EMBEDDINGS]);x=scaler.transform(frame[EMBEDDINGS])
    y=frame[list(REASONS)].to_numpy();prior=y[observed].mean(axis=0)
    models=[];q=[];iterations=[];fallbacks=[]
    for j in range(5):
        if setting['method']=='knn':
            model=KNeighborsClassifier(n_neighbors=setting['neighbors'],weights='distance',metric='euclidean',n_jobs=1)
            with threadpool_limits(limits=1):model.fit(x[observed],y[observed,j])
        else:
            targets=np.where(observed,y[:,j],-1).astype(int)
            model=LabelSpreading(kernel='knn',n_neighbors=setting['neighbors'],alpha=.2,max_iter=200,tol=1e-4,n_jobs=1)
            with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter('always');model.fit(x,targets)
            if any(issubclass(w.category,ConvergenceWarning) for w in recorded) or model.n_iter_>=200:
                raise RuntimeError('Graph label spreading did not converge.')
            scores,missing=safe_positive_probability(model.label_distributions_,model.classes_,prior[j])
            q.append(scores[~observed]);iterations.append(int(model.n_iter_));fallbacks.append(int(missing[~observed].sum()))
        models.append(model)
    bundle={'setting':setting,'scaler':scaler,'geometry_models':models,'prior':prior,'fit_ids':frame.ID.tolist(),
        'seed_ids':frame.loc[observed,'ID'].tolist(),'threshold':.35,'empty_policy':'argmax',
        'iterations':iterations,'unconnected_pseudo_fallbacks':fallbacks}
    if setting['method']=='graph_lr':
        q=np.column_stack(q)
        bundle['classifier_models']=fit_binary_targets(x[observed],y[observed],x[~observed],q,C=.1,pseudo_weight=.1)
        bundle['pseudo_q']=q
    elif setting['method']=='graph':bundle['pseudo_q']=np.column_stack(q)
    return bundle


def predict_geometry(bundle,frame):
    x=bundle['scaler'].transform(frame[EMBEDDINGS])
    if bundle['setting']['method']=='graph_lr':
        scores=predict_reason_probabilities(bundle['classifier_models'],x);fallbacks=np.zeros_like(scores,dtype=bool)
    else:
        scores=[];fallbacks=[]
        for j,model in enumerate(bundle['geometry_models']):
            with threadpool_limits(limits=1),np.errstate(invalid='ignore',divide='ignore'):
                distribution=model.predict_proba(x)
            score,missing=safe_positive_probability(distribution,model.classes_,bundle['prior'][j])
            scores.append(score);fallbacks.append(missing)
        scores=np.column_stack(scores);fallbacks=np.column_stack(fallbacks)
    result=pd.DataFrame(scores,columns=REASONS);result.insert(0,'ID',frame.ID.to_numpy())
    return result,fallbacks


def run_geometry(root, output):
    root, folder = prepare_output(root, output)
    plan={'settings':SETTINGS,'seed':42,'cv_folds':4,'threshold':.35,'empty_set_policy':'argmax',
        'label_spreading':{'kernel':'knn','alpha':.2,'max_iter':200,'tol':1e-4,'disconnected_fallback':'observed-training reason prior'},
        'graph_lr':{'C':.1,'pseudo_weight':.1,'targets':'soft training-graph distributions'},
        'selection':'One graph winner and one kNN control by mean training-CV sample F1, then provided validation confirmation',
        'input_sha256':{s:file_sha256(dataset_path(root,'part1',s)) for s in ['train','val']},
        'public_score_used_for_selection':False,'test_used_for_selection':False}
    path=folder/'experiment_plan.json'
    if path.exists():
        if json.loads(path.read_text())!=plan:raise ValueError('Local geometry plan changed.')
    else:write_json(path,plan)
    if (folder/'completed_checksums.json').exists():
        for name,digest in json.loads((folder/'completed_checksums.json').read_text()).items():
            if file_sha256(folder/name)!=digest:raise ValueError(f'Changed geometry result: {name}')
        return json.loads((folder/'summary.json').read_text())
    train=pd.read_csv(dataset_path(root,'part1','train'));observed=observed_label_mask(train[list(REASONS)].to_numpy())
    positions=np.flatnonzero(observed);strata=train.loc[observed,list(REASONS)].to_numpy().argmax(axis=1)
    cv=StratifiedKFold(n_splits=4,shuffle=True,random_state=42)
    metrics=[];provenance=[]
    for fold,(_,hold) in enumerate(cv.split(positions,strata)):
        held=positions[hold];fit=train.drop(index=train.index[held]).reset_index(drop=True);reference=train.iloc[held]
        assert set(fit.ID).isdisjoint(reference.ID)
        provenance.append({'fold':fold,'fit_ids':fit.ID.tolist(),'held_out_ids':reference.ID.tolist(),
            'seed_ids':fit.loc[observed_label_mask(fit[list(REASONS)].to_numpy()),'ID'].tolist()})
        for setting in SETTINGS:
            bundle=fit_geometry(fit,setting);probabilities,fallback=predict_geometry(bundle,reference)
            result=evaluate_part1(reference,probabilities,.35,empty_set_policy='argmax')
            metrics.append({'setting_id':key(setting),'fold':fold,**setting,**result.summary,'graph_unconnected_predictions':int(fallback.sum())})
        print(f'Completed local-geometry CV fold {fold+1}/4',flush=True)
    folds=pd.DataFrame(metrics);folds.to_csv(folder/'cv_fold_metrics.csv',index=False)
    comparison=folds.groupby(['setting_id','method','neighbors'],as_index=False).agg(
        mean_f1_samples=('f1_samples','mean'),std_f1_samples=('f1_samples','std'),mean_f1_macro=('f1_macro','mean'))
    comparison=comparison.sort_values(['mean_f1_samples','setting_id'],ascending=[False,True]).reset_index(drop=True)
    comparison.to_csv(folder/'cv_comparison.csv',index=False);write_json(folder/'cv_fit_scope.json',provenance)
    graph_id=comparison.loc[comparison.method.ne('knn'),'setting_id'].iloc[0]
    knn_id=comparison.loc[comparison.method.eq('knn'),'setting_id'].iloc[0]
    write_json(folder/'cv_selection_locked.json',{'graph_id':graph_id,'knn_id':knn_id,'validation_used':False})
    validation=pd.read_csv(dataset_path(root,'part1','val'))
    current=pd.read_csv(root/'models/part1/02_kmeans_lr/evidence/selected/selected_validation/validation_predictions.csv')
    current_scores=current[['ID',*[f'prob_{r}' for r in REASONS]]].rename(columns={f'prob_{r}':r for r in REASONS})
    original=evaluate_part1(validation,current_scores,.35,empty_set_policy='argmax')
    results={'current_submission':original};models={};summary_rows=[{'setting_id':'current_submission',**original.summary}]
    for run_id in [graph_id,knn_id]:
        setting=next(s for s in SETTINGS if key(s)==run_id)
        bundle=fit_geometry(train,setting);probabilities,fallbacks=predict_geometry(bundle,validation)
        result=evaluate_part1(validation,probabilities,.35,empty_set_policy='argmax')
        config={'task':'part1','setting':setting,'threshold':.35,'empty_policy':'argmax','input_files':plan['input_sha256'],
            'CV_selection_locked_before_validation':True}
        save_run(folder/'runs'/run_id,config,bundle,result,{'fit_ids':bundle['fit_ids'],'seed_ids':bundle['seed_ids'],
            'graph_iterations':bundle['iterations'],'unconnected_pseudo_rows_by_reason':bundle['unconnected_pseudo_fallbacks'],
            'validation_unconnected_predictions':int(fallbacks.sum())})
        results[run_id]=result;models[run_id]=bundle;summary_rows.append({'setting_id':run_id,**result.summary})
    pd.DataFrame(summary_rows).to_csv(folder/'validation_comparison.csv',index=False)
    pd.concat([r.by_group.assign(setting_id=k) for k,r in results.items()]).to_csv(folder/'validation_by_reason.csv',index=False)
    graph_result=results[graph_id];better=graph_result.summary['f1_samples']>original.summary['f1_samples']+1e-12
    selection={'graph_cv_winner':graph_id,'knn_cv_winner':knn_id,'graph_validation_f1':graph_result.summary['f1_samples'],
        'current_validation_f1':original.summary['f1_samples'],'improved_official_validation':better,
        'selected_submission':graph_id if better else 'current_submission','test_used':False}
    write_json(folder/'submission_selection_locked.json',selection)
    def row_f1(result):
        y=result.predictions[[f'true_{r}' for r in REASONS]].to_numpy();d=result.predictions[[f'pred_{r}' for r in REASONS]].to_numpy()
        return 2*(y*d).sum(axis=1)/(y.sum(axis=1)+d.sum(axis=1))
    differences=row_f1(graph_result)-row_f1(original);rng=np.random.default_rng(42)
    draws=differences[rng.integers(0,len(differences),size=(5000,len(differences)))].mean(axis=1)
    write_json(folder/'paired_validation_uncertainty.json',{'delta_sample_f1':float(differences.mean()),
        'descriptive_95_percentile_interval':np.quantile(draws,[.025,.975]).tolist(),
        'warning':'Descriptive interval does not correct for model selection or nonrandom seeds.'})
    if better:
        test=pd.read_csv(dataset_path(root,'part1','test'));bundle=models[graph_id]
        probabilities,fallbacks=predict_geometry(bundle,test)
        decisions=binary_decisions(probabilities[list(REASONS)],dict.fromkeys(REASONS,.35),empty_set_policy='argmax')
        pred=pd.DataFrame(decisions,columns=REASONS);pred.insert(0,'ID',test.ID.to_numpy())
        sample=pd.read_csv(dataset_path(root,'part1','test').parent/'part1_sample_submission.csv')
        submission=build_submission(sample,pred);dest=folder/'candidate_submissions';dest.mkdir(exist_ok=True)
        validate_submission(submission,sample,1);submission.to_csv(dest/'part1_graph_candidate.csv',index=False)
        probabilities.to_csv(dest/'part1_graph_candidate_probabilities.csv',index=False)
        frames={'train':train,'val':validation,'test':test}
        lookup=build_reason_lookup(frames,{s:predict_geometry(bundle,f)[0][list(REASONS)].to_numpy() for s,f in frames.items()},
            dict.fromkeys(REASONS,.35),graph_id,empty_set_policy='argmax')
        lookup.to_csv(folder/'reason_lookup_candidate.csv',index=False)
        filled=train.copy();q=bundle.get('pseudo_q')
        if q is not None:
            masked=train.loc[~observed,['ID']].copy()
            for j,r in enumerate(REASONS):masked[f'prob_{r}']=q[:,j]
            masked.to_csv(folder/'masked_training_graph_probabilities.csv',index=False)
            targets=binary_decisions(q,dict.fromkeys(REASONS,.35),empty_set_policy='argmax')
            filled.loc[~observed,list(REASONS)]=targets
            assert np.array_equal(filled.loc[observed,list(REASONS)],train.loc[observed,list(REASONS)])
            filled.to_csv(folder/'part1_train_graph_augmented.csv',index=False)
    write_json(folder/'summary.json',selection)
    write_json(folder/'completed_checksums.json',{str(p.relative_to(folder)):file_sha256(p) for p in folder.rglob('*') if p.is_file()})
    return selection


if __name__ == "__main__":
    from replay import main
    main(default_study='graph')
