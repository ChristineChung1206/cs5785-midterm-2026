"""Behavioral checks for soft-label weighting and outcome feature isolation."""

import _bootstrap  # noqa: F401; add repository src for standalone discovery
import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from midterm_augmentation import logistic_parameters
from midterm_refinement import fit_binary_targets,fit_field_outcomes,predict_field_outcomes

class RefinementTests(unittest.TestCase):
    def test_binary_soft_targets_match_the_equivalent_weighted_hard_fit(self):
        gold_x=np.array([[-2.],[-1.],[1.],[2.]])
        gold_y=np.tile([0,0,1,1],(5,1)).T
        pseudo_x=np.array([[-1.5],[1.5]])
        q=np.tile([0,1],(5,1)).T
        models=fit_binary_targets(gold_x,gold_y,pseudo_x,q,C=.1)
        reference=LogisticRegression(**{**logistic_parameters(42),'C':.1}).fit(
            np.vstack([gold_x,pseudo_x]),[0,0,1,1,0,1],sample_weight=[1,1,1,1,.1,.1])
        for model in models:
            np.testing.assert_allclose(model.coef_,reference.coef_,atol=1e-6)
            np.testing.assert_allclose(model.intercept_,reference.intercept_,atol=1e-6)
    def test_zero_pseudo_weight_preserves_the_gold_only_fit(self):
        x=np.array([[-2.],[-1.],[1.],[2.]])
        y=np.tile([0,0,1,1],(5,1)).T
        gold=fit_binary_targets(x,y,np.zeros((2,1)),None,C=.1)
        augmented=fit_binary_targets(x,y,np.zeros((2,1)),np.full((2,5),.2),C=.1,pseudo_weight=0)
        for a,b in zip(gold,augmented):np.testing.assert_allclose(a.coef_,b.coef_,atol=1e-6)
    def test_invalid_soft_probabilities_are_rejected(self):
        x=np.array([[-1.],[1.]]);y=np.tile([0,1],(5,1)).T
        with self.assertRaises(ValueError):fit_binary_targets(x,y,np.array([[0.]]),np.full((1,5),1.1),C=.1)
    def test_outcome_predictions_ignore_forbidden_and_unused_columns(self):
        rows=[]
        for phase in ['I','II','III']:
            for i in range(8):
                rows.append({'ID':f'{phase}_{i}','trial_id':f't{phase}{i}','phase':phase,'label':i%2,
                    'criteria':'adult eligible active' if i%2 else 'adult eligible placebo',
                    'diseases_cleaned':'common disease','drugs':'common drug'})
        train=pd.DataFrame(rows);model,_=fit_field_outcomes(train)
        baseline=predict_field_outcomes(model,train)
        changed=train.copy();changed['label']=1-changed.label;changed['Safety']=1;changed['emb_0']=999;changed['start_year']=2099
        np.testing.assert_array_equal(baseline.p_success,predict_field_outcomes(model,changed).p_success)
        vocabulary={p:{f:dict(v.vocabulary_) for f,v in m['vectorizers'].items()} for p,m in model['models'].items()}
        changed['criteria']='totally_unseen_token'
        predict_field_outcomes(model,changed)
        for p,m in model['models'].items():
            for f,v in m['vectorizers'].items():self.assertEqual(v.vocabulary_,vocabulary[p][f])

if __name__=='__main__':unittest.main()
