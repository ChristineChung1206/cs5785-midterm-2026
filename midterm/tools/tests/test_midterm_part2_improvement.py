"""Behavioral checks for feature isolation, temporal preprocessing, and submission contracts."""

import _bootstrap  # noqa: F401; add repository src for standalone discovery
import unittest
import numpy as np
import pandas as pd
from midterm_part2_improvement import OutcomeMetadataEncoder, icd_categories, validate_submission

class ImprovementTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'ID':['a','b','c'], 'trial_id':['a','b','c'],
            'icdcodes':["[\"['C22.0', 'C22.0', 'E11.9']\"]", "['C22.1']", None],
            'drug_pathways':['R-A;R-A','R-A',None], 'matched_drugs':['drug','drug',None],
            'start_year':[2000.,2010.,np.nan]})
        self.catalog = pd.DataFrame({'reactome_id':['R-A','R-B'], 'pathway_name':['A','B'], 'pathway_level':[0,0]})
    def test_pooled_failure_sources_are_marked_mixed(self):
        from midterm_part2_baseline import _rate_row
        rows=pd.DataFrame({'prediction':[1,0,1],'source_type':['observed','predicted','unmatched'],
                           'prediction_is_training_resubstitution':[False,True,False]})
        self.assertEqual(_rate_row(rows,'I','all_failed_trials')['label_evidence'],'mixed')

    def test_nested_icd_codes_are_unique_categories(self):
        self.assertEqual(icd_categories(self.frame.loc[0,'icdcodes']), {'C22','E11'})
        self.assertEqual(icd_categories(None),set())
    def test_unknown_validation_tokens_cannot_expand_features(self):
        e=OutcomeMetadataEncoder(min_df=2).fit(self.frame,self.catalog)
        before=list(e.feature_names_)
        v=self.frame.copy();v['icdcodes']="['Z99.9']";v['drug_pathways']='R-B'
        matrix=e.transform(v).toarray()
        self.assertEqual(e.feature_names_,before)
        self.assertEqual(e.icd_vocabulary_,['C22'])
        self.assertEqual(matrix[:,0].sum(),0)
        self.assertEqual(e.pathways_.vocabulary_,['R-A'])
    def test_year_statistics_are_training_only_and_missingness_retained(self):
        e=OutcomeMetadataEncoder(min_df=1).fit(self.frame,self.catalog)
        self.assertEqual(e.year_median_,2005)
        v=self.frame.copy();v['start_year']=[np.nan,2022.,1990.]
        m=e.transform(v).toarray()
        self.assertEqual(m[0,-1],1)
        np.testing.assert_allclose(m[:,-2],e.year_scaler_.transform(np.array([[2005.],[2010.],[2000.]])).ravel())
        self.assertEqual(e.year_max_,2010)
    def test_forbidden_fields_cannot_change_metadata(self):
        e=OutcomeMetadataEncoder(min_df=1).fit(self.frame,self.catalog)
        v=self.frame.copy();v['ID']=['x','y','z'];v['trial_id']=['x','y','z'];v['label']=1;v['Safety']=1;v['emb_0']=999
        np.testing.assert_array_equal(e.transform(v).toarray(),e.transform(self.frame).toarray())
    def test_outcome_submission_rejects_invalid_scores(self):
        sample=pd.DataFrame({'ID':['a','b'],'label':[.5,.5]})
        for scores in [[np.nan,.3],[-.1,.4],[1.1,.4]]:
            with self.assertRaises(ValueError):validate_submission(sample.assign(label=scores),sample,2)
    def test_submission_requires_sample_order_and_unique_ids(self):
        sample=pd.DataFrame({'ID':['a','b'],'label':[.5,.5]})
        with self.assertRaises(ValueError):validate_submission(sample.iloc[::-1],sample,2)
        with self.assertRaises(ValueError):validate_submission(sample.assign(ID=['a','a']),sample,2)
        self.assertEqual(validate_submission(sample,sample,2)['rows'],2)
    def test_reason_submission_rejects_empty_null_and_invalid_labels(self):
        sample=pd.DataFrame({'ID':['a','b'],'Predicted':['Safety','Other']})
        self.assertEqual(validate_submission(sample.assign(Predicted=['Other','Safety Funding']),sample,1)['status'],'PASS')
        for text in ['', '   ', None, np.nan, 'Safety Safety','Unknown']:
            with self.assertRaises(ValueError):validate_submission(sample.assign(Predicted=[text,'Other']),sample,1)
if __name__=='__main__':unittest.main()
