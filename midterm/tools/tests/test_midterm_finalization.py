"""Check ensemble ID alignment and immutable historical evidence resolution."""

import _bootstrap  # noqa: F401; add repository src for standalone discovery
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest
import numpy as np
import pandas as pd
from midterm_finalization import blend_probabilities,predict_final_outcomes
from midterm_history import verify_snapshot,historical_path,augmentation_metadata_compatible
from midterm_evaluation import file_sha256


class FinalizationTests(unittest.TestCase):
    def test_environmental_cpu_warning_is_not_a_model_change(self):
        a={'training_code_sha256':'old','augmentation_module_sha256':'same','cluster_iterations':19,
           'baseline_control_max_probability_difference':5e-14,
           'cluster_warnings':['Could not find the number of physical cores for the following reason: unavailable'],
           'classifier_warnings':[]}
        b={**a,'training_code_sha256':'updated_checks','cluster_warnings':[],
           'baseline_control_max_probability_difference':0.0}
        self.assertTrue(augmentation_metadata_compatible(a,b))
        self.assertFalse(augmentation_metadata_compatible(a,{**b,'cluster_iterations':20}))

    def test_solver_warning_is_not_ignored(self):
        a={'baseline_control_max_probability_difference':0.,'cluster_warnings':[],'classifier_warnings':[]}
        self.assertFalse(augmentation_metadata_compatible(a,{**a,'classifier_warnings':['Optimizer did not converge']}))

    def test_large_control_prediction_difference_is_rejected(self):
        a={'baseline_control_max_probability_difference':0.}
        self.assertFalse(augmentation_metadata_compatible(a,{**a,'baseline_control_max_probability_difference':.01}))

    def test_blend_aligns_shuffled_component_ids(self):
        reference=pd.DataFrame({'ID':['a','b']})
        a=pd.DataFrame({'ID':['a','b'],'p_success':[.2,.8]})
        b=pd.DataFrame({'ID':['b','a'],'p_success':[.4,.6]})
        result=blend_probabilities(reference,[a,b],[.5,.5])
        self.assertEqual(result.ID.tolist(),['a','b'])
        np.testing.assert_allclose(result.p_success,[.4,.6])

    def test_blend_rejects_bad_weights_and_incomplete_ids(self):
        frame=pd.DataFrame({'ID':['a','b'],'p_success':[.2,.8]})
        for weights in [[1,1],[-.1,1.1],[np.nan,.5],[.5]]:
            with self.assertRaises(ValueError):blend_probabilities(frame,[frame,frame],weights)
        with self.assertRaises(ValueError):blend_probabilities(frame,[frame.iloc[:1],frame],[.5,.5])

    def test_final_bundle_cannot_silently_change_selected_weight(self):
        bundle={'kind':'equal_probability_blend','positive_class':1,'reason_features_used':False,'weights':[.2,.8]}
        with self.assertRaises(ValueError):predict_final_outcomes(bundle,pd.DataFrame({'ID':['a']}))

    def test_historical_original_is_verified_after_current_replacement(self):
        with TemporaryDirectory() as name:
            root=Path(name);(root/'history/old').mkdir(parents=True)
            original=root/'history/old/model';original.write_text('original')
            digest=file_sha256(original)
            (root/'model').write_text('new final model')
            (root/'history/manifest.json').write_text(json.dumps({'archived':{'model':{'archive_path':'history/old/model','sha256':digest}},'regenerated':{}}))
            report=verify_snapshot(root,{'model':digest})
            self.assertEqual(report['verified_archived'],1)
            original.write_text('corrupted')
            with self.assertRaises(AssertionError):verify_snapshot(root,{'model':digest})

    def test_unarchived_immutable_file_change_is_rejected(self):
        with TemporaryDirectory() as name:
            root=Path(name);p=root/'data.csv';p.write_text('original')
            digest=file_sha256(p);p.write_text('modified')
            with self.assertRaises(AssertionError):verify_snapshot(root,{'data.csv':digest})

    def test_history_cannot_escape_project(self):
        with TemporaryDirectory() as name:
            with self.assertRaises(ValueError):historical_path(name,'../outside')

    def test_only_recorded_generated_zip_can_be_excluded(self):
        with TemporaryDirectory() as name:
            root=Path(name);(root/'history').mkdir()
            manifest={'archived':{},'regenerated':{'deliverables/midterm_code.zip':{'before_sha256':'old'},'data.csv':{'before_sha256':'old'}}}
            (root/'history/manifest.json').write_text(json.dumps(manifest))
            result=verify_snapshot(root,{'deliverables/midterm_code.zip':'old'})
            self.assertEqual(result['excluded_regenerated_archive'],1)
            with self.assertRaises(FileNotFoundError):verify_snapshot(root,{'data.csv':'old'})


if __name__=='__main__':unittest.main()
