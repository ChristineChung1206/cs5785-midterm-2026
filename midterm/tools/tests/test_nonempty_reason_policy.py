"""Check nonempty final decisions without changing historical thresholds or probabilities."""

import _bootstrap  # noqa: F401; add repository src for standalone discovery
import io
import unittest
import numpy as np
import pandas as pd
from midterm_evaluation import REASONS,evaluate_part1
from midterm_part1_finalization import binary_decisions,build_reason_lookup,build_submission
from midterm_part2_improvement import validate_submission

class NonemptyPolicyTests(unittest.TestCase):
    def test_empty_row_uses_largest_probability_and_stable_tie_order(self):
        scores=np.array([[.1,.2,.3,.2,.1],[.2,.2,.1,.1,.1]])
        actual=binary_decisions(scores,dict.fromkeys(REASONS,.35),empty_set_policy='argmax')
        np.testing.assert_array_equal(actual,[[0,0,1,0,0],[1,0,0,0,0]])
        np.testing.assert_array_equal(scores,[[.1,.2,.3,.2,.1],[.2,.2,.1,.1,.1]])
    def test_existing_multilabels_and_threshold_equality_are_unchanged(self):
        scores=np.array([[.35,.1,.5,.1,.34]])
        actual=binary_decisions(scores,dict.fromkeys(REASONS,.35),empty_set_policy='argmax')
        np.testing.assert_array_equal(actual,[[1,0,1,0,0]])
    def test_validation_uses_the_same_policy_as_submission(self):
        truth=pd.DataFrame([['x',0,1,0,0,0]],columns=['ID',*REASONS])
        scores=pd.DataFrame([['x',.1,.3,.2,.1,.1]],columns=['ID',*REASONS])
        raw=evaluate_part1(truth,scores,.35)
        final=evaluate_part1(truth,scores,.35,empty_set_policy='argmax')
        self.assertEqual(raw.summary['f1_samples'],0)
        self.assertEqual(final.summary['f1_samples'],1)
        self.assertEqual(final.summary['n_empty_set_fallbacks'],1)
        np.testing.assert_array_equal(raw.predictions[[f'prob_{r}' for r in REASONS]],final.predictions[[f'prob_{r}' for r in REASONS]])
    def test_lookup_keeps_observed_truth_and_flags_only_inferred_fallbacks(self):
        train=pd.DataFrame([['gold',1,0,0,0,0],['masked',-100,-100,-100,-100,-100]],columns=['ID',*REASONS])
        val=pd.DataFrame([['v',0,0,1,0,0]],columns=['ID',*REASONS]);test=pd.DataFrame({'ID':['test']})
        scores={s:np.tile([.1,.3,.2,.1,.1],(n,1)) for s,n in [('train',2),('val',1),('test',1)]}
        result=build_reason_lookup({'train':train,'val':val,'test':test},scores,dict.fromkeys(REASONS,.35),'source',empty_set_policy='argmax').set_index('ID')
        self.assertEqual(result.loc['gold','Safety'],1);self.assertEqual(result.loc['gold','Funding'],0)
        self.assertFalse(result.loc['gold','empty_set_fallback_applied'])
        self.assertTrue(result.loc['masked','empty_set_fallback_applied']);self.assertEqual(result.loc['masked','Funding'],1)
        self.assertEqual(result.loc['v','Efficacy'],1)
    def test_final_csv_round_trips_without_null_values(self):
        sample=pd.DataFrame({'ID':['x'],'Predicted':['Other']})
        decisions=binary_decisions(np.array([[.1,.1,.2,.1,.3]]),dict.fromkeys(REASONS,.35),empty_set_policy='argmax')
        pred=pd.DataFrame(decisions,columns=REASONS);pred.insert(0,'ID',['x'])
        result=build_submission(sample,pred)
        parsed=pd.read_csv(io.StringIO(result.to_csv(index=False)))
        self.assertFalse(parsed.isna().any().any());self.assertEqual(parsed.Predicted.tolist(),['Other'])
        self.assertEqual(validate_submission(parsed,sample,1)['status'],'PASS')
    def test_unknown_policy_is_rejected(self):
        with self.assertRaises(ValueError):binary_decisions(np.zeros((1,5)),dict.fromkeys(REASONS,.35),empty_set_policy='other')

if __name__=='__main__':unittest.main()
