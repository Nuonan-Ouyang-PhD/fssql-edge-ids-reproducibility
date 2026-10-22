import unittest
import itertools
import numpy as np
import pandas as pd
from scripts.analyze_pi4b8_r6 import metric_columns, validate_cells, paired_statistics, POLICIES, CONDITIONS, SEEDS

class R6AnalysisTests(unittest.TestCase):
    def test_fpr_is_not_removed_by_count_filter(self):
        frame=pd.DataFrame({"tp":[1],"fp":[2],"fn":[3],"tn":[4],"fpr":[.3],"f1":[.5],"seed":[5101]})
        self.assertEqual(set(metric_columns(frame)),{"fpr","f1"})
    def test_complete_grid(self):
        cells=[dict(policy=p,condition=c,seed=s,accepted_run_id=f"{p}:{c}:{s}") for p,c,s in itertools.product(POLICIES,CONDITIONS,SEEDS)]
        validate_cells(cells)
        with self.assertRaises(ValueError): validate_cells(cells[:-1])
        with self.assertRaises(ValueError): validate_cells(cells[:-1]+[cells[0]])
    def test_exact_signflip_resolution(self):
        self.assertEqual(paired_statistics([1,2,3,4,5],[0]*5)["exact_signflip_p"],.0625)
    def test_identical_pairs(self):
        result=paired_statistics([1]*5,[1]*5)
        self.assertEqual(result["exact_signflip_p"],1.)
        self.assertEqual(result["ci95_low"],0.)
        self.assertIsNone(result["cohen_dz"])
    def test_missing_or_nonfinite_pairs_rejected(self):
        with self.assertRaises(ValueError): paired_statistics([1]*4,[1]*4)
        with self.assertRaises(ValueError): paired_statistics([1,1,1,1,np.nan],[1]*5)

if __name__=="__main__": unittest.main()
