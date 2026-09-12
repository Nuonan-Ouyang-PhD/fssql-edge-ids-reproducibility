# Official UNSW-NB15 one-shot final evaluation

All 82,332 official test rows were transformed with the frozen portable preprocessor and evaluated once with frozen validation thresholds. No retraining, threshold selection, Q/guard/workload/policy changes were performed. These results are independent detector-generalization evidence and are separate from Pi4B8 scheduler-development online results.

- FISVDD: F1=0.710177, precision=0.550600, recall=1.000000, FPR=1.000000, AUROC=0.535053, PR-AUC=0.621889
- LUCID: F1=0.824033, precision=0.702045, recall=0.997331, FPR=0.518595, AUROC=0.862063, PR-AUC=0.846851
- TinyDL: F1=0.868649, precision=0.780178, recall=0.979749, FPR=0.338216, AUROC=0.970411, PR-AUC=0.978446
- OI-SVDD+AS-ELM: F1=0.852117, precision=0.743729, recall=0.997485, FPR=0.421108, AUROC=0.951306, PR-AUC=0.958379
