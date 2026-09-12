#!/usr/bin/env python3
import csv,json,hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from runtime.preprocessor_numpy import PortablePreprocessor
from runtime.action_pool_numpy import NumpyActionModel
R=Path(__file__).resolve().parent; OUT=R/'official_test_results'; OUT.mkdir(exist_ok=True)
src=R/'data/raw/UNSW_NB15_testing-set.csv'; rows=list(csv.DictReader(src.open(encoding='utf-8-sig',newline='')))
labels=np.array([int(x['label']) for x in rows],dtype=np.int8); features=[{k:v for k,v in x.items() if k not in ('id','attack_cat','label')} for x in rows]
X=PortablePreprocessor(R/'preprocessing/preprocessor_portable.npz').transform(features)
spec=[('FISVDD','fisvdd_revision',0.05),('LUCID','lucid_revision',0.05),('TinyDL','tinydl_revision',0.315),('OI-SVDD+AS-ELM','oi_svdd_as_elm_revision',0.065)]
out=[]
for label,name,threshold in spec:
 model=NumpyActionModel(R/f'models/artifacts/{name}.npz'); scores=model.predict_proba(X); pred=(scores>=threshold).astype(np.int8); tp=int(((pred==1)&(labels==1)).sum()); fp=int(((pred==1)&(labels==0)).sum()); fn=int(((pred==0)&(labels==1)).sum()); tn=int(((pred==0)&(labels==0)).sum()); precision=tp/(tp+fp) if tp+fp else 0; recall=tp/(tp+fn) if tp+fn else 0; f1=2*precision*recall/(precision+recall) if precision+recall else 0; fpr=fp/(fp+tn) if fp+tn else 0
 out.append({'detector':label,'threshold_frozen':threshold,'rows':len(rows),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':precision,'recall':recall,'f1':f1,'fpr':fpr,'auroc':float(roc_auc_score(labels,scores)),'pr_auc':float(average_precision_score(labels,scores))})
with (R/'OFFICIAL_TEST_RESULTS.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
payload={'status':'PASS_ONE_SHOT_COMPLETE','rows_evaluated':len(rows),'detectors':out,'test_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'threshold_source':'models/MODEL_HYPERPARAMETERS.json','official_test':'opened_once','no_tuning':True,'development_phase':'closed'}
(R/'OFFICIAL_TEST_VALIDATION.json').write_text(json.dumps(payload,indent=2)+'\n'); (R/'OFFICIAL_TEST_SUMMARY.md').write_text('# Official UNSW-NB15 one-shot final evaluation\n\nAll 82,332 official test rows were transformed with the frozen portable preprocessor and evaluated once with frozen validation thresholds. No retraining, threshold selection, Q/guard/workload/policy changes were performed. These results are independent detector-generalization evidence and are separate from Pi4B8 scheduler-development online results.\n\n'+ '\n'.join(f"- {x['detector']}: F1={x['f1']:.6f}, precision={x['precision']:.6f}, recall={x['recall']:.6f}, FPR={x['fpr']:.6f}, AUROC={x['auroc']:.6f}, PR-AUC={x['pr_auc']:.6f}" for x in out)+'\n')
print('OFFICIAL_TEST_ONE_SHOT_PASS',len(rows)); print(json.dumps(out,indent=2))
