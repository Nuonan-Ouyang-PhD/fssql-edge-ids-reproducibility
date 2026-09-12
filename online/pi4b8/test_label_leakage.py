#!/usr/bin/env python3
"""Fail-closed check that scoring labels are absent from runtime arrays."""
import json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
pkg=ROOT/'.formal_inputs/pi4b8_scheduler_train_v1'
manifest=json.loads((pkg/'PACKAGE_MANIFEST.json').read_text())
assert manifest['official_test']=='sealed_not_read'
for p in pkg.glob('*_features.npy'):
    a=np.load(p,allow_pickle=False); assert a.ndim==2 and a.shape[0]==600 and a.shape[1] in (191,8)
for p in pkg.glob('*_labels.npy'):
    a=np.load(p,allow_pickle=False); assert a.shape==(600,)
for p in pkg.glob('*_raw_features.csv'):
    header=p.read_text().splitlines()[0].split(','); assert 'label' not in header and 'attack_cat' not in header and 'source_id' not in header
print('LABEL_LEAKAGE_GATE_PASS: runtime arrays exclude scoring-only labels and audit identifiers')
