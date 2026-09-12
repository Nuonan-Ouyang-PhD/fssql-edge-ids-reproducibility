#!/usr/bin/env python3
"""Materialize the frozen Pi4B8 runtime package and verify it end-to-end."""
from pathlib import Path
import hashlib, json, shutil, subprocess, sys
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "online/pi4b8/deployment/pi4b8_formal_runtime_v4"
missing = []
entries = []

def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def add(src, rel, role, ref, expected=None):
    src = ROOT / src
    if not src.exists():
        missing.append({"artifact": rel, "expected_path": str(src), "expected_sha256": expected,
                        "referenced_by": ref})
        return
    got = digest(src)
    if expected and got != expected:
        missing.append({"artifact": rel, "expected_path": str(src), "expected_sha256": expected,
                        "observed_sha256": got, "referenced_by": ref})
        return
    dst = OUT / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
    entries.append({"path": rel, "bytes": dst.stat().st_size, "sha256": digest(dst),
                    "role": role, "source_path": str(src)})

cfg = json.loads((ROOT/"configs/pi4b8_online_experiment_v1.json").read_text())
qm = json.loads((ROOT/"q_training/pi4b8_q_v1/Q_TABLE_MANIFEST.json").read_text())
im = json.loads((ROOT/".formal_inputs/pi4b8_scheduler_train_v1/PACKAGE_MANIFEST.json").read_text())
if OUT.exists():
    print(f"refusing to overwrite existing package: {OUT}", file=sys.stderr); sys.exit(2)
OUT.mkdir(parents=True)
ref = "configs/pi4b8_online_experiment_v1.json"
add("online/pi4b8/pi4b8_online_worker.py", "pi4b8_online_worker.py", "online worker", ref)
for f in ["runtime/action_pool_numpy.py", "runtime/preprocessor_numpy.py", "runtime/risk_proxy_numpy.py"]:
    add(f, f, "runtime helper", ref)
add("preprocessing/preprocessor_portable.npz", "preprocessing/preprocessor_portable.npz", "preprocessor", ref)
add("risk_proxy/risk_proxy.npz", "risk_proxy/risk_proxy.npz", "risk proxy", ref)
for n in ["fisvdd_revision", "lucid_revision", "tinydl_revision", "oi_svdd_as_elm_revision"]:
    add(f"models/artifacts/{n}.npz", f"models/{n}.npz", "frozen detector model", ref)
for key, rel in [("latency_predictor", "guard/latency_predictor.csv"), ("thermal_predictor", "guard/thermal_predictor.json"), ("margins", "guard/guard_margins.json")]:
    add(cfg["guard"][key], rel, f"Pi4B8 {key}", ref, cfg["guard"]["hashes"].get(key+"_sha256"))
for seed in qm["seeds"]:
    fn = f"q_training/pi4b8_q_v1/tables/fssql_r_seed_{seed}.npz"
    expected = qm.get("tables", {}).get(f"fssql_r_seed_{seed}.npz", {}).get("sha256")
    if not expected:
        # tolerate manifest layouts that store a list of records
        for x in qm.get("table_files", qm.get("files", [])):
            if isinstance(x, dict) and x.get("path", "").endswith(f"seed_{seed}.npz"): expected=x.get("sha256")
    add(fn, f"q_tables/fssql_r_seed_{seed}.npz", "Pi4B8 FSSQL-R Q table", "q_training/pi4b8_q_v1/Q_TABLE_MANIFEST.json", expected)
add("configs/pi4b8_online_experiment_v1.json", "config/pi4b8_online_experiment_v1.json", "formal configuration", ref)
add("configs/POWER_Z_HID_ACQUISITION_V1.md", "config/POWER_Z_HID_ACQUISITION_V1.md", "POWER-Z protocol", "configs/POWER_Z_HID_ACQUISITION_V1.md")
add("online/pi4b8/powerz_hid_capture.py", "powerz_hid_capture.py", "POWER-Z acquisition", "configs/POWER_Z_HID_ACQUISITION_V1.md")
add("device_calibration/inputs/scheduler_train_6000_label_lineage.csv", "scoring/scheduler_train_6000_label_lineage.csv", "audit-only label lineage", "device_calibration/inputs/scheduler_train_6000_label_lineage.csv")

conditions = ["natural", "robustness_B1", "robustness_B2"]
for c in conditions:
    for seed in [5101,5102,5103,5104,5105]:
        base=f"{c}_s{seed}"
        for kind, top, role in [("features", "inputs", "runtime features"), ("risk_features", "inputs", "runtime risk features"), ("labels", "scoring", "scoring-only labels")]:
            src=f".formal_inputs/pi4b8_scheduler_train_v1/{base}_{kind}.npy"
            add(src, f"{top}/{c}/{seed}/{kind}.npy", role, ".formal_inputs/pi4b8_scheduler_train_v1/PACKAGE_MANIFEST.json", im.get("files",{}).get(f"{base}_{kind}.npy",{}).get("sha256"))

for e in entries:
    if digest(OUT/e["path"]) != e["sha256"]: raise SystemExit("packaged hash mismatch")
for c in conditions:
    for seed in [5101,5102,5103,5104,5105]:
        np.load(OUT/f"inputs/{c}/{seed}/features.npy", allow_pickle=False)
        np.load(OUT/f"inputs/{c}/{seed}/risk_features.npy", allow_pickle=False)
        y=np.load(OUT/f"scoring/{c}/{seed}/labels.npy", allow_pickle=False); assert len(y)>=600
for n in ["fisvdd_revision","lucid_revision","tinydl_revision","oi_svdd_as_elm_revision"]: np.load(OUT/f"models/{n}.npz", allow_pickle=False)
np.load(OUT/"preprocessing/preprocessor_portable.npz", allow_pickle=False); np.load(OUT/"risk_proxy/risk_proxy.npz", allow_pickle=False)
for seed in [5101,5102,5103,5104,5105]:
    z=np.load(OUT/f"q_tables/fssql_r_seed_{seed}.npz"); assert z["q_values"].shape==(1280,4) and np.isfinite(z["q_values"]).all()
bad=[e for e in entries if any(x in e["path"].lower() for x in ["official_test","pi3b","near_cap"])]
if bad: raise SystemExit("forbidden artifacts packaged")
manifest={"status":"PI4B8_RUNTIME_V4_MATERIALIZED","official_test":"sealed_not_read","entries":entries,
          "entry_count":len(entries),"source_git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()}
mp=ROOT/"PI4B8_FORMAL_RUNTIME_PACKAGE_MANIFEST_V4.json"; mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
if missing:
    (ROOT/"PI4B8_RUNTIME_V4_MISSING_ARTIFACTS.json").write_text(json.dumps(missing,indent=2)+"\n")
    print("V4 MISSING ARTIFACTS"); print(json.dumps(missing,indent=2)); sys.exit(1)
print("PI4B8_RUNTIME_V4_LOCAL_GATE_PASS")
print(f"files={len(entries)} manifest_sha256={digest(mp)}")
