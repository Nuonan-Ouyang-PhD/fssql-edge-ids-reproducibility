# Pi4B true-n16 Q worker-hash erratum

Date: 21 October 2026

The corrected V5 R3 remote smoke stopped before inference because
`q_training/pi4b8_true_n16_v2/Q_TABLE_MANIFEST.json` bound the complete worker
file at SHA-256 `277ad5e1d07b46599b0f6b43cfd063a371d126d83fd6a831e26a3ea2016d4c0f`,
whereas the R3 worker had SHA-256
`81fddc60b327f03d3178aff86128d8f940ecda69ff76b570199fe3d0d99afdb2`.
The failed smoke and the immutable R3 package are retained; the hash check was
not bypassed and the old Q manifest was not edited.

The worker diff adds background-process affinity validation and additional
per-window evidence fields. It does not change the functions imported by the
Q trainer: state construction, latency and thermal guard prediction, safe-set
construction, reward components, or Q-update dynamics.

To verify this rather than infer it from the diff, the frozen training command
was executed again with the current worker and the same development-only
configuration and inputs. The five `1280 x 4` tables, their embedded metadata,
visit counts, and the 300-row checkpoint CSV reproduce the V2 artifacts byte
for byte. No label values or official-test data were accessed. The only Q
manifest field that changes is `worker_semantics_sha256`.

The reproduced lineage is stored under `q_training/pi4b8_true_n16_v3/` and is
the only Q lineage permitted in the next runtime-package revision. This is a
lineage repair with identical Q artifacts, not new tuning or a new scientific
experiment.
