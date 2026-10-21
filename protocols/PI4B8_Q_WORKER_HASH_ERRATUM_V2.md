# Pi4B true-n16 Q worker-hash erratum V2

Date: 21 October 2026

The corrected R5 formal matrix stopped before the second cell entered its
window loop. The B2 background process had not yet completed its
`sched_setaffinity({3})` call when the parent performed the fail-closed
affinity check. The failed R5 attempt and partial matrix are retained and no
scientific outcome was used to make this decision.

The worker now requires an explicit child-process readiness event after the
affinity call and before the parent independently verifies the observed CPU
set. This changes neither the frozen B1/B2 load definition nor any function
used by Q training: state construction, guard prediction, safe-set
construction, reward components and Q updates are unchanged.

The frozen development-only Q-training command was executed again with the
same configuration and inputs. All five `1280 x 4` Q-table files and the
300-row checkpoint CSV reproduced the V3 lineage byte for byte. No labels or
official-test data were accessed. The only Q-manifest field that changes is
the full-worker `worker_semantics_sha256` binding.

The reproduced lineage is stored under `q_training/pi4b8_true_n16_v4/` and is
the only Q lineage permitted in the next runtime-package revision. This is a
lineage repair with identical Q artifacts, not model tuning or a new
scientific experiment.
