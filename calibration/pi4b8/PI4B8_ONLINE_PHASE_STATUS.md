# Pi4B8 online phase status

Pi4B8 calibration is accepted and frozen at commit `9628ad4`. Retained held-out coverage is latency `1.000000`, thermal `0.998958`, and joint `0.998958`. The three uncovered thermal observations and their 0.975 by-cell results remain retained; no recalibration or Pi3B+ predictor/margin reuse is permitted.

Track A (Pi3B+ Stateful Transition Extension) is frozen `INVALID / NOT EXECUTABLE` after its outcome-blind service-model preflight failed the pre-registered accumulation-and-recovery criterion. No stateful policy runs may be executed.

The Pi4B8 online preregistration is frozen in `configs/pi4b8_online_experiment_v1.json` at commit `9d87594`. The Pi4B8 gate `online/pi4b8/pi4b8_online_gate.py` is development-only and deliberately refuses formal execution until a Pi4B8-specific input package and policy runner are hashed and frozen. This prevents accidental use of the Pi3B+ collector, Q/guard lineage, or an opened official test.

Non-energy preparation may continue while POWER-Z is installed manually. No physical W/J result is claimed until a timestamped meter CSV is aligned to formal run IDs.
