# Pi4B8 corrected V5 R6 statistical summary

This report uses the 60 accepted cells in `PI4B8_FORMAL_EVIDENCE_FINAL_VALIDATION.json` only. The two original temperature-invalid attempts are excluded from primary metrics; their valid replacement attempts are used for all metrics of cells 21 and 45. No official-test data are used.

Run-level metrics are in `PI4B8_R6_RUN_LEVEL_METRICS.csv`. Matched-seed paired inference for FSSQL-R minus Safe-Greedy is in `PI4B8_R6_PAIRED_FSSQL_R_VS_SAFE_GREEDY.csv`. The paired test is the exact two-sided sign-flip test with five matched seeds per condition; bootstrap intervals are percentile 95% CIs for the paired difference.

The principal pattern is not a universal FSSQL-R advantage. In n16/B0 and n16/B1, mean F1 is essentially tied with Safe-Greedy while FSSQL-R has higher P95 end-to-end latency. In n16/B2, FSSQL-R is materially worse in F1, reward and FPR, with no compensating safety advantage. Threshold remains a strong baseline. Physical energy is reported as measured DC input-side energy and is not interpreted as an energy-saving claim.

All 60 cells, 36,000 windows, Pi raw artifacts and POWER-Z bindings passed the final evidence validator before this analysis.
