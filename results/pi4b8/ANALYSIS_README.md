# Pi4B V4 historical result status

Status: `HISTORICAL_V4_PROVENANCE_ONLY`

The files in this directory reproduce arithmetic summaries from the retained
V4 logs. A 21 October 2026 code-to-paper audit found that the V4 worker did not
implement the declared true-`n=16` input, RiskProxy, Threshold, Safe-Greedy,
Static-Light, reward, joint thermal guard, or timing semantics, and the
available record does not prove that B1/B2 CPU load was applied.

Accordingly, these summaries are excluded from current Pi4B policy,
background-robustness, controller-latency and physical-energy comparison
claims. Technical run validity and arithmetic reproducibility do not make the
underlying scientific implementation valid. The original files remain
unchanged for provenance; no corrected result has been generated.

The official test was sealed during V4 collection and was subsequently opened
once for the fixed-threshold detector evaluation. It must not be opened again
or used to select repair parameters.
