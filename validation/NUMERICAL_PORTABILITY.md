# Numerical portability check

Frozen artifact SHA-256 identities remain strict. A re-fit uses the same archived fit/margin observations and configuration but may differ in last floating-point bits across BLAS/platform versions. The test now uses absolute tolerance 1e-10 and relative tolerance 1e-12 for recomputed numeric fields, while categorical/schema fields match exactly. It separately requires identical held-out latency/thermal coverage flags and identical cap-admission flags for both devices. The tolerance is far below measured temperature and timing resolution; it does not permit changed guard decisions. Frozen artifacts are never overwritten.

This correction addresses the two old OLS serialization subcase failures (approximately 1e-14 differences), not an empirical or hardware re-run.
