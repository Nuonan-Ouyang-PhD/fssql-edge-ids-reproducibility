# Corrected Pi 4B R6 analysis

These tables are derived only from the 60 accepted R6 cells. Original results in `results/pi4b8_r6/` remain unchanged.
The update explicitly includes FPR in all 18 condition-specific policy contrasts. The statistical unit is the matched run/seed (five pairs per condition), not records or windows. Percentile paired bootstrap intervals use 2,000 resamples and seed 20261022; exact two-sided sign-flip tests enumerate all 32 sign assignments. P-values are unadjusted descriptive results, not confirmatory findings across all endpoints. Cohen's dz is undefined when the paired differences have zero standard deviation and is left blank.
Q state-support analysis is a post hoc diagnostic using archived visits and Q values; it did not select or tune a policy. Measured energy uses actual meter timestamps within the ACK interval. No official test, model training or hardware execution is performed by this analysis.
