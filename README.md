# TSUSC-2026-07-0200 — Major Revision

This repository contains the frozen manuscript, supplement, response to reviewers, figures, references, and cover letter for the TSUSC Major Revision.

## What is included

- `main.tex` and `TSUSC-2026-07-0200-main.pdf`: revised manuscript.
- `supplement.tex` and `TSUSC-2026-07-0200-supp.pdf`: supplementary material.
- `response_to_reviewers.tex` and `TSUSC-2026-07-0200-response.pdf`: point-by-point response.
- `cover_letter.tex` and `TSUSC-2026-07-0200-cover-letter.pdf`: submission cover letter.
- `figures/`: frozen Pi4B8 robustness and POWER-Z energy figures.
- `references.bib`: bibliography.

## Devices and measurements

The study uses Raspberry Pi 3B+ and Raspberry Pi 4B 8GB hardware, with device-specific calibration, latency/thermal guards, controller timing, CPU telemetry, and active cooling. Physical energy is measured at the Raspberry Pi DC input path with a POWER-Z KM003C (serial 075356); it is not whole-site AC energy. The reported online deployment evidence uses matched seeds, 100-ms decision windows, 40-ms latency and 82-C thermal caps, and retained invalid-attempt provenance.

## Datasets

The primary intrusion-detection benchmark is UNSW-NB15, using frozen train/development lineage and a one-shot untouched official test evaluation. Supplementary pipelines use TON-IoT, CICIoT2023, and N-BaIoT where described in the manuscript. Raw datasets are not redistributed; obtain them from their official sources and respect their licenses.

## Algorithms

Detector actions: FISVDD, LUCID, TinyDL, and OI-SVDD+AS-ELM. Scheduling policies: FSSQL-R, Safe-Greedy, Threshold, Static-Light, Round-Robin, and Unshielded-Q. FSSQL-R applies a calibrated feasible-set shield followed by masked tabular Q-learning. Guard calibration is device-specific; evaluation uses frozen thresholds and no post-test tuning.

## Reproduction

Install a LaTeX distribution and compile:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
pdflatex supplement.tex
pdflatex response_to_reviewers.tex
pdflatex cover_letter.tex
```

The PDFs in this repository are the frozen compiled outputs. Reproducing physical results additionally requires the named Raspberry Pi hardware, cooling configuration, KM003C meter, frozen input artifacts, and the documented experiment protocols. Do not infer physical W/J from CPU utilisation, temperature, or a normalized proxy.

## Interpretation boundaries

The manuscript reports empirical calibrated feasible-set enforcement, not unconditional complete-pipeline hard safety. Q-learning benefit is device-/regime-dependent; Threshold is a strong baseline; controller overhead is part of deployment cost; and physical measurements do not establish a universal FSSQL-R energy advantage. Pi online F1 is a scheduler-development deployment result, while the official UNSW-NB15 evaluation is the independent detector-generalization result.
