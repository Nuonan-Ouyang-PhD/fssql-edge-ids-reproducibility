# Four-Action Pool Reconstruction Decision

Status: source-audit checkpoint; implementation and validation hyperparameters
are not yet frozen.

## Evidence boundary

The submitted manuscript retains the four action labels and their roles, but it
does not retain byte-identical model artifacts, full architectures, feature
maps, or training hyperparameters. Therefore every action below is a **Major
Revision reimplementation**. Historical manuscript metrics and artifacts from
other FSSQL/TNSM/TDSC projects are not inputs to this rebuild.

The shared UNSW-NB15 preprocessor is fitted only on `model_train`. Model and
operating-point choices may use `validation`; `scheduler_train` is reserved for
the scheduler and calibration workload. The official test file remains sealed
until all models, risk proxy, controller, workload, and guard parameters are
frozen.

## Source audit

| Submitted action label | Recoverable public basis | Reconstruction consequence |
|---|---|---|
| FISVDD | H. Jiang et al., *Fast Incremental SVDD Learning Algorithm with the Gaussian Kernel*, AAAI 2019; authors' public repository commit `e19e624bbe41fecac892a03f83848eda82cdac11` | Reimplement the published Gaussian-kernel incremental SVDD update. Training-set order, bandwidth, tolerances, normal-only fit subset, and threshold must be selected and disclosed for this revision. The upstream code is academic LGPLv3 and patent-pending; it is used as a readable reference, not silently copied as a historical artifact. |
| LUCID | R. Doriguzzi-Corin et al., IEEE TNSM 2020, DOI `10.1109/TNSM.2020.2971776`; authors' Apache-2.0 repository commit `960ef782efae9f0535e4eef0c6d83edd311cda23` | The public implementation expects packet-flow tensors built from PCAP, whereas this rebuild's frozen input is the official UNSW-NB15 tabular CSV. Preserve the lightweight single-convolution/global-max/binary-output design, but label the tabular feature-vector adaptation as revision-specific. |
| OI-SVDD+AS-ELM | E. Gyamfi and A. D. Jurcut, IEEE IoT Journal 2023, DOI `10.1109/JIOT.2022.3172393` | No author implementation was located in the source audit. Reimplement a disclosed two-stage online-SVDD plus adaptive sequential ELM cascade. The cited work places AS-ELM on an MEC server; this rebuild will execute both stages Pi-locally so the action is measurable under the submitted scheduler's device-local latency rule. This deployment difference must appear in the manuscript. |
| TinyDL | P. Fusco et al., TinyIDS, ICCSA 2024, DOI `10.1007/978-3-031-65223-3_5`; authors' public repository commit `32cdef3407664cdf151cabd6473f96111d647cdb` | The public model uses 30-sample windows, two 32-unit hidden layers, and a multiclass ToN-IoT output. Preserve the two-by-32 compact MLP motif, but adapt the input and output to the frozen UNSW-NB15 binary task. The repository exposes no license file, so it is treated as a design reference rather than redistributed code. |

## Why the revision implementations remain fit for the scheduler question

The paper's primary causal comparison is among scheduling policies operating on
the same heterogeneous, frozen action pool under the same calibrated safety
constraints. It does not require a claim that the lost historical classifiers
were recreated byte-for-byte. It does require that all four new actions are
fully disclosed, frozen before final-test access, and measured on each device.
Accordingly, action-specific validation quality, model size, latency, thermal
effect, and physical energy will be reported as new evidence; no submitted
number will be reused as if reproduced.

## Freeze gate

This file does not authorize final-test access. Before creating
`FINAL_TEST_LOCK.json`, the rebuild must add and freeze:

- exact architecture and feature input for all four actions;
- training and validation seeds, candidate grids, and selected values;
- serialized model hashes and portable Pi inference code;
- risk-proxy specification and hash;
- workload generator, controller, and per-device guard calibration protocol.
