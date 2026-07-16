# ADR-0024: Blind Reference-Free Evaluation Must Use Separate Truth

- **Status**: Accepted
- **Date**: 2026-07-13
- **Decision makers**: Project team
- **Related**: ADR-0023

## Context

The reference-free detector must not read a training trigger, target output,
payload, poison data, attack configuration, or clean reference model.  Final
benchmark metrics nevertheless require post-hoc model labels and candidate
target recall.  Mixing that truth into the detector command would turn an
evaluation harness into an Oracle.

## Decision

1. The detector writes only model-derived candidates, soft-probe evidence,
   frozen calibration metadata, verdict, duration, and peak CUDA memory.
2. A separate evaluation-only truth manifest pairs a completed report's target
   artifact with its clean/backdoor label, blind split, attack family, and
   expected target markers.  It is not a detector input.
3. The aggregator accepts only `reference_free_soft_probe`, `formal_blind`,
   calibrated reports and rejects development truth records, missing truth,
   duplicate artifacts, or differing calibration profiles.
4. Aggregate outputs report model-level Precision, Recall, F1, FPR, PR-AUC,
   attack-family recall, candidate target recall, duration, and peak memory.

## Consequences

- Candidate recall is measured after detection without granting the detector
  access to the expected payload.
- Development models remain useful for calibration and implementation work but
  cannot silently enter final metrics.
- The aggregate is evidence only after the blind matrix is complete; this ADR
  does not itself establish detection efficacy.
