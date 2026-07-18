# BdShield Paper Model Matched-Pair Bundle

This private package runs one matched backdoor/clean Competition Core experiment for
`{{BASE_MODEL}}`. It trains two LoRA Adapters, runs the backdoor quality gate, scans the full
tokenizer vocabulary in {{SHARDS}} resumable shards, probes {{CANDIDATES}} candidates, saves
soft-trigger replay artifacts, and builds a verified return ZIP.

Read `START_HERE_AI.md` first. The one-command entry point is `RUN_TEAM_PAIR.cmd`.

The source package contains a backdoor-training configuration, but no model weights, training
samples, local reports, or unpublished paper. Keep it private.

Thresholds are not fitted on the teammate computer. The detection configuration uses family
support {{COLLECTION_FLOOR}} only as a broad candidate-collection floor. Raw candidate evidence
is returned to the captain, who versions global development thresholds after collection.

A success ZIP cannot be generated without both final Adapter weight files. Every returned file
is listed with its size and SHA256 in `submission_manifest.json`, then the completed ZIP is
opened and verified before `RETURN_READY` is printed.
