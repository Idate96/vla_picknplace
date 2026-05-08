# MolmoAct2 Completion Audit

Objective:

```text
Prepare a repo-local MolmoAct2 data/check/Brev workflow for the Carmen SO100
screwdriver task.
```

## Current Status

The repo-local setup is prepared, but real fine-tuning is still blocked.

Concrete blockers:

```text
1. Brev auth/instance must be available in this shell.
2. Ai2's public MolmoAct2 LeRobot wrapper is currently inference-only.
3. The target real SO100/SO101 screwdriver dataset still needs to be provided.
```

## What Is Done

```text
Correct checkpoint selected: allenai/MolmoAct2-SO100_101
Dataset contract documented: 30 Hz, one front RGB image, 6D state, 6D action
Real collection guide written: docs/so100_vla_data_collection_guide.md
MolmoAct2 collection note written: docs/molmoact2_data_collection.md
Brev Newton-style workflow ported: cluster/brev/
Readiness gate added: molmoact2/check_finetune_readiness.py
Repo-local verifier added: molmoact2/verify_molmoact2_artifacts.py
```

## What Was Removed From The Required Path

```text
No HW3/private simulator dependency.
No artificial ACT sim export as a required data source.
No MuJoCo control smoke as a collaborator requirement.
```

The old artificial ACT data remains historical context only. It is not a
MolmoAct2 training input.

## Completion Gate

Only call the fine-tune path ready when:

```text
cluster/brev/.env.brev points at a reachable Brev VM, e.g. mw-newton-dev
cluster/brev/setup_brev_env.sh completes remotely
check_finetune_readiness.py has no BLOCKED rows for the target dataset
MolmoAct2 has a trainable public command or approved public local recipe
cluster/brev/submit_finetune_brev.sh launches that command and records a log
```
