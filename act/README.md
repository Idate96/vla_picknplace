# ACT Training

This folder contains the collaborator-facing ACT entry points. LeRobot itself is installed as a dependency instead of vendored into this repo.

## Sanity Check

```bash
.venv/bin/python act/check_act_dataset.py
```

The checker validates that the dataset exposes the fields ACT expects:

```text
observation.images.front
observation.state
action
```

It also asks `LeRobotDataset` for a default 100-step future action chunk, matching ACT's default `chunk_size`.

## Training

```bash
./act/train_act.sh
```

Defaults:

```text
DATASET_REPO_ID=carmensc/record-test-screwdriver
DATASET_REVISION=main
OUTPUT_DIR=outputs/train/act_screwdriver
DEVICE=cuda
STEPS=100000
BATCH_SIZE=8
VIDEO_BACKEND=torchcodec
```

Any extra CLI arguments are forwarded to `lerobot-train`.

For exact raw CLI examples, see:

```text
act/configs/act_screwdriver_smoke.args
act/configs/act_screwdriver_full.args
```
