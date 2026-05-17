# SmolVLA Training

This folder contains the collaborator-facing SmolVLA entry points. LeRobot itself is installed as a dependency instead of vendored into this repo.

## Sanity Check

```bash
.venv/bin/python smolvla/check_smolvla_dataset.py
```

The checker validates that the dataset exposes the fields SmolVLA expects after the camera-key rename:

```text
observation.images.front  -> remapped to observation.images.camera1
observation.state
action
```

It also asks `LeRobotDataset` for the default future action chunk that matches `SmolVLAConfig.chunk_size`.

## Training

```bash
./smolvla/train_smolvla.sh
```

Defaults:

```text
DATASET_REPO_ID=carmensc/record-test-screwdriver
DATASET_REVISION=main
OUTPUT_DIR=outputs/train/smolvla_screwdriver
DEVICE=cuda
STEPS=20000
BATCH_SIZE=4
VIDEO_BACKEND=torchcodec
POLICY_PATH=lerobot/smolvla_base
RENAME_MAP={"observation.images.front":"observation.images.camera1"}
```

The script fine-tunes from the `lerobot/smolvla_base` checkpoint and renames the dataset's `observation.images.front` to `observation.images.camera1` so SmolVLA's expected camera key is satisfied. Any extra CLI arguments are forwarded to `lerobot-train`.

Cluster launchers live outside this folder. For Euler:

```bash
sbatch cluster/euler/train_smolvla.sbatch
```

For exact raw CLI examples, see:

```text
smolvla/configs/smolvla_screwdriver_smoke.args
smolvla/configs/smolvla_screwdriver_full.args
```
