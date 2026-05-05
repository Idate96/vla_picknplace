# vla_picknplace

Collaborator workspace for pick-and-place VLA/ACT experiments. The repo keeps the model training and dataset processing code small and reviewable. Datasets, generated previews, and model checkpoints are intentionally not committed.

Current dataset target:

```text
carmensc/record-test-screwdriver
```

## Layout

```text
act/              ACT training and dataset sanity checks
cluster/euler/    Euler Slurm launchers
data_processing/ LeRobot dataset visualization and inspection utilities
docs/             Push-scope and collaboration notes
```

## Setup

The scripts use Hugging Face LeRobot. This repo was prepared against local LeRobot commit `ce24063e`. For a local editable setup:

```bash
git clone https://github.com/huggingface/lerobot.git ../lerobot
git -C ../lerobot checkout ce24063e
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e "../lerobot[training]"
uv pip install --python .venv/bin/python -r requirements.txt
```

If `lerobot-train` is already available in your environment, you can skip the editable LeRobot install.

## Dataset Check

```bash
.venv/bin/python act/check_act_dataset.py
```

Expected shape for ACT:

```text
observation.images.front -> (3, 480, 640)
observation.state        -> (6,)
action                   -> (100, 6) with default ACT chunk_size=100
```

## Visualize

```bash
.venv/bin/python data_processing/visualize_lerobot_dataset.py --episode 0 --stride 2
```

This writes generated files under `outputs/`, which is ignored by git.

## Train ACT

```bash
./act/train_act.sh
```

Useful overrides:

```bash
DEVICE=cuda STEPS=100000 BATCH_SIZE=8 ./act/train_act.sh
DATASET_ROOT=/path/to/local/lerobot/dataset ./act/train_act.sh
```

The script trains from the Hugging Face dataset by default. Set `DATASET_ROOT` only if you have a local LeRobot dataset mirror.

For a five-step smoke run:

```bash
STEPS=5 SAVE_FREQ=5 LOG_FREQ=1 NUM_WORKERS=2 BATCH_SIZE=2 OUTPUT_DIR=outputs/train/act_screwdriver_smoke ./act/train_act.sh
```

## Euler

```bash
sbatch cluster/euler/train_act.sbatch
```
