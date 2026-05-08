# vla_picknplace

Collaborator workspace for pick-and-place VLA/ACT experiments. The repo keeps the model training and dataset processing code small and reviewable. Datasets, generated previews, and model checkpoints are intentionally not committed.

Current diagnostic dataset:

```text
carmensc/record-test-screwdriver
```

Do not treat that old screwdriver dataset as the final MolmoAct2/Pi0.5/SmolVLA
training set. The real collection target is a recollected LeRobot SO100/SO101
dataset using the guide below:

```text
docs/so100_vla_data_collection_guide.md
```

## Layout

```text
act/              ACT training and dataset sanity checks
cluster/euler/    Euler Slurm launchers
data_processing/ LeRobot dataset visualization and inspection utilities
docs/             Push-scope and collaboration notes
molmoact2/        MolmoAct2 reference and one-frame test scripts
experiments/      Isolated short-lived experiments and runbooks
```

## Setup

The scripts use Hugging Face LeRobot. This repo pins LeRobot in
`requirements.txt`. For a local setup:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

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

## MolmoAct2

```bash
.venv/bin/python molmoact2/inspect_molmoact2.py --dataset-repo-id <hf_user>/<dataset>
.venv/bin/python molmoact2/test_on_lerobot_frame.py --dataset-repo-id <hf_user>/<dataset> --dry-run
.venv/bin/python molmoact2/verify_molmoact2_artifacts.py
.venv/bin/python molmoact2/check_finetune_readiness.py --dataset-repo-id <hf_user>/<dataset>
```

These scripts inspect `allenai/MolmoAct2-SO100_101`, show the model sample
image views, prepare a one-frame screwdriver-task test, verify the repo-local
Brev setup, and check whether fine-tuning is currently unblocked. This workflow
does not depend on HW3 or any private simulator repo.

Real-robot recollection guide:

```text
docs/so100_vla_data_collection_guide.md
```

Use that guide as the handoff document for the Carmen screwdriver collection.
It covers MolmoAct2, Pi0.5, SmolVLA, and ACT; it also states why the old
`carmensc/record-test-screwdriver` dataset is only a diagnostic reference.

MolmoAct2-specific notes:

```text
docs/molmoact2_data_collection.md
```

Investigation status and next-step plan:

```text
docs/molmoact2_investigation_plan.md
docs/molmoact2_command_sheet.md
docs/molmoact2_artificial_dataset_compatibility.md
docs/molmoact2_completion_audit.md
docs/molmoact2_brev_finetuning.md
```

## Experiments

Short-lived decision experiments live under `experiments/` so they do not look
like the main training path. The old sim-only camera placement ablation is now
archived as result context in `experiments/camera_placement/`; it is not an
operational dependency for MolmoAct2 or current ACT collection.
