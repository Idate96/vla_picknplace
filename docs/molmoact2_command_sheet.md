# MolmoAct2 Command Sheet

Run everything from:

```bash
cd ~/git/vla_picknplace
```

## Local Verify

```bash
.venv/bin/python molmoact2/verify_molmoact2_artifacts.py
```

Expected:

```text
All MolmoAct2 local artifacts verified.
```

## Dataset Readiness

Replace `<hf_user>/<dataset>` with the recollected Carmen screwdriver dataset.

```bash
.venv/bin/python molmoact2/inspect_molmoact2.py \
  --dataset-repo-id <hf_user>/<dataset>

.venv/bin/python molmoact2/check_finetune_readiness.py \
  --dataset-repo-id <hf_user>/<dataset>
```

Machine-readable readiness:

```bash
.venv/bin/python molmoact2/check_finetune_readiness.py \
  --dataset-repo-id <hf_user>/<dataset> \
  --output-json outputs/molmoact2/readiness.json
```

Collection preflight after a pilot recording:

```bash
.venv/bin/python molmoact2/check_collection_dataset.py \
  --dataset-repo-id <hf_user>/<dataset> \
  --output-json outputs/molmoact2/collection_preflight.json
```

For a local LeRobot dataset mirror:

```bash
.venv/bin/python molmoact2/check_finetune_readiness.py \
  --dataset-repo-id <hf_user>/<dataset> \
  --dataset-root /path/to/local/lerobot/dataset
```

## One-Frame Inference

Dry-run, no 5B model download:

```bash
.venv/bin/python molmoact2/test_on_lerobot_frame.py \
  --dataset-repo-id <hf_user>/<dataset> \
  --dry-run
```

GPU inference:

```bash
.venv/bin/python molmoact2/test_on_lerobot_frame.py \
  --dataset-repo-id <hf_user>/<dataset> \
  --run-model \
  --output outputs/molmoact2/one_frame_inference.json \
  --device cuda \
  --dtype bfloat16
```

## Joint-Space Sim Smoke

```bash
.venv/bin/python molmoact2/simulate_joint_control.py \
  --model-output outputs/molmoact2/one_frame_inference.json \
  --output outputs/molmoact2/joint_control_smoke.json
```

This is a public command-path smoke for the absolute 6D SO100/SO101 joint
targets. It is not a physics/contact/task-success simulator.

## MuJoCo SO101 Smoke

```bash
.venv/bin/python molmoact2/simulate_mujoco_so101.py \
  --model-output outputs/molmoact2/one_frame_inference.json \
  --output outputs/molmoact2/mujoco_so101_smoke.json
```

This uses the public RobotStudio SO101 MuJoCo model at a pinned commit. It
steps arm physics only; it does not simulate the screwdriver task.

## Closed-Loop Sim Smoke

```bash
.venv/bin/python molmoact2/rollout_mujoco_so101.py \
  --rollout-steps 1 \
  --actions-per-inference 1 \
  --device cuda \
  --dtype bfloat16 \
  --output outputs/molmoact2/closed_loop_molmo_one_step.json \
  --frames-dir outputs/molmoact2/closed_loop_molmo_frames
```

This renders a fixed simulated front camera view, calls MolmoAct2 on that
image/state, and executes the first returned target in MuJoCo. It is a
closed-loop smoke test, not a validated success benchmark.

Short multi-call variant:

```bash
.venv/bin/python molmoact2/rollout_mujoco_so101.py \
  --rollout-steps 3 \
  --actions-per-inference 2 \
  --width 320 \
  --height 240 \
  --num-steps 6 \
  --device cuda \
  --dtype bfloat16 \
  --output outputs/molmoact2/closed_loop_molmo_three_step.json \
  --frames-dir outputs/molmoact2/closed_loop_molmo_three_step_frames
```

## Brev Access

The default path reuses the Newton Brev SSH instance:

```bash
ssh mw-newton-dev "hostname && nvidia-smi -L"
```

If that SSH alias works, no separate Brev CLI login is needed for this repo.
Use `brev login` only when the SSH alias is missing or stale:

```bash
brev login --auth nvidia --skip-browser
brev refresh
brev ls --all
```

Token path:

```bash
export BREV_TOKEN='<token>'
brev login --auth nvidia --token "$BREV_TOKEN"
brev refresh
brev ls --all
```

## Brev Setup

```bash
cp cluster/brev/.env.brev.template cluster/brev/.env.brev
```

Edit `cluster/brev/.env.brev`:

```bash
export BREV_INSTANCE_NAME=mw-newton-dev
export BREV_CODE_DIR=/home/nvidia/code/vla_picknplace
export BREV_LOGS_DIR=/home/nvidia/logs/vla_picknplace
export BREV_DATA_DIR=/home/nvidia/data/vla_picknplace
```

Bootstrap:

```bash
cluster/brev/setup_brev_env.sh
```

## Brev Fine-Tune

Current expected gate:

```text
BLOCKED until Ai2 publishes a trainable MolmoAct2 fine-tuning entrypoint or we
approve a public local training recipe.
```

When that command exists:

```bash
cluster/brev/submit_finetune_brev.sh \
  --dataset-repo-id <hf_user>/<dataset> \
  --gpus 1 \
  --time 24h \
  --train-command '<official MolmoAct2 fine-tuning command>'
```

Diagnostic blocked dry-run only:

```bash
cluster/brev/submit_finetune_brev.sh \
  --dataset-repo-id carmensc/record-test-screwdriver \
  --dry-run \
  --allow-blocked-dry-run \
  --train-command 'echo would train'
```
