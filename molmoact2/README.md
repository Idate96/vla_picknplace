# MolmoAct2

Repo-local utilities for `allenai/MolmoAct2-SO100_101`.

This workflow intentionally has no HW3/private-simulator dependency. Collaborators
only need this repo, public Python packages, LeRobot, Hugging Face access, and
the real LeRobot SO100/SO101 dataset.

## Dataset Contract

Use the SO100/SO101 checkpoint:

```text
model:        allenai/MolmoAct2-SO100_101
norm_tag:     so100_so101_molmoact2
state key:    observation.state
action key:   action
state/action: float32[6]
action:       absolute calibrated 6D joint target
joint order:  shoulder_pan, shoulder_lift, elbow_flex,
              wrist_flex, wrist_roll, gripper
image key:    observation.images.front
fps:          30
```

For the Carmen screwdriver collection, use one fixed 3D camera RGB stream as
`observation.images.front`. Do not synthesize fake extra cameras.

## Local Checks

```bash
.venv/bin/python molmoact2/inspect_molmoact2.py \
  --dataset-repo-id <hf_user>/<dataset>

.venv/bin/python molmoact2/test_on_lerobot_frame.py \
  --dataset-repo-id <hf_user>/<dataset> \
  --dry-run

.venv/bin/python molmoact2/check_finetune_readiness.py \
  --dataset-repo-id <hf_user>/<dataset>

.venv/bin/python molmoact2/verify_molmoact2_artifacts.py
```

Only run model inference on a GPU machine with enough memory:

```bash
.venv/bin/python molmoact2/test_on_lerobot_frame.py \
  --dataset-repo-id <hf_user>/<dataset> \
  --run-model \
  --output outputs/molmoact2/one_frame_inference.json \
  --device cuda \
  --dtype bfloat16
```

Treat returned actions as offline diagnostics until calibration and action
limits are checked.

Run the repo-local command-path sim smoke on that output:

```bash
.venv/bin/python molmoact2/simulate_joint_control.py \
  --model-output outputs/molmoact2/one_frame_inference.json \
  --output outputs/molmoact2/joint_control_smoke.json
```

This simulates the SO100/SO101 absolute joint-target command stream with
per-step relative clipping. It does not model contacts, object motion, camera
feedback, or task success.

## Brev

The canonical Brev path is under `cluster/brev/`:

```bash
cp cluster/brev/.env.brev.template cluster/brev/.env.brev
# edit BREV_INSTANCE_NAME

cluster/brev/setup_brev_env.sh

cluster/brev/submit_finetune_brev.sh \
  --dataset-repo-id <hf_user>/<dataset> \
  --gpus 1 \
  --time 24h \
  --train-command '<official MolmoAct2 fine-tuning command>'
```

As of 2026-05-08, this should still block before real training because Ai2's
public MolmoAct2 LeRobot wrapper is inference-only. Do not replace that with a
private simulator workaround.
