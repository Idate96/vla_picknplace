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

.venv/bin/python molmoact2/check_finetune_readiness.py \
  --dataset-repo-id <hf_user>/<dataset> \
  --output-json outputs/molmoact2/readiness.json

.venv/bin/python molmoact2/summarize_readiness.py \
  outputs/molmoact2/readiness.json \
  --strict-exit-code

.venv/bin/python molmoact2/check_collection_dataset.py \
  --dataset-repo-id <hf_user>/<dataset> \
  --output-json outputs/molmoact2/collection_preflight.json

.venv/bin/python molmoact2/verify_molmoact2_artifacts.py
```

Use `check_collection_dataset.py` right after a pilot recording. It checks the
SO100/SO101 collection contract without requiring Brev or upstream fine-tuning
support, including loading sampled `observation.images.front` frames to catch
missing or blank camera video early.

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

Run the optional public MuJoCo SO101 arm physics smoke:

```bash
.venv/bin/python molmoact2/simulate_mujoco_so101.py \
  --model-output outputs/molmoact2/one_frame_inference.json \
  --output outputs/molmoact2/mujoco_so101_smoke.json
```

This downloads the pinned RobotStudio SO101 MJCF/mesh assets under
`outputs/molmoact2/so101_mujoco_assets/` and steps the arm physics. It still
does not include a screwdriver object, camera feedback, or a task-success
metric.

Run a one-step closed-loop MolmoAct2 sim rollout:

```bash
.venv/bin/python molmoact2/rollout_mujoco_so101.py \
  --rollout-steps 1 \
  --actions-per-inference 1 \
  --device cuda \
  --dtype bfloat16 \
  --output outputs/molmoact2/closed_loop_molmo_one_step.json \
  --frames-dir outputs/molmoact2/closed_loop_molmo_frames
```

This renders a fixed simulated front camera view with the SO101 arm, a simple
screwdriver proxy, and a target area, calls MolmoAct2 on that image/state, and
executes the first returned target in MuJoCo. It is still a smoke test, not a
validated success metric.

Short multi-call closed-loop smoke:

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

`check_finetune_readiness.py` reads `cluster/brev/.env.brev` or the template
for `BREV_INSTANCE_NAME` when the variable is not exported, so the default path
uses the Newton `mw-newton-dev` SSH alias before Brev CLI.

As of 2026-05-08, this should still block before real training because Ai2's
public MolmoAct2 LeRobot wrapper is inference-only. Do not replace that with a
private simulator workaround.
