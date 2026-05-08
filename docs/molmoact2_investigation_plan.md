# MolmoAct2 Investigation Plan

Target checkpoint:

```text
allenai/MolmoAct2-SO100_101
```

## Findings

The public `allenai/molmoact2` repo points users to model cards, datasets, and
an Ai2 LeRobot branch. As of 2026-05-08, the public MolmoAct2 LeRobot wrapper is
inference-only: training hooks are not implemented.

`MolmoAct2-SO100_101` is still the right checkpoint for our SO100/SO101 robot:

```text
norm_tag: so100_so101_molmoact2
control_mode: absolute joint pose
state/action: 6D shoulder_pan, shoulder_lift, elbow_flex,
              wrist_flex, wrist_roll, gripper
action_horizon: 30
```

## Dataset Decision

Do not use private HW3/artificial simulator data as a dependency. Collaborators
should work only from `~/git/vla_picknplace` plus public dependencies and the
real LeRobot dataset.

The target Carmen screwdriver dataset must be recollected or validated as:

```text
observation.images.front: fixed 3D camera RGB stream
observation.state:       current calibrated 6D SO100/SO101 joints
action:                  absolute calibrated 6D SO100/SO101 target joints
fps:                     30
task:                    pickup screwdriver
```

Use:

```text
docs/so100_vla_data_collection_guide.md
docs/molmoact2_data_collection.md
```

## Local Checks

```bash
.venv/bin/python molmoact2/inspect_molmoact2.py \
  --dataset-repo-id <hf_user>/<dataset>

.venv/bin/python molmoact2/test_on_lerobot_frame.py \
  --dataset-repo-id <hf_user>/<dataset> \
  --dry-run

.venv/bin/python molmoact2/check_finetune_readiness.py \
  --dataset-repo-id <hf_user>/<dataset>
```

GPU inference is diagnostic only:

```bash
.venv/bin/python molmoact2/test_on_lerobot_frame.py \
  --dataset-repo-id <hf_user>/<dataset> \
  --run-model \
  --output outputs/molmoact2/one_frame_inference.json \
  --device cuda \
  --dtype bfloat16
```

Public sim/control smoke:

```bash
.venv/bin/python molmoact2/simulate_joint_control.py \
  --model-output outputs/molmoact2/one_frame_inference.json \
  --output outputs/molmoact2/joint_control_smoke.json
```

This exercises the returned absolute 6D joint-target horizon through a small
SO100/SO101 joint-space follower simulation with per-step relative clipping. It
does not prove screwdriver task success because it has no physics/contact
model; use the MuJoCo smoke below for arm physics.

Public MuJoCo arm-physics smoke:

```bash
.venv/bin/python molmoact2/simulate_mujoco_so101.py \
  --model-output outputs/molmoact2/one_frame_inference.json \
  --output outputs/molmoact2/mujoco_so101_smoke.json
```

This downloads the pinned public RobotStudio SO101 MuJoCo assets and steps the
arm physics. It still does not model the screwdriver object, task contacts,
camera feedback, or task success.

Closed-loop MolmoAct2 sim smoke:

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
screwdriver proxy, and a target area, feeds that image plus current joint state
to MolmoAct2, then executes the first predicted target in MuJoCo. Treat it as a
smoke test, not as a success-rate benchmark.

## Brev

Use the Newton-style direct VM workflow in `cluster/brev/`:

```bash
cp cluster/brev/.env.brev.template cluster/brev/.env.brev
cluster/brev/setup_brev_env.sh
cluster/brev/submit_finetune_brev.sh \
  --dataset-repo-id <hf_user>/<dataset> \
  --gpus 1 \
  --time 24h \
  --train-command '<official MolmoAct2 fine-tuning command>'
```

The submit script runs the readiness gate before launching. It should not run a
real fine-tune until `ssh mw-newton-dev` works and a trainable MolmoAct2 recipe
exists. A separate Brev CLI login is only needed if the SSH alias is missing or
stale.
