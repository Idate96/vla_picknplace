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
  --device cuda \
  --dtype bfloat16
```

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
real fine-tune until Brev auth works and a trainable MolmoAct2 recipe exists.
