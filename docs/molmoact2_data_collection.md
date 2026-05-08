# MolmoAct2 SO100 Data Collection

For the broader policy-agnostic guide covering MolmoAct2, Pi0.5, SmolVLA, and
ACT, see:

```text
docs/so100_vla_data_collection_guide.md
```

Use the current LeRobot SO100/SO101 recorder for collection:

```text
lerobot-record + so100_follower/so100_leader
or
lerobot-record + so101_follower/so101_leader
```

That recorder is sufficient for the dataset as long as the main `action` field
is the calibrated absolute 6D SO joint target described below. Do not collect a
MolmoAct2 dataset through the old artificial ACT sim collector or by replacing
`action` with end-effector deltas.

Checkpoint:

```text
allenai/MolmoAct2-SO100_101
```

Record LeRobot v3 episodes at `30 Hz`.

## Per Frame

```text
observation.state float32[6]
action            float32[6]
```

Joint order for both:

```text
shoulder_pan
shoulder_lift
elbow_flex
wrist_flex
wrist_roll
gripper
```

`observation.state` is the current calibrated robot-scale joint position.

`action` is the absolute calibrated robot-scale joint-pose target command to
apply after this observation, not a delta.

## Cameras

Record real RGB video, synchronized with state/action.

Current hardware:

```text
observation.images.front
```

This matches the current Carmen dataset camera key. Keep using `front`; the
change is physical setup discipline, not the feature name. Use the RGB stream
from the single 3D camera and mount it rigidly so one fixed view sees:

```text
gripper
screwdriver
pickup area
place/target area
```

If the camera also provides depth, save aligned depth, intrinsics, and
extrinsics as extra metadata if convenient, but do not depend on depth for
`allenai/MolmoAct2-SO100_101`; this checkpoint's depth reasoning is disabled.

Do not duplicate the single RGB view as a fake second camera.

## Task Text

Store a natural-language `task` for every episode:

```text
pickup screwdriver
```

Use the same wording for all screwdriver demos unless the task actually
changes.

## Keep Raw

Do not downsample below `30 Hz`.

Do not resize the saved source videos.

Do not compress stationary frames.

Do not convert actions to end-effector deltas.

## Joint Calibration

For the new screwdriver dataset, use one LeRobot SO100/SO101 calibration
convention end to end.

Preferred convention:

```text
body joints: LeRobot SO follower use_degrees=True
gripper:    0..100 linear gripper position
```

This is the convention expected by current LeRobot SO follower deployment, and
it is close to the `MolmoAct2-SO100_101` action/state scale.

The old Carmen screwdriver dataset is not just a camera problem. Its values look
like an older or different SO calibration. Some joints can probably be repaired
offline with per-joint affine transforms, but do not treat it as one global
shift. At minimum the candidate repair is:

```text
shoulder_lift: sign flip + offset
elbow_flex:    offset
wrist_roll:    offset / wrap check
gripper:       range check
```

Only use an offline conversion if it passes these checks:

- Apply the same transform to both `observation.state` and `action`.
- Recompute dataset statistics after conversion.
- Replay or dry-run 2-3 converted episodes with clipped, slow commands.
- Confirm the converted ranges match the real robot's current readings for the
  same physical poses.

If those checks are not possible, recollect. Range matching alone is not enough;
wrong sign, wrong zero, or encoder wrap can produce plausible numbers with the
wrong physical motion.

Pi0.5 and SmolVLA use the same underlying LeRobot robot calibration during real
robot rollout. Pi0.5 may train with relative actions, but the processor converts
them back to absolute robot actions before sending commands. SmolVLA uses
absolute actions for the SO100/SO101 path. Therefore the dataset still needs
absolute joint targets in the calibrated robot convention.

## Compared With ACT Guide

Same as normal LeRobot ACT collection:

- Use `lerobot-record` / LeRobot v3 format.
- Record synchronized camera, state, action, timestamps, episode indices, and
  task metadata.
- Replay/visualize episodes after recording and discard bad demos.
- `50` good episodes is a reasonable first checkpoint.

Slightly stricter for MolmoAct2:

- Keep a superset dataset. ACT can ignore extra cameras later; MolmoAct2 benefits
  from preserving any real extra sensor streams.
- One real camera view is acceptable. If there is only one 3D camera, record its
  RGB view once and keep the view fixed; do not create a fake multi-view dataset.
- Preserve the exact natural-language `task`; ACT often treats it as secondary,
  MolmoAct2 conditions on it directly.
- Keep full source video resolution and original 30 Hz timing; resize or
  downselect only in training/eval transforms.
- Ensure `action` is absolute 6D joint-pose target commands. Do not store
  end-effector deltas or a custom sim action schema.
- Keep joint units and calibration identical between `observation.state`,
  `action`, training, and deployment.

Official LeRobot docs support this distinction: the v3 dataset guide records
SO-101 with camera video, state/action tables, task text, and raw images stored
without recording-time transforms; the processor guide says the learning
representation can be joint positions, absolute end-effector poses, or relative
end-effector deltas. For MolmoAct2-SO100_101, choose joint positions.

References:

```text
https://huggingface.co/docs/lerobot/lerobot-dataset-v3
https://huggingface.co/docs/lerobot/v0.4.3/processors_robots_teleop
https://huggingface.co/docs/lerobot/v0.5.1/il_robots
```

## Carmen Screwdriver Delta

Keep the existing high-level shape from `carmensc/record-test-screwdriver`:

```text
fps: 30
camera: observation.images.front
state/action: float32[6]
task: pickup screwdriver
```

Change/check these before collecting the next real dataset:

- Keep the camera key as `observation.images.front`; make it the fixed 3D
  camera RGB stream.
- Verify the SO100 calibration before recording. The previous dataset has
  shoulder lift, elbow, wrist roll, and gripper ranges far from MolmoAct2's
  SO100/SO101 q01/q99 ranges.
- Save `action` as the commanded absolute joint target in the same calibrated
  units and joint order as `observation.state`.
- If ACT also needs end-effector deltas, store them as an extra field or derive
  an ACT-specific dataset offline. Do not replace the main `action` key with
  end-effector deltas for MolmoAct2.
- Do a 2-3 episode pilot, run `inspect_molmoact2.py`, and only continue to
  `50+` demos once the range audit looks coherent.
- Keep the single 3D camera fixed. The view should prioritize task visibility
  over matching the old camera placement exactly.

## Episodes

Save successful demonstrations for training.

If a demo fails, either discard it or mark it separately; do not mix unlabeled
failures into the first fine-tuning set.

Start with `50` clean successful demos, then scale to `100-200`.

## Validation

After upload/local export:

```bash
.venv/bin/python molmoact2/inspect_molmoact2.py --dataset-repo-id <dataset>
```

The dataset should report:

```text
fps: 30
observation.state: float32[6]
action: float32[6]
at least one observation.images.* RGB video
task string present
```
