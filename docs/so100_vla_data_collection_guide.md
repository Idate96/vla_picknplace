# SO100/SO101 Carmen Screwdriver VLA Collection Guide

This is the source-of-truth handoff guide for collecting the real Carmen
screwdriver dataset for:

```text
MolmoAct2-SO100_101
Pi0.5
SmolVLA
ACT
```

Use it when setting up the real SO100/SO101 robot collection. The short answer
is: recollect with the official LeRobot SO follower/leader recording path,
store absolute calibrated 6D joint targets, and use the single fixed 3D camera
RGB stream as `observation.images.front`.

## Practical Verdict

Use the current LeRobot SO recorder. It is sufficient for collection as long as
we use the SO100/SO101 follower/leader path and do not replace the LeRobot joint
action with an ACT-style end-effector delta.

The collection stack should be:

```text
driver/recorder: lerobot-record
robot:           so100_follower or so101_follower
teleop:          so100_leader or so101_leader
camera:          one fixed 3D camera RGB stream, named front
dataset schema:  LeRobot v3
fps:             30
main action:     absolute calibrated 6D joint target
```

That is enough for the dataset. It does not remove the separate MolmoAct2
fine-tuning-code blocker: the public `allenai/MolmoAct2-SO100_101` checkpoint
is the right checkpoint and its model card says it is intended for SO100/SO101
inference or further fine-tuning, but our current local readiness check still
blocks actual Brev fine-tuning until an approved trainable MolmoAct2 entrypoint
is available.

Do not use these as the main collection path:

```text
old carmensc/record-test-screwdriver as-is
the artificial ACT camera-ablation sim collector
custom 4D end-effector delta action logs
duplicated fake camera views from the single 3D camera
```

## Decision

Recollect the Carmen screwdriver dataset before serious MolmoAct2, Pi0.5, or
SmolVLA fine-tuning.

The existing `carmensc/record-test-screwdriver` dataset is useful as a
diagnostic reference, but it is not sufficient as the main VLA dataset yet:

```text
good:  LeRobot v3-style shape, 30 Hz, one front RGB camera, 6D state/action
bad:   joint ranges do not match the current SO100/SO101 calibration convention
risk:  likely old/different calibration, not just a camera issue
```

Do not train/deploy a robot policy from the old data unless the calibration is
verified or converted with known old/new calibration files.

If "latest driver" means the current LeRobot SO100/SO101 robot recorder, then
yes: use that. The LeRobot SO follower implementation uses the same motor names
we need, defaults body joints to degrees, and keeps the gripper on a `0..100`
range. The SO100 and SO101 follower configs are registered under
`so100_follower` and `so101_follower`.

Do not use the artificial ACT sim collector as the source format for the real
VLA dataset.

Use:

```text
lerobot-record
robot.type = so100_follower or so101_follower
teleop.type = so100_leader or so101_leader
```

Do not use:

```text
the old artificial ACT camera-ablation collector
custom 4D end-effector delta action logs as the main dataset
the old Carmen screwdriver data as-is for MolmoAct2/Pi0.5/SmolVLA
```

## Is The Old Screwdriver Dataset Sufficient?

No, not as the main training or deployment dataset.

It is useful for:

```text
checking task wording
checking camera framing examples
testing dataset loaders
debugging MolmoAct2 one-frame inference
comparing old/new calibration ranges
```

It is not sufficient for:

```text
MolmoAct2 fine-tuning
Pi0.5 fine-tuning
SmolVLA fine-tuning
real-robot rollout
final ACT comparison against the new VLA policies, unless recalibrated/converted
```

The issue is not only camera placement. The old data likely uses a different
joint convention. A policy can look broken if shoulder lift is sign-flipped,
elbow zero is shifted, wrist roll wraps differently, or gripper scale differs.

## One Dataset To Collect

Collect one clean LeRobot v3 dataset that is the superset for all policies:

```text
observation.images.front   fixed 3D camera RGB stream
observation.state          float32[6] current joint positions
action                     float32[6] commanded absolute joint targets
task                       stable natural-language instruction
fps                        30
```

Use this joint order for both `observation.state` and `action`:

```text
shoulder_pan
shoulder_lift
elbow_flex
wrist_flex
wrist_roll
gripper
```

Use this calibration convention end to end:

```text
body joints: LeRobot SO follower use_degrees=True
gripper:     0..100 linear gripper position
```

The important rule:

```text
main action = absolute calibrated 6D joint target
```

Do not replace `action` with end-effector deltas.

This single dataset is the right base artifact. Downstream model-specific
variants should be derived offline:

```text
MolmoAct2: use the dataset as absolute 6D joint actions.
SmolVLA:   use the dataset as absolute 6D joint actions.
Pi0.5:     store absolute actions; enable relative actions only in the processor/training config if desired.
ACT:       train directly on joint actions, or derive EE-delta actions into a separate dataset/field.
```

## Exact Feature Contract

Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `observation.images.front` | RGB video | Single fixed 3D camera RGB stream |
| `observation.state` | `float32[6]` | Current calibrated SO arm joint positions |
| `action` | `float32[6]` | Absolute calibrated SO arm joint target command |
| `task` | string | Natural-language instruction, stable across the task |
| `timestamp` / episode indices | LeRobot v3 metadata | Normal LeRobot synchronization/indexing |

Optional fields:

| Field | Use |
| --- | --- |
| `observation.depth.front` | Keep only if easy; MolmoAct2-SO100_101 does not need it |
| camera intrinsics/extrinsics | Useful for debugging, not required by this checkpoint |
| `observation.ee_pose` | Derived later if ACT or analysis needs it |
| `action.ee_delta` | Derived later for ACT experiments, not the main `action` key |

The main `action` key must stay absolute 6D joint target because it is the
shared contract for MolmoAct2, Pi0.5, and SmolVLA.

## Camera

We only have one 3D camera view. That is acceptable.

Store its RGB stream as:

```text
observation.images.front
```

Keep the camera physically fixed for the full dataset. It should see:

```text
gripper
screwdriver
pickup area
place/target area
```

Do not create fake second-camera views by duplicating the same image.

If convenient, save depth, intrinsics, and extrinsics as extra metadata, but do
not depend on depth for `allenai/MolmoAct2-SO100_101`; that checkpoint has depth
reasoning disabled.

Camera acceptance criteria:

```text
the gripper is visible during approach, grasp, lift, and place
the screwdriver is visible before grasp
the target/place area is visible before release
the robot does not occlude the whole object at the critical grasp moment
camera mount does not move between episodes
exposure/focus are stable enough to see the screwdriver tip and handle
```

## What To Collect

Start with:

```text
pilot:      2-3 episodes
first run:  50 clean successful episodes
better run: 100-200 clean successful episodes
```

For screwdriver pickup/place, vary only the things we expect at deployment:

```text
screwdriver pose:     small x/y shifts, small rotations
pickup area:          same table/workspace
target/place area:    same target region, a few positions if needed
lighting/background:  mild variation only after the basic task works
robot start pose:     small natural teleop variation, not arbitrary starts
```

For the first fine-tune, prefer successful demos only. If failures are recorded,
keep them separate or explicitly label them; do not mix unlabeled failures into
the first training set.

Episode definition:

```text
start: robot near a normal pre-pick pose, gripper open, screwdriver visible
task: pick up the screwdriver and place/drop it in the target area
end: screwdriver released in target area, robot no longer moving significantly
```

Keep the task text boring and stable:

```text
pickup screwdriver
```

Only change the text if the real task changes. For example, do not alternate
between "pick up screwdriver", "grab tool", and "move screwdriver" for the same
behavior in the first dataset.

## Model-Specific Use

| Policy | Use this dataset? | Action format to store |
| --- | --- | --- |
| MolmoAct2-SO100_101 | Yes | absolute 6D joint target |
| Pi0.5 | Yes | absolute 6D joint target |
| SmolVLA | Yes | absolute 6D joint target |
| ACT | Yes | absolute 6D joint target, or derive an ACT-specific EE-delta dataset offline |

Pi0.5 relative actions are a training processor choice. The stored dataset still
uses absolute actions; LeRobot computes `relative = action - state` internally
and converts predictions back before sending commands.

SmolVLA on SO100/SO101 uses the normal LeRobot absolute joint action path.

MolmoAct2-SO100_101 expects absolute joint-pose control and returns actions in
robot scale.

Important model-specific implications:

```text
MolmoAct2: use allenai/MolmoAct2-SO100_101, norm_tag so100_so101_molmoact2.
Pi0.5: absolute dataset is still the base format; relative action is optional processor config.
SmolVLA: standard LeRobot SO100/SO101 path consumes the same dataset.
ACT: can train from this dataset directly, or use an offline derived EE-delta variant.
```

## Can We Save Both Joint Targets And EE Deltas?

Yes. Recommended layout:

```text
action                     absolute 6D joint target, for MolmoAct2/Pi0.5/SmolVLA
observation.ee_pose        optional derived EE pose
action.ee_delta            optional derived EE delta for ACT experiments
```

The main `action` key must remain absolute 6D joint target. If ACT needs a
different action representation, derive a separate ACT dataset or use a separate
field.

Do not make the MolmoAct2/Pi0.5/SmolVLA dataset depend on the ACT-specific
choice. ACT is the flexible one here; the VLA checkpoints are the stricter
constraint.

## Calibration Checklist

Before recording:

1. Calibrate SO100/SO101 follower and leader with the exact robot IDs that will
   be used for recording and evaluation.
2. Confirm teleop moves correctly across shoulder lift, elbow, wrist flex, wrist
   roll, and gripper.
3. Record a tiny calibration note with readings for 2-3 known physical poses:

```text
home/rest pose
arm extended-ish pose
gripper open/closed readings
```

4. Do not recalibrate midway through a dataset. If calibration changes, start a
   new dataset or record the calibration change explicitly.

Community reports around SO100/SO101 calibration mention opposite shoulder-lift
motion, 180-degree flips, encoder wrap near 0/4095, and gripper mismatch because
`use_degrees` does not apply to the gripper. These look like policy failures if
not caught during data collection.

Minimum calibration proof to write down before the 50-episode run:

```text
pose name             shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll gripper
home/rest             ...
arm extended-ish      ...
gripper open          ...
gripper closed        ...
```

This makes an offline conversion possible later if needed. Without this, old
data repair becomes guesswork.

## Recording Command Shape

Use the actual ports/camera index for the robot:

```bash
lerobot-record \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=<follower_id> \
  --robot.cameras="{ front: {type: opencv, index_or_path: <camera>, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so100_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=<leader_id> \
  --dataset.repo_id=<hf_user>/carmen_screwdriver_so100_vla \
  --dataset.single_task="pickup screwdriver" \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=30 \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --display_data=true
```

If using SO101 identifiers, use `so101_follower` / `so101_leader` consistently.

If the 3D camera is exposed through a RealSense backend instead of OpenCV,
change only the camera config. Keep the LeRobot feature name `front`:

```bash
--robot.cameras="{ front: {type: intelrealsense, serial_number_or_name: <serial>, width: 640, height: 480, fps: 30}}"
```

The important part is not the backend name. The important part is that
`observation.images.front` is synchronized RGB at 30 Hz and the robot state and
action are the calibrated 6D SO joints.

## Step-By-Step Collection Runbook

Use this order on collection day:

1. Identify ports.

```bash
lerobot-find-port
```

2. Calibrate the arm with the final ID.

If you only have one physical arm, calibrate it as the follower/robot arm and
reuse that same `--robot.id` for recording and rollout. You do not need a
leader calibration unless you actually have a second leader arm.

```bash
lerobot-calibrate \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=<follower_id>
```

If you also have a separate leader arm, calibrate it too:

```bash
lerobot-calibrate \
  --teleop.type=so100_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=<leader_id>
```

3. Teleoperate with the camera enabled and no recording. Fix camera placement
   before saving demos.

With a single arm, use a LeRobot-supported single-arm/manual control path if
available, or a custom controller that still sends commands through the LeRobot
SO follower interface. The logged `action` must be the absolute 6D joint target
sent to the calibrated arm, not the keyboard/joystick/task-space delta.

```bash
lerobot-teleoperate \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=<follower_id> \
  --robot.cameras="{ front: {type: opencv, index_or_path: <camera>, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so100_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=<leader_id> \
  --display_data=true
```

4. Write down the calibration proof table from 2-3 physical poses.

5. Record 2-3 pilot episodes with `lerobot-record`.

6. Run the validation script and visually inspect/replay the pilot.

7. Only if the pilot passes, record the first 50 clean successful demos.

8. If the first fine-tune partially works, scale to 100-200 clean demos with
   modest object/target variation.

## Validation After Pilot

After the first 2-3 episodes:

```bash
/home/lorenzo/git/ethz-course-2026/hw3_imitation_learning/.venv/bin/python \
  molmoact2/inspect_molmoact2.py \
  --dataset-repo-id <hf_user>/carmen_screwdriver_so100_vla
```

Expected:

```text
fps: 30
camera: observation.images.front
state/action: float32[6]
task: pickup screwdriver
joint ranges: coherent with current SO100/SO101 calibration
```

## Expected Range Sanity Checks

Use these as warning envelopes after the 2-3 pilot episodes. These are
MolmoAct2-SO100_101 training q01/q99 statistics, not mechanical safety limits.
They are useful for catching convention mistakes.

Expected approximate state ranges:

```text
shoulder_pan:   [-92,  91]
shoulder_lift:  [ 44, 185]
elbow_flex:     [ 38, 173]
wrist_flex:     [  6,  92]
wrist_roll:     [-63,  43]
gripper:        [  1,  44]
```

Reject the pilot or investigate before continuing if:

```text
shoulder_lift is mostly negative or moves opposite to teleop
elbow_flex is mostly negative or offset by about 90 degrees
wrist_roll is around the old Carmen range, roughly -160 to -90
gripper open/close is inverted or not on the expected 0..100-ish scale
any joint has sudden jumps from encoder wrap
observation.state and action use different units or joint order
```

Also replay or visualize the pilot. Check that:

```text
image sees full task
gripper state matches visual open/close
actions are close to next commanded joint targets
no sudden jumps from calibration wrap
```

Only then collect the full 50+ episodes.

Pilot rejection criteria:

```text
any joint changes direction opposite to teleop
shoulder_lift/elbow/wrist_roll ranges look like the old Carmen convention
gripper open/close is visually inverted or has an unexpected numeric range
camera misses the grasp or target release
episodes contain long idle chunks or manual resets inside the task window
timestamps/fps are not close to 30 Hz
```

## Old Carmen Dataset

The old dataset can possibly be repaired offline, but only if we can prove the
old convention. Range matching suggests something like:

```text
shoulder_lift ~= 90 - old
elbow_flex    ~= old + 90
wrist_roll    ~= old + 120
```

That is not enough by itself. Use offline conversion only if:

```text
old and new calibration conventions are known
same transform is applied to observation.state and action
converted episodes replay slowly and safely
converted ranges match real robot readings for known physical poses
dataset statistics are recomputed after conversion
```

Otherwise, recollect.

The cheapest safe path is to recollect. Offline conversion is only worth doing
after collecting the calibration proof poses above and comparing old/new readings
for the same physical robot poses.

## Artificial ACT Dataset Difference

The artificial ACT camera-ablation data is a different schema:

```text
fps:    10
state:  4D ee_x, ee_y, ee_z, gripper
action: 4D delta_ee_x, delta_ee_y, delta_ee_z, gripper
images: small sim render views
```

That is fine for the old ACT camera-placement experiment, but it is not a
MolmoAct2-SO100_101 collection format. For real Carmen collection, use the
LeRobot SO joint dataset described in this guide.

## Collection Day Checklist

Before recording:

```text
robot calibrated with final leader/follower IDs
camera rigidly mounted and named front
task area taped or otherwise repeatable
2-3 known physical calibration poses written down
teleop direction checked for all joints
gripper open/close numeric range checked
HF dataset repo name chosen
emergency stop and command limits ready
```

Pilot:

```text
record 2-3 episodes
run inspect_molmoact2.py
visualize replay
check camera framing and joint ranges
discard and fix setup if any pilot rejection criterion triggers
```

Main run:

```text
record 50 clean successful episodes
keep failures separate
do not recalibrate mid-run
do not move camera mid-run
do not change task wording mid-run
```

Scale-up:

```text
if first fine-tune works partially, collect 100-200 clean episodes
add modest screwdriver pose and target variation
add lighting/background variation only after the base task works
```

## Minimum Answer

For the next Carmen screwdriver collection:

```text
Use one fixed 3D front RGB camera.
Use LeRobot SO follower use_degrees=True.
Store observation.state as current 6D calibrated joint positions.
Store action as absolute 6D calibrated joint targets.
If there is only one arm, calibrate it as the follower and keep the same robot.id.
Collect 2-3 pilot episodes, validate ranges, then collect 50 clean successes.
Do not store EE deltas in the main action field.
```

## References

Checked on 2026-05-08 against the public docs/model cards and the local
LeRobot checkout at commit `ce24063e`.

```text
MolmoAct2-SO100_101 model card:
https://huggingface.co/allenai/MolmoAct2-SO100_101

MolmoAct2 upstream repository:
https://github.com/allenai/molmoact2

LeRobot real-world robot recording guide:
https://huggingface.co/docs/lerobot/main/il_robots

LeRobot SO-101 setup and calibration:
https://huggingface.co/docs/lerobot/main/so101

LeRobot v3 dataset recording:
https://huggingface.co/docs/lerobot/lerobot-dataset-v3

LeRobot action representations:
https://huggingface.co/docs/lerobot/action_representations

Pi0.5 model card:
https://huggingface.co/lerobot/pi05_base

SmolVLA model card:
https://huggingface.co/lerobot/smolvla_base
```
