# Camera Placement Ablation

This is a narrow sim experiment to choose one camera for the single-camera ACT
policy. It is not the main ACT training run on the target pick-and-place data.

## Question

Compare two single-camera views while holding the expert, dataset size, ACT
configuration, cube randomization, bin randomization, and evaluation seeds fixed.

- `top_wrist`: wrist-mounted camera on the wrist body, so it moves with the
  gripper.
- `angle`: fixed third-person task camera, zoomed in on the cube, obstacle, and
  bin area.

The result should decide which camera to use next, not produce the final target
policy.

## Source Boundary

The experiment depends on the HW3 MuJoCo source tree for the simulator and
expert:

```text
hw3/sim_env.py
hw3/obstacle_expert.py
scripts/export_expert_lerobot.py
scripts/eval_lerobot_act.py
```

Those files are intentionally not vendored into this repo. This repo only keeps
the camera-ablation protocol, cluster wrappers, and result summarizer.

## Camera Setup

Inspected image pairs were generated locally at:

```text
/home/lorenzo/git/ethz-course-2026/hw3_imitation_learning/outputs/debug_setup/randomized_goal_camera_inspect/
```

Inspected reset poses:

| camera | mount | position | fovy |
| --- | --- | --- | --- |
| `top_wrist` | wrist body, fixed relative to wrist | `0 0.030 0` in the wrist body | `88` |
| `angle` | world camera targeting `task_view_target` | `0 0.16 0.54` | `38` |

## Dataset Protocol

- `256` saved successful demonstrations per camera.
- Train obstacle distribution for demonstrations.
- `cube_pos_std=0.006` m.
- `goal_pos_std=0.006` m.
- Expert failures are discarded; the saved dataset contains only successful
  demonstrations.
- Stationary dwell compression: keep `3` leading and `1` trailing frame per
  zero-XYZ action run.
- Image size: `96x96`.
- Policy state: end-effector xyz plus gripper only.
- Privileged cube, obstacle, and goal state is used only by the scripted expert.

The current generated datasets both saved `256/256` successful demonstrations
after `258` attempts, with `2` discarded expert failures. That means the saved
demo success rate is `100%`, while the rollout attempt success rate was about
`99.2%`.

Action-balance audit after dwell compression, identical for both camera
datasets:

| metric | value |
| --- | ---: |
| frames | `77092` |
| mean saved episode length | `301.14` |
| skipped stationary frames | `25473` |
| zero-XYZ action fraction | `3.16%` |
| moving action fraction | `96.84%` |
| positive-z lift fraction | `7.33%` |
| negative-z carry/lower fraction | `19.94%` |
| gripper-change events | `512` |

## ACT Protocol

- LeRobot ACT.
- `chunk_size=16`.
- `n_action_steps=1`.
- `temporal_ensemble_coeff=0.01`.
- `300000` training steps.
- Save every `50000` steps.
- Evaluate `100` episodes on train obstacles and `100` episodes on adversarial
  obstacles.

## Current Results

The `50000` step checkpoint already showed:

| camera | train | adversarial |
| --- | ---: | ---: |
| `top_wrist` | `90/100` | `25/100` |
| `angle` | `95/100` | `25/100` |

The `100000` step checkpoint showed:

| camera | train | adversarial |
| --- | ---: | ---: |
| `top_wrist` | `86/100` | `18/100` |
| `angle` | `94/100` | `26/100` |

The `150000` step checkpoint showed a ranking reversal:

| camera | train | adversarial |
| --- | ---: | ---: |
| `top_wrist` | `100/100` | `25/100` |
| `angle` | `46/100` | `18/100` |

The experiment was stopped after the `150000` checkpoint comparison. The
`300000` step training jobs were intentionally cancelled before completion, so
there is no final `300000` audit for this run. At the stopping point,
`top_wrist` was the stronger camera.

## Euler Paths

Default cluster paths used by the wrappers:

```text
/cluster/work/rsl/$USER/hw3_camera_ablation/src
/cluster/work/rsl/$USER/hw3_camera_ablation/lerobot
/cluster/scratch/$USER/hw3_camera_ablation/datasets_rand_goal_h6/<camera>
/cluster/work/rsl/$USER/hw3_camera_ablation/outputs/rand_goal_h6/<camera>
```

## Launch

Stage the HW3 source tree and LeRobot checkout under the paths above, then run:

```bash
sbatch -J act_rand_wrist experiments/camera_placement/euler/train_and_eval.sbatch top_wrist
sbatch -J act_rand_angle experiments/camera_placement/euler/train_and_eval.sbatch angle
```

Benchmark an intermediate checkpoint:

```bash
sbatch -J eval100_wrist experiments/camera_placement/euler/eval_checkpoint.sbatch top_wrist 100000
sbatch -J eval100_angle experiments/camera_placement/euler/eval_checkpoint.sbatch angle 100000
```

Summarize available results:

```bash
python experiments/camera_placement/summarize_results.py \
  --dataset-root /cluster/scratch/$USER/hw3_camera_ablation/datasets_rand_goal_h6 \
  --run-root /cluster/work/rsl/$USER/hw3_camera_ablation/outputs/rand_goal_h6 \
  --checkpoint-step 100000
```

## Decision Rule

Prefer the camera with the higher mean success across train and adversarial
evaluations at the same checkpoint. If the mean is tied, prefer the view with
better train-distribution success because that is the distribution used for the
expert demonstrations.
