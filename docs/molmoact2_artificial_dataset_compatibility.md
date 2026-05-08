# MolmoAct2 And Old Artificial ACT Data

The old artificial ACT camera-ablation data is not part of this repo's
collaborator workflow. It depended on a private HW3 simulator checkout, so it is
kept only as historical context.

Do not use it as the MolmoAct2 fine-tuning or rollout dataset.

The relevant mismatch was:

```text
old ACT sim data:  two camera positions, 10 Hz, 4D end-effector state,
                   4D end-effector delta action
MolmoAct2 SO100:   one fixed front RGB camera, 30 Hz, 6D joint state,
                   6D absolute joint target action
```

Safe uses:

```text
historical ACT camera-ablation comparison
loader/model smoke after exporting a tiny LeRobot-shaped diagnostic dataset
visual sanity checks for camera framing ideas
```

Unsafe uses:

```text
Brev fine-tuning for MolmoAct2-SO100_101
real-robot deployment validation
claiming SO100/SO101 action compatibility from the old 4D delta labels
```

An offline conversion would only be acceptable if it creates a new dataset with
30 Hz frames, `observation.images.front`, calibrated 6D current joint state,
6D absolute target joint action, and recomputed LeRobot statistics. Without
those labels, it is not a simple camera remap.

Current path:

```text
Collect/recollect a real LeRobot SO100/SO101 dataset.
Use observation.images.front from the fixed 3D camera.
Use observation.state as calibrated current 6D joints.
Use action as calibrated absolute target 6D joints.
Run molmoact2/check_finetune_readiness.py on that dataset.
```

No HW3 simulator scripts are required or expected.
