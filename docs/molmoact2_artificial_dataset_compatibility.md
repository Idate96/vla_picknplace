# MolmoAct2 And Old Artificial ACT Data

The old artificial ACT camera-ablation data is not part of this repo's
collaborator workflow. It depended on a private HW3 simulator checkout, so it is
kept only as historical context.

Do not use it for MolmoAct2 fine-tuning.

The relevant mismatch was:

```text
old ACT sim data: 10 Hz, 4D end-effector state, 4D end-effector delta action
MolmoAct2 SO100: 30 Hz, 6D joint state, 6D absolute joint target action
```

Current path:

```text
Collect/recollect a real LeRobot SO100/SO101 dataset.
Use observation.images.front from the fixed 3D camera.
Use observation.state as calibrated current 6D joints.
Use action as calibrated absolute target 6D joints.
Run molmoact2/check_finetune_readiness.py on that dataset.
```

No HW3 simulator scripts are required or expected.
