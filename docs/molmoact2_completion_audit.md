# MolmoAct2 Completion Audit

Objective:

```text
Investigate allenai/molmoact2; decide whether the old artificial ACT
two-camera-position dataset can be used; make a plan; run the model locally on
GPU; try model control in sim; prepare Brev fine-tuning.
```

## Checklist

| Requirement | Current evidence | Status |
| --- | --- | --- |
| Investigate upstream MolmoAct2 | `molmoact2/check_finetune_readiness.py` inspects `allenai/molmoact2`, `allenai/lerobot@molmoact2-hf-inference`, and `allenai/MolmoAct2-SO100_101`. Current run reports the public MolmoAct2 LeRobot wrapper is inference-only. | Done |
| Pick the correct checkpoint | Docs and manifest use `allenai/MolmoAct2-SO100_101`, `norm_tag=so100_so101_molmoact2`, 6D absolute joint-pose actions. | Done |
| Compare old artificial ACT two-camera data | `docs/molmoact2_artificial_dataset_compatibility.md` records the decision: old ACT sim data is historical only, not a collaborator dependency, and mismatches MolmoAct2 because it is 10 Hz, 4D end-effector state, and 4D end-effector delta action. | Done |
| Preserve the actionable data plan | `docs/so100_vla_data_collection_guide.md` and `docs/molmoact2_data_collection.md` specify one fixed 3D RGB camera as `observation.images.front`, 30 Hz, 6D calibrated current joint state, and 6D absolute target joint action. | Done |
| Run current local GPU inference | Fresh command on this repo: `.venv/bin/python molmoact2/test_on_lerobot_frame.py --dataset-repo-id carmensc/record-test-screwdriver --frame 30 --run-model --device cuda --dtype bfloat16 --output outputs/molmoact2/carmen_current_one_frame_inference.json`. It loaded the checkpoint on the local RTX 4090 and returned `Actions shape: (1, 30, 6)`. | Done |
| Try model control in sim | Historical ignored outputs exist under `outputs/molmoact2/mujoco_zero_shot_25_angle.json` and `outputs/molmoact2/mujoco_zero_shot_25_top_wrist.json`, both 25-step zero-shot MuJoCo attempts with success rate 0.0. The committed repo intentionally no longer includes the private HW3 simulator runner, so collaborators cannot reproduce this from the repo alone. | Partially done |
| Prepare Brev fine-tuning workflow | `cluster/brev/` contains the SSH/rsync/uv workflow. `cluster/brev/.env.brev.template` defaults to `BREV_INSTANCE_NAME=mw-newton-dev`, with separate `/home/nvidia/code/vla_picknplace` and `/home/nvidia/logs/vla_picknplace` paths. | Done |
| Verify Brev can use the Newton instance | With `BREV_INSTANCE_NAME=mw-newton-dev`, `.venv/bin/python molmoact2/check_finetune_readiness.py --dataset-repo-id carmensc/record-test-screwdriver --skip-ranges` reports `OK brev: SSH to configured Brev instance 'mw-newton-dev' works`. | Done |
| Attempt real Brev fine-tuning | Not launched. The readiness gate still blocks because public upstream MolmoAct2 training support is inference-only. Without `--skip-ranges`, the old Carmen diagnostic dataset also blocks on joint range mismatch. | Blocked |

## Current Blockers

```text
1. Public MolmoAct2 fine-tuning code is still not available.
2. The old Carmen diagnostic dataset has joint range/calibration mismatches.
3. The original HW3/private-sim control path is not reproducible from this repo
   by design, because collaborators should not depend on that private checkout.
```

## Completion Decision

Do not mark the original goal complete yet.

The investigation, data decision, collection plan, current local GPU inference,
and Brev plumbing are complete. A real Brev fine-tune is blocked on upstream
trainability or an approved public local training recipe. The sim-control
attempt exists only as historical ignored output, not as a committed,
collaborator-reproducible workflow.
