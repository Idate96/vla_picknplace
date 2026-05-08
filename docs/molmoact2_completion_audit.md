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
| Compare old artificial ACT two-camera data | `docs/molmoact2_artificial_dataset_compatibility.md` records the decision: old ACT sim data is historical only, not a collaborator dependency, and must not be used as the MolmoAct2 fine-tuning or rollout dataset. It distinguishes safe diagnostic uses from unsafe fine-tuning/deployment uses and records the mismatch: two camera positions, 10 Hz, 4D end-effector state, and 4D end-effector delta action versus one fixed front RGB camera, 30 Hz, 6D joint state, and 6D absolute joint target action. `molmoact2/brev_finetune_manifest.json` also carries this as a machine-readable `artificial_act_dataset` block, and `molmoact2/verify_molmoact2_artifacts.py` has an `artificial ACT compatibility decision` guard for it. | Done |
| Remove private HW3 operational dependency | `experiments/camera_placement/` is now an archived result note only. The old Slurm launch wrappers that staged the private HW3 simulator were removed, and `molmoact2/verify_molmoact2_artifacts.py` now scans tracked text files for private-course path fragments. | Done |
| Preserve the actionable data plan | `docs/so100_vla_data_collection_guide.md` and `docs/molmoact2_data_collection.md` specify one fixed 3D RGB camera as `observation.images.front`, 30 Hz, 6D calibrated current joint state, and 6D absolute target joint action. | Done |
| Provide a post-collection dataset gate | `molmoact2/check_collection_dataset.py` preflights a newly collected LeRobot SO100/SO101 dataset without Brev or upstream training code. It checks metadata, sampled state/action rows, sampled nonblank `observation.images.front` RGB frames, and MolmoAct2 norm ranges. Metadata-only mode accepts the old screwdriver dataset shape, while the full command loads front RGB frames and blocks it on the known joint range/calibration mismatch. `molmoact2/summarize_readiness.py` converts JSON readiness/preflight reports into a launch/no-launch summary with next actions. `molmoact2/run_dataset_gate.py` runs collection preflight plus fine-tune readiness and writes a combined summary under `outputs/molmoact2/dataset_gate/`. `molmoact2/verify_molmoact2_artifacts.py` covers both preflight paths, the summary smoke, and the combined gate smoke. | Done |
| Run current local GPU inference | Fresh command on this repo: `.venv/bin/python molmoact2/test_on_lerobot_frame.py --dataset-repo-id carmensc/record-test-screwdriver --frame 30 --run-model --device cuda --dtype bfloat16 --output outputs/molmoact2/carmen_current_one_frame_inference.json`. It loaded the checkpoint on the local RTX 4090 and returned `Actions shape: (1, 30, 6)`. `molmoact2/gpu_sim_smoke_manifest.json` tracks this ignored output summary, and `molmoact2/verify_molmoact2_artifacts.py` validates the manifest plus the local output file when present. | Done |
| Try model control in sim | `molmoact2/simulate_joint_control.py` is a repo-local public joint-space command sim for the MolmoAct2 absolute 6D SO100/SO101 action horizon. Fresh command: `.venv/bin/python molmoact2/simulate_joint_control.py --model-output outputs/molmoact2/carmen_current_one_frame_inference.json --output outputs/molmoact2/carmen_joint_control_smoke.json`. It simulated 30 commands, clipped 6 steps, and ended at `[-3.30, 45.47, 35.40, 88.38, -65.58, 32.04]`. `molmoact2/simulate_mujoco_so101.py` also replayed the same horizon through the public RobotStudio SO101 MuJoCo arm model: 30 commands, 17 MuJoCo steps per command, no initial/control clipping, final LeRobot-unit state `[-3.35, 11.42, 31.44, 88.48, -65.61, 32.08]`. `molmoact2/rollout_mujoco_so101.py` then ran MolmoAct2 closed-loop on a simulated fixed front camera view. The current-schema one-step GPU run, `.venv/bin/python molmoact2/rollout_mujoco_so101.py --rollout-steps 1 --actions-per-inference 1 --width 320 --height 240 --num-steps 6 --device cuda --dtype bfloat16 --output outputs/molmoact2/closed_loop_molmo_schema_check.json --frames-dir outputs/molmoact2/closed_loop_molmo_schema_check_frames`, wrote `model_loaded=true`, `horizon_source=molmoact2_predict_action`, `horizon_shape=[30,6]`, nonblank image stats with `std=75.89`, `clipped_control_count=0`, and final state `[-6.32, 42.35, 35.53, 88.29, -76.10, 32.88]`. A stronger local RTX 4090 run, `.venv/bin/python molmoact2/rollout_mujoco_so101.py --rollout-steps 3 --actions-per-inference 2 --width 320 --height 240 --num-steps 6 --device cuda --dtype bfloat16 --output outputs/molmoact2/closed_loop_molmo_three_step.json --frames-dir outputs/molmoact2/closed_loop_molmo_three_step_frames`, completed three model calls, executed two targets per call, returned a 30x6 horizon each time, had `clipped_control_count=0`, wrote four camera frames, and ended at `[-6.61, 40.36, 31.37, 90.70, -87.56, 33.79]`. `molmoact2/gpu_sim_smoke_manifest.json` tracks these ignored output summaries, and the verifier cross-checks the local output files and frame count when present. This is still a smoke test, not a validated task-success benchmark. | Done for sim-control smoke |
| Prepare Brev fine-tuning workflow | `cluster/brev/` contains the SSH/rsync/uv workflow. `cluster/brev/.env.brev.template` defaults to `BREV_INSTANCE_NAME=mw-newton-dev`, with separate `/home/nvidia/code/vla_picknplace` and `/home/nvidia/logs/vla_picknplace` paths. | Done |
| Verify Brev can use the Newton instance | `ssh mw-newton-dev 'hostname && nvidia-smi -L'` reaches `brev-e5yzhjkxe` with 8 A100-SXM4-80GB GPUs. `cluster/brev/setup_brev_env.sh` synced the repo to `/home/nvidia/code/vla_picknplace`, ensured FFmpeg shared libraries for LeRobot video decoding, created `.venv`, installed requirements, passed import checks, and printed GPU visibility. With `BREV_INSTANCE_NAME` unset, local readiness reads `cluster/brev/.env.brev` and reports `OK brev: SSH to Brev instance 'mw-newton-dev' works from cluster/brev/.env.brev`, so Brev CLI login is not required when the SSH alias works. Remote command `ssh mw-newton-dev 'cd /home/nvidia/code/vla_picknplace && .venv/bin/python molmoact2/verify_molmoact2_artifacts.py'` passed, including the `Brev video decode setup` guard, remote Python imports, script compile checks, and MuJoCo dry-run smoke. | Done |
| Attempt real Brev fine-tuning | Guarded launch command tested: `cluster/brev/submit_finetune_brev.sh --dataset-repo-id carmensc/record-test-screwdriver --dry-run --train-command 'echo would train'`. It printed the Brev host/log plan, ran the local readiness gate, printed the readiness summary/no-launch decision, and exited nonzero before launch. Diagnostic-only launch shape also tested with `--allow-blocked-dry-run --readiness-report outputs/molmoact2/test_brev_submit_readiness.json`; it printed the same blockers, wrote machine-readable JSON with `ready=false`, then exited 0 without syncing or launching. `molmoact2/verify_molmoact2_artifacts.py` now executes this blocked dry-run when `cluster/brev/.env.brev` is present and asserts that it prints `Brev launch: NO` without `Synced to Brev.` or `Training started`. Blockers were old Carmen dataset range/calibration mismatch and public upstream MolmoAct2 training support still inference-only. | Blocked |

## Latest Audit Evidence

Checked on 2026-05-08 from `~/git/vla_picknplace`.

```text
git status --short --branch
## main...origin/main

git ls-remote https://github.com/allenai/molmoact2.git HEAD
d2e022b1c282c1f428d07d9abf61fdb1eaa0097a HEAD

git ls-remote https://github.com/allenai/lerobot.git refs/heads/molmoact2-hf-inference
c123084cf840c00af5c0833832fc58e590412851 refs/heads/molmoact2-hf-inference
```

The persisted diagnostic Brev report at
`outputs/molmoact2/test_brev_submit_readiness.json` has:

```text
ready=false
status=blocked
checks:
  model norm: OK
  dataset metadata: OK
  dataset ranges: BLOCKED
  brev: OK
  upstream fine-tune code: BLOCKED
blockers:
  dataset ranges
  upstream fine-tune code
```

The synced Brev tree also passes:

```bash
ssh mw-newton-dev \
  'cd /home/nvidia/code/vla_picknplace && .venv/bin/python molmoact2/verify_molmoact2_artifacts.py'
```

Direct local readiness with `BREV_INSTANCE_NAME` removed from the environment
now uses the repo Brev SSH default:

```bash
env -u BREV_INSTANCE_NAME \
  .venv/bin/python molmoact2/check_finetune_readiness.py \
  --dataset-repo-id carmensc/record-test-screwdriver \
  --skip-ranges \
  --output-json outputs/molmoact2/current_skip_ranges_reaudit.json
```

The resulting JSON has `brev: OK` from `cluster/brev/.env.brev` and blocks only
on `upstream fine-tune code` in this skip-ranges audit.

Summary command:

```bash
.venv/bin/python molmoact2/summarize_readiness.py \
  outputs/molmoact2/current_skip_ranges_reaudit.json \
  --strict-exit-code
```

It reports `Brev launch: NO`, names `upstream fine-tune code`, and exits 1 in
strict mode.

The same summary helper also handles the machine-readable manifest:

```bash
.venv/bin/python molmoact2/summarize_readiness.py \
  molmoact2/brev_finetune_manifest.json \
  --strict-exit-code
```

It reports both manifest blockers: the old Carmen diagnostic dataset range issue
and the inference-only upstream MolmoAct2 wrapper.

## Current Blockers

```text
1. Public MolmoAct2 fine-tuning code is still not available.
2. The old Carmen diagnostic dataset has joint range/calibration mismatches.
3. The committed closed-loop MuJoCo path includes a simple screwdriver proxy and
   camera feedback, and has run for multiple model-control steps, but it is not
   a validated task-success benchmark.
```

## Completion Decision

Do not mark the original goal complete yet.

The investigation, data decision, collection plan, current local GPU inference,
and Brev plumbing are complete. A real Brev fine-tune is blocked on upstream
trainability or an approved public local training recipe. The sim-control path
is now committed and collaborator-reproducible as a joint-space command smoke, a
public RobotStudio SO101 MuJoCo arm-physics smoke, and closed-loop MolmoAct2
MuJoCo smoke tests with a simple screwdriver proxy. It is not a full screwdriver
task-success benchmark.
