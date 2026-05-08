# MolmoAct2 Brev Fine-Tuning Runbook

Status: blocked until trainable MolmoAct2 fine-tuning code is available.

Machine-readable handoff:

```text
molmoact2/brev_finetune_manifest.json
```

Canonical Brev workflow in this repo is now the Newton-style direct VM path:

```text
cluster/brev/.env.brev.template
cluster/brev/sync_code_brev.sh
cluster/brev/setup_brev_env.sh
cluster/brev/submit_finetune_brev.sh
```

Use the existing Newton Brev SSH alias when available, then use
`ssh`/`rsync`/remote `uv`. The actual VM setup/launch path mirrors Newton and
lives in `cluster/brev/`.

The top-level `allenai/molmoact2` repository includes an Ai2 LeRobot submodule
branch, `allenai/lerobot@molmoact2-hf-inference`. That branch does contain:

```text
src/lerobot/policies/molmoact2/
src/lerobot/scripts/lerobot_train.py
```

But the MolmoAct2 policy wrapper is inference-only: the config optimizer preset,
policy optimizer params, and policy `forward(...)` raise `NotImplementedError`.
Do not launch `lerobot-train --policy.type=molmoact2` until that changes or Ai2
publishes a separate training recipe.

The cached Hugging Face checkpoint is also not enough by itself to define a safe
fine-tuning recipe. We can call `predict_action(...)` for inference, but the
public files do not expose a documented training loss, optimizer preset, or
LeRobot training `forward(...)` contract for MolmoAct2. A local fine-tuning
script would require reverse-engineering private training internals, so it is
not treated as an approved workaround.

The readiness gate is deliberately conservative: if upstream fine-tuning code
cannot be inspected because of network/API limits, it reports `BLOCKED` instead
of `WARN`. Brev launch should only proceed when trainability is positively
verified.

## Preconditions

```text
ssh mw-newton-dev works, or Brev CLI login can refresh that SSH alias
HF_TOKEN with read/write access
recollected LeRobot v3 SO100/SO101 dataset
trainable MolmoAct2 fine-tuning script, or approved adapted LeRobot recipe
```

Only use Brev CLI login if the existing SSH alias is unavailable:

```bash
brev login --skip-browser
brev login --token <brev_or_nvidia_auth_token>
```

`brev login --token ...` is the noninteractive path if a token is available.
Clearing Brev auth is not the blocker when `ssh mw-newton-dev` works. The
submit script still runs the readiness gate before launch.

The target dataset should follow:

```text
observation.images.front: fixed 3D camera RGB stream
observation.state:       float32[6] current calibrated joint positions
action:                  float32[6] absolute calibrated joint targets
fps:                     30
task:                    stable language instruction
```

## Local Preflight

Run before uploading or training:

```bash
.venv/bin/python \
  molmoact2/inspect_molmoact2.py \
  --dataset-repo-id <hf_user>/<dataset>
```

Do not continue if state/action ranges are inconsistent with the deployed robot
calibration.

## Brev Skeleton

Once logged in, configure the Newton-style Brev env:

```bash
cp cluster/brev/.env.brev.template cluster/brev/.env.brev
```

Edit:

```bash
export CLUSTER_TYPE=brev
export BREV_INSTANCE_NAME=mw-newton-dev
export BREV_CODE_DIR=/home/nvidia/code/vla_picknplace
export BREV_LOGS_DIR=/home/nvidia/logs/vla_picknplace
export BREV_DATA_DIR=/home/nvidia/data/vla_picknplace
```

This intentionally reuses the Newton Brev VM while keeping this repo under its
own remote code/log/data directories.

Then:

```bash
cluster/brev/setup_brev_env.sh
```

This syncs the repo with rsync, installs `uv`, creates `.venv`, installs
LeRobot and `requirements.txt`, checks imports, and prints visible GPUs.

Verified on 2026-05-08 against `mw-newton-dev`: the setup script synced to
`/home/nvidia/code/vla_picknplace`, installed the remote Python environment,
passed the import check, and saw 8 A100-SXM4-80GB GPUs.

Remote repo verifier also passed on 2026-05-08:

```bash
ssh mw-newton-dev \
  'cd /home/nvidia/code/vla_picknplace && .venv/bin/python molmoact2/verify_molmoact2_artifacts.py'
```

That validates the Brev-side Python environment, script syntax, dependency
imports, and MuJoCo dry-run smoke. The verifier may report that the Brev SSH
self-check is not reachable when run inside the VM; that is not a fine-tune
blocker because local submission uses the `mw-newton-dev` SSH alias from this
workstation.

Readiness check:

```bash
.venv/bin/python \
  molmoact2/check_finetune_readiness.py \
  --dataset-repo-id <hf_user>/<dataset>
```

Once the gate is clear and a trainable command exists:

```bash
cluster/brev/submit_finetune_brev.sh \
  --dataset-repo-id <hf_user>/<dataset> \
  --gpus 1 \
  --time 24h \
  --train-command '<official MolmoAct2 fine-tuning command>'
```

Blocked launch check performed on 2026-05-08:

```bash
cluster/brev/submit_finetune_brev.sh \
  --dataset-repo-id carmensc/record-test-screwdriver \
  --dry-run \
  --readiness-report outputs/molmoact2/blocked_brev_readiness.json \
  --train-command 'echo would train'
```

Result: the script printed the `mw-newton-dev` launch plan, ran local readiness,
wrote the local JSON readiness report, and exited before any remote launch.
Current blockers are the old Carmen dataset range/calibration mismatch and
upstream MolmoAct2 trainability.

For diagnostics only, the dry-run plan can be printed even while readiness is
blocked:

```bash
cluster/brev/submit_finetune_brev.sh \
  --dataset-repo-id carmensc/record-test-screwdriver \
  --dry-run \
  --allow-blocked-dry-run \
  --readiness-report outputs/molmoact2/blocked_brev_readiness.json \
  --train-command 'echo would train'
```

This still does not sync, SSH launch, or start a remote process.

## Deployment Gate

Before real robot rollout:

```text
1. One-frame local inference returns 30x6 actions.
2. Action range stays inside the recollected dataset q01/q99 envelope.
3. One-step real robot test is clipped, rate-limited, and starts at low speed.
4. Longer rollout starts only after one-step commands are physically coherent.
```
