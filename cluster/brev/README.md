# Brev

This is the Newton-style Brev workflow port for `vla_picknplace`.

The normal path reuses the existing Newton SSH alias, `mw-newton-dev`. Code
setup and training launch use direct SSH/rsync:

```text
ssh mw-newton-dev ...
rsync local repo -> Brev VM
remote uv venv + pip install
nohup remote train command
```

## SSH Alias

```bash
ssh mw-newton-dev "hostname && nvidia-smi -L"
```

No separate Brev CLI login is needed when that command works. Use Brev CLI only
to repair a missing or stale SSH alias:

```bash
brev login --auth nvidia --skip-browser
brev refresh
brev ls --all
```

If you have a token, `brev login --auth nvidia --token "$BREV_TOKEN"` is the
noninteractive repair path.

## Configure

```bash
cp cluster/brev/.env.brev.template cluster/brev/.env.brev
```

Edit:

```bash
export BREV_INSTANCE_NAME=mw-newton-dev
export BREV_CODE_DIR=/home/nvidia/code/vla_picknplace
export BREV_LOGS_DIR=/home/nvidia/logs/vla_picknplace
export BREV_DATA_DIR=/home/nvidia/data/vla_picknplace
```

This can reuse the same Brev VM as `moleworks_newton`; the code/log/data paths
stay separate.

## Setup

```bash
cluster/brev/setup_brev_env.sh
```

This syncs code, installs FFmpeg shared libraries for LeRobot video decoding,
installs `uv`, creates `.venv`, installs LeRobot and `requirements.txt`, and
checks imports/GPU visibility.

## Submit

The submit script is intentionally guarded. It runs
`molmoact2/check_finetune_readiness.py` locally before SSH launch. Today that
should still block because MolmoAct2 upstream fine-tuning is not public.

When an official train command exists:

```bash
cluster/brev/submit_finetune_brev.sh \
  --dataset-repo-id <hf_user>/<dataset> \
  --gpus 1 \
  --time 24h \
  --train-command '<official MolmoAct2 fine-tuning command>'
```

Dry-run the remote launch shape without starting a process:

```bash
cluster/brev/submit_finetune_brev.sh \
  --dataset-repo-id <hf_user>/<dataset> \
  --dry-run \
  --readiness-report outputs/molmoact2/brev_readiness.json \
  --train-command 'echo would train'
```

If you need to inspect the dry-run shape while the readiness gate is expected to
block, keep it diagnostic-only:

```bash
cluster/brev/submit_finetune_brev.sh \
  --dataset-repo-id carmensc/record-test-screwdriver \
  --dry-run \
  --allow-blocked-dry-run \
  --readiness-report outputs/molmoact2/blocked_brev_readiness.json \
  --train-command 'echo would train'
```

`--allow-blocked-dry-run` never launches a remote process. The readiness report
is local JSON and records the exact blockers from
`molmoact2/check_finetune_readiness.py`.

## Monitor

After launch, the script records SSH tail/kill commands in:

```text
cluster/brev/submitted_jobs.txt
```

Manual tail:

```bash
ssh <instance-name> "tail -f /home/nvidia/logs/vla_picknplace/brev-<run-id>.log"
```
