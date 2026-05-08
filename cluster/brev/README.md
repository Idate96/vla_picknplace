# Brev

This is the Newton-style Brev workflow port for `vla_picknplace`.

It uses Brev only for authentication and SSH metadata. Code setup and training
launch use direct SSH/rsync:

```text
brev login / brev refresh / brev ls --all
ssh <instance> ...
rsync local repo -> Brev VM
remote uv venv + pip install
nohup remote train command
```

## Login

```bash
brev login --auth nvidia --skip-browser
brev refresh
brev ls --all
ssh <instance-name> "hostname && nvidia-smi -L"
```

If you have a token:

```bash
export BREV_TOKEN='<token>'
brev login --auth nvidia --token "$BREV_TOKEN"
brev refresh
brev ls --all
```

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

This syncs code, installs `uv`, creates `.venv`, installs LeRobot and
`requirements.txt`, and checks imports/GPU visibility.

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
  --train-command 'echo would train'
```

## Monitor

After launch, the script records SSH tail/kill commands in:

```text
cluster/brev/submitted_jobs.txt
```

Manual tail:

```bash
ssh <instance-name> "tail -f /home/nvidia/logs/vla_picknplace/brev-<run-id>.log"
```
