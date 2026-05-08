# MolmoAct2 Command Sheet

Run everything from:

```bash
cd ~/git/vla_picknplace
```

## Local Verify

```bash
.venv/bin/python molmoact2/verify_molmoact2_artifacts.py
```

Expected:

```text
All MolmoAct2 local artifacts verified.
```

## Dataset Readiness

Replace `<hf_user>/<dataset>` with the recollected Carmen screwdriver dataset.

```bash
.venv/bin/python molmoact2/inspect_molmoact2.py \
  --dataset-repo-id <hf_user>/<dataset>

.venv/bin/python molmoact2/check_finetune_readiness.py \
  --dataset-repo-id <hf_user>/<dataset>
```

For a local LeRobot dataset mirror:

```bash
.venv/bin/python molmoact2/check_finetune_readiness.py \
  --dataset-repo-id <hf_user>/<dataset> \
  --dataset-root /path/to/local/lerobot/dataset
```

## One-Frame Inference

Dry-run, no 5B model download:

```bash
.venv/bin/python molmoact2/test_on_lerobot_frame.py \
  --dataset-repo-id <hf_user>/<dataset> \
  --dry-run
```

GPU inference:

```bash
.venv/bin/python molmoact2/test_on_lerobot_frame.py \
  --dataset-repo-id <hf_user>/<dataset> \
  --run-model \
  --device cuda \
  --dtype bfloat16
```

## Brev Login

```bash
brev login --auth nvidia --skip-browser
brev refresh
brev ls --all
ssh <instance-name> "hostname && nvidia-smi -L"
```

Token path:

```bash
export BREV_TOKEN='<token>'
brev login --auth nvidia --token "$BREV_TOKEN"
brev refresh
brev ls --all
```

## Brev Setup

```bash
cp cluster/brev/.env.brev.template cluster/brev/.env.brev
```

Edit `cluster/brev/.env.brev`:

```bash
export BREV_INSTANCE_NAME=<instance-name>
export BREV_CODE_DIR=/home/nvidia/code/vla_picknplace
export BREV_LOGS_DIR=/home/nvidia/logs/vla_picknplace
export BREV_DATA_DIR=/home/nvidia/data/vla_picknplace
```

Bootstrap:

```bash
cluster/brev/setup_brev_env.sh
```

## Brev Fine-Tune

Current expected gate:

```text
BLOCKED until Ai2 publishes a trainable MolmoAct2 fine-tuning entrypoint or we
approve a public local training recipe.
```

When that command exists:

```bash
cluster/brev/submit_finetune_brev.sh \
  --dataset-repo-id <hf_user>/<dataset> \
  --gpus 1 \
  --time 24h \
  --train-command '<official MolmoAct2 fine-tuning command>'
```
