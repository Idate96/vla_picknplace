# Push Scope

This public repo should contain only code, docs, and small configuration files.

## Include

- `act/` training and sanity-check scripts.
- `data_processing/` dataset inspection and visualization utilities.
- `README.md`, `requirements.txt`, and collaborator notes.
- Small shell or Slurm templates after paths are parameterized.

## Exclude

- Local datasets and downloaded Hugging Face dataset files.
- Generated preview videos, contact sheets, and Rerun recordings.
- Training outputs, checkpoints, model weights, optimizer states, and W&B runs.
- Local virtual environments and caches.
- The full upstream `lerobot/` checkout. Install it as a dependency instead.

## Review Rule

Before pushing, run:

```bash
git status --short
git ls-files
```

The tracked list should be small and should not include `data/`, `outputs/`, `.venv/`, `hf_meta/`, `.parquet`, `.mp4`, `.safetensors`, `.pt`, `.pth`, or `.ckpt` files.
