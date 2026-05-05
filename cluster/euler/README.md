# Euler

Slurm launchers for running this project on Euler.

Keep reusable training logic in `act/`. Files here should only handle cluster setup, scratch/cache paths, and `sbatch` resource requests.

```bash
sbatch cluster/euler/train_act.sbatch
```
