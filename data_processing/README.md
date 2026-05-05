# Data Processing

Utilities for inspecting and visualizing LeRobot datasets before training.

The current default dataset is:

```text
carmensc/record-test-screwdriver
```

## Visualize an Episode

```bash
.venv/bin/python data_processing/visualize_lerobot_dataset.py --episode 0
```

The script writes:

```text
outputs/playground/record-test-screwdriver/*_episode_000.mp4
outputs/playground/record-test-screwdriver/*_contact_sheet.jpg
outputs/playground/record-test-screwdriver/*_summary.json
```

Those outputs are intentionally ignored by git.

## Why Not `datasets.load_dataset` Alone?

`datasets.load_dataset("carmensc/record-test-screwdriver")` loads the parquet table with action, state, timestamps, and episode indices. The camera frames live in LeRobot video sidecars, so use `LeRobotDataset` when you need synchronized images for visualization or ACT training.
