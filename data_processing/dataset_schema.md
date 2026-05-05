# Dataset Schema

Default dataset:

```text
carmensc/record-test-screwdriver
```

Verified LeRobot metadata:

```text
robot_type: so_follower
fps: 30
episodes: 20
frames: 7069
task: pickup screwdriver
```

Features:

```text
observation.images.front  video, HWC metadata shape (480, 640, 3)
observation.state         float32[6]
action                    float32[6]
timestamp                 float32
frame_index               int64
episode_index             int64
index                     int64
task_index                int64
```

When loaded through `LeRobotDataset`, the image is returned channel-first for policy training:

```text
observation.images.front -> torch tensor (3, 480, 640)
```

ACT with default `chunk_size=100` asks for future action chunks:

```text
action -> torch tensor (100, 6)
```

Use `act/check_act_dataset.py` whenever changing datasets, camera keys, or LeRobot versions.
