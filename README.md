# fog_rtx

[![codecov](https://codecov.io/gh/KeplerC/fog_rtx/branch/main/graph/badge.svg?token=fog_rtx_token_here)](https://codecov.io/gh/KeplerC/fog_rtx)
[![CI](https://github.com/KeplerC/fog_rtx/actions/workflows/main.yml/badge.svg)](https://github.com/KeplerC/fog_rtx/actions/workflows/main.yml)

An Efficient and Scalable Data Collection and Management Framework For Robotics Learning. Support RT-X, HuggingFace. 

## Install 

```bash
pip install fog_rtx
```

## Usage

```py
import fog_rtx

# create a new dataset
dataset = fog_rtx.dataset.Dataset(
    name="test_rtx", path="/tmp/rtx", 
)  

# create a new episode / trajectory
episode = dataset.new_episode(
    description = "grasp teddy bear from the shelf"
)

# populate the episode with FeatureTypes
episode.add(feature = "arm_view", value = "image1.jpg")
episode.add(feature = "camera_pose", value = "image1.jpg")

# mark the current state as terminal state 
# and save the episode
episode.close()

# load the dataset
episodes = dataset.query("SELECT *")
for episode in episodes:
    print(episode)

# export the dataset
dataset.export("/tmp/rtx_export", format="rtx")
```

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file. See [TODO](./design_doc/planning_doc.md)
