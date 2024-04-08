# 🦊 fog_rtx

[![codecov](https://codecov.io/gh/KeplerC/fog_rtx/branch/main/graph/badge.svg?token=fog_rtx_token_here)](https://codecov.io/gh/KeplerC/fog_rtx)
[![CI](https://github.com/KeplerC/fog_rtx/actions/workflows/main.yml/badge.svg)](https://github.com/KeplerC/fog_rtx/actions/workflows/main.yml)

An Efficient and Scalable Data Collection and Management Framework For Robotics Learning. Support RT-X, HuggingFace. 

## Install 

```bash
pip install fog_rtx
```

## Usage

```py
import fog_rtx as fox 

# create a new dataset
dataset = fox.Dataset(
    name="test_rtx", path="/tmp/rtx", 
)  

# Data collection: 
# create a new episode / trajectory
episode = dataset.new_episode(
    description = "grasp teddy bear from the shelf"
)

# collect step data for the episode
episode.add(feature = "arm_view", value = "image1.jpg")
episode.add(feature = "camera_pose", value = "image1.jpg")

# mark the current state as terminal state 
episode.close()

# Alternatively, 
# load from existing RT-X/Open-X datasets 
dataset.load_rtx_episodes(
    name="berkeley_autolab_ur5",
    additional_metadata={"collector": "User 2"}
)

# Data Analytics and management
episode_info = dataset.get_episode_info()
metadata = episode_info.filter(episode_info["collector"] == "User 2")
episodes = dataset.read_by(metadata)

# export and share the dataset as standard RT-X format
dataset.export(episodes, format="rtx")
```


## More Coming Soon!
Currently we see a 60\% space saving on some existing RT-X datasets. This can be even more with re-paritioning the dataset. Our next steps can be found in the [planning doc](./design_doc/planning_doc.md). Feedback welcome through issues or PR to planning doc!

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file. 
