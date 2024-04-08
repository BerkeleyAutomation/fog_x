# 🦊 Fog-X

[![codecov](https://codecov.io/gh/KeplerC/fog_rtx/branch/main/graph/badge.svg?token=fog_rtx_token_here)](https://codecov.io/gh/KeplerC/fog_rtx)
[![CI](https://github.com/KeplerC/fog_rtx/actions/workflows/main.yml/badge.svg)](https://github.com/KeplerC/fog_rtx/actions/workflows/main.yml)

🦊Fog-X: An Efficient and Scalable Data Collection and Management Framework For Robotics Learning. Support Open-X-Embodiment, HuggingFace. 

🦊Fog-X considers memory efficiency and speed by working with a trajectory-level metadata and a lazily-loaded dataset. Implemented on [Apache Pyarrows](https://arrow.apache.org/docs/python/index.html) dataset, it allows flexible partitioning of the dataset on distributed storage. 

## Install 

```bash
pip install fog_rtx
```

## Usage

```py
import fog_rtx as fox 

# 🦊 Dataset Creation 
# from distributed dataset storage 
dataset = fox.Dataset(load_from = ["/tmp/rtx", "s3://fox_stroage/"])  

# 🦊 Data collection: 
# create a new trajectory
episode = dataset.new_episode()
# collect step data for the episode
episode.add(feature = "arm_view", value = "image1.jpg")
# Automatically time-aligns the features
episode.close()

# 🦊 Data Loading:
# load from existing RT-X/Open-X datasets
dataset.load_rtx_episodes(
    name="berkeley_autolab_ur5",
    additional_metadata={"collector": "User 2"}
)

# 🦊 Data Management and Analytics
# Compute and memory efficient filter, map, aggregate, groupby
episode_info = dataset.get_episode_info()
metadata = episode_info.filter(episode_info["collector"] == "User 2")

# 🦊 Data Sharing and Usage:
# Export and share the dataset as standard Open-X-Embodiment format
dataset.export(metadata, format="rtx")
# Use with pytorch dataloader
torch.utils.data.DataLoader(dataset.as_pytorch_dataset(metadata))
```

## More Coming Soon!
Currently we see a more than 60\% space saving on some existing RT-X datasets. This can be even more by re-paritioning the dataset. Our next steps can be found in the [planning doc](./design_doc/planning_doc.md). Feedback welcome through issues or PR to planning doc!

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file. 
