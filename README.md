# 🦊 Fog-X

[![codecov](https://codecov.io/gh/KeplerC/fog_rtx/branch/main/graph/badge.svg?token=fog_rtx_token_here)](https://codecov.io/gh/KeplerC/fog_rtx)
[![CI](https://github.com/KeplerC/fog_rtx/actions/workflows/main.yml/badge.svg)](https://github.com/KeplerC/fog_rtx/actions/workflows/main.yml)

🦊 Fog-X: An Efficient and Scalable Data Collection and Management Framework For Robotics Learning. Support [Open-X-Embodiment](https://robotics-transformer-x.github.io/), 🤗[HuggingFace](https://huggingface.co/). 

🦊 Fog-X considers both speed 🚀 and memory efficiency 📈 with active metadata and lazily-loaded trajectory data. It supports flexible and distributed dataset partitioning. 

## Install 

```bash
pip install fogx
```

## Usage

```py
import fogx as fox 

# 🦊 Dataset Creation 
# from distributed dataset storage 
dataset = fox.Dataset(load_from = ["/tmp/rtx", "s3://fox_stroage/"])  

# 🦊 Data collection: 
# create a new trajectory
episode = dataset.new_episode()
# collect step data for the episode
episode.add(feature = "arm_view", value = "image1.jpg")
# Automatically time-aligns and saves the trajectory
episode.close()

# 🦊 Data Loading:
# load from existing RT-X/Open-X datasets
dataset.load_rtx_episodes(
    name="berkeley_autolab_ur5",
    additional_metadata={"collector": "User 2"}
)

# 🦊 Data Management and Analytics: 
# Compute and memory efficient filter, map, aggregate, groupby
episode_info = dataset.get_episode_info()
desired_episodes = episode_info.filter(episode_info["collector"] == "User 2")

# 🦊 Data Sharing and Usage:
# Export and share the dataset as standard Open-X-Embodiment format
dataset.export(desired_episodes, format="rtx")
# Load with pytorch dataloader
torch.utils.data.DataLoader(dataset.as_pytorch_dataset(desired_episodes))
```

## Design
🦊 Fog-X recognizes most post-processing, analytics and management involves the trajectory-level data, such as tags, while actual trajectory steps are rarely read, written and transformed. Acessing and modifying trajectory data is very expensive and hard. 

As a result, 🦊 Fog-X proposes 
* a user-friendly metadata table via Pandas Datframe for speed and freedom
* a LazyFrame from Polars for the trajectory dataset that only loads and transform the data if needed 
* parquet as storage format for distributed storage and columnar support compared to tensorflow records 
* Easy and automatic RT-X/Open-X dataset export and pytorch dataloading 


## More Coming Soon!
Currently we see a more than 60\% space saving on some existing RT-X datasets. This can be even more by re-paritioning the dataset. Our next steps can be found in the [planning doc](./design_doc/planning_doc.md). Feedback welcome through issues or PR to planning doc!

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file. 
