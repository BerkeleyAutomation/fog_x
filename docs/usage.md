# Usage Guide
The code examples below will assume the following import:
```py
import fog_x as fox 
```

## Definitions
- **episode**: one robot trajectory or action, consisting of multiple step data.
- **step data**: data representing a snapshot of the robot action at a certain time
- **metadata**: information that is consistent across a certain episode, e.g. the language instruction associated with the robot action, the name of the person collecting the data, or any other tags/labels.

## The Fog-X Dataset
To start, create a `Dataset` object. Any data that is collected, loaded, or exported
will be saved to the provided path.
There can be existing Fog-X data located at the path as well, so that you can continue
right where you left off.
```py
dataset = fox.Dataset(name="my_fogx_dataset", path="/local/path/my_fogx_dataset")  
```

## Collecting Robot Data

```py
# create a new trajectory
episode = dataset.new_episode()

# run robot and collect data
while robot_is_running:
    # at each step, add data to the episode
    episode.add(feature = "arm_view", value = "image1.jpg")

# Automatically time-aligns and saves the trajectory
episode.close()
```

## Exporting Data
By default, the exported data will be located under the `/export` directory within
the initialized `dataset.path`.
Currently, the supported data formats are `rtx`, `open-x`, and `rlds`.

```py
# Export and share the dataset as standard Open-X-Embodiment format
dataset.export(desired_episodes, format="rtx")
```

### PyTorch
```py
import torch 

metadata = dataset.get_episode_info()
metadata = metadata.filter(metadata["feature1"] == "value1")
pytorch_ds = dataset.pytorch_dataset_builder(metadata=metadata)

# get samples from the dataset
for data in torch.utils.data.DataLoader(
    pytorch_ds,
    batch_size=2,
    collate_fn=lambda x: x,
    sampler=torch.utils.data.RandomSampler(pytorch_ds),
):
    print(data)
```

### HuggingFace
WIP: Currently there is limited support for HuggingFace.

```py
huggingface_ds = dataset.get_as_huggingface_dataset()
```


## Loading Data from Existing Datasets

### RT-X / Tensorflow Datasets
Load any RT-X robotics data available at [Tensorflow Datasets](https://www.tensorflow.org/datasets/catalog/).
You can also find a preview of all the RT-X datasets [here](https://dibyaghosh.com/rtx_viz/).

When loading the episodes, you can optionally specify `additional_metadata` to be associated with it.
You can also load a specific portion of train or test data with the `split` parameter. See the [Tensorflow Split API](https://www.tensorflow.org/datasets/splits) for specifics.

```py
# load all berkeley_autolab_ur5 data
dataset.load_rtx_episodes(name="berkeley_autolab_ur5")

# load 75% of berkeley_autolab_ur5 train data labeled with my_label as train`
dataset.load_rtx_episodes(
    name="berkeley_autolab_ur5",
    split="train[:75%]",
    additional_metadata={"my_label": "train1"}
)
```

## Data Management

### Episode Metadata
You can retrieve episode-level information (metadata) using `dataset.get_episode_info()`.
This is a [pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html),
meaning you have access to pandas data management methods including `filter`, `map`, `aggregate`, `groupby`, etc.
After processing the metadata, you can then use the metadata to obtain your
desired episodes with `dataset.read_by(desired_metadata)`.

```py
# Retrieve episode-level data as a pandas DataFrame
episode_info = dataset.get_episode_info()

# Use pandas DataFrame filter to select episodes
desired_episode_metadata = episode_info.filter(episode_info["natural_language_instruction"] == "open door")

# Obtain the actual episodes containing their step data
episodes = dataset.read_by(desired_episode_metadata)
```

### Step Data
Step data is stored as a [Polars LazyFrame](https://docs.pola.rs/py-polars/html/reference/lazyframe/index.html).
Lazy loading with Polars results in speedups of 10 to 100 times compared to pandas.
```py
# Retrieve Fog-X dataset as a Polars LazyFrame
step_data = dataset.get_step_data()

# select only the episode_id and natural_language_instruction
lazy_id_to_language = step_data.select("episode_id", "natural_language_instruction")

# the frame is lazily evaluated at memory when we call collect(). returns a Polars DataFrame
id_to_language = lazy_id_to_language.collect() 

# drop rows with duplicate natural_language_instruction to see unique instructions
id_to_language.unique(subset=["natural_language_instruction"], maintain_order=True)
```

Polars also allows chaining methods:
```py
# Same as above example, but chained
id_to_language = (
    dataset.get_step_data()
        .select("episode_id", "natural_language_instruction")
        .collect()
        .unique(subset=["natural_language_instruction"], maintain_order=True)
)
```

