# 🦊 Robo-DM 

🦊 Robo-DM : An Efficient and Scalable Data Collection and Management Framework For Robotics Learning. Support [Open-X-Embodiment](https://robotics-transformer-x.github.io/), 🤗[HuggingFace](https://huggingface.co/). 

🦊 Robo-DM (Former Name: fog_x)  considers both speed 🚀 and memory efficiency 📈 with active metadata and lazily-loaded trajectory data. It supports flexible and distributed dataset partitioning. It provides native support to cloud storage. 

[Design Doc](https://docs.google.com/document/d/1woLQVLWsySGjFuz8aCsaLoc74dXQgIccnWRemjlNDws/edit#heading=h.irrfcedesnvr) | [Dataset Visualization](https://keplerc.github.io/openxvisualizer/)

## Note to ICRA Reviewers
We are actively developing the framework. See commit `a35a6` for the version we developed.  


## Install 

```bash
git clone https://github.com/BerkeleyAutomation/fog_x.git
cd fog_x
pip install -e .
```

## Usage

```py
import fog_x

path = "/tmp/output.vla"

# 🦊 Data collection: 
# create a new trajectory
traj = fog_x.Trajectory(
    path = path
)

traj.add(feature = "arm_view", value = "image1.jpg")
# Automatically time-aligns and saves the trajectory
traj.close()

# load it 
fog_x.Trajectory(
    path = path
)
```

## Examples

* [Data Collection and Loading](./examples/data_collection_and_load.py)
* [Convert From Open_X](./examples/openx_loader.py)
* [Convert From H5](./examples/h5_loader.py)
* [Running Benchmarks](./benchmarks/openx.py)

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file. 
