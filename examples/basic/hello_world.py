import fog_rtx

# create a new dataset
dataset = fog_rtx.Dataset(
    name="test_rtx",
    path="/tmp/rtx",
)

# create a new episode / trajectory
trajectory = dataset.new_episode()
trajectory.add(feature="hello", value="world")
trajectory.te r z()

# iterate through the dataset
for episode in dataset.load():
    print(episode)

# export the dataset as standard rt-x format
dataset.export("/tmp/rtx_export", format="rtx")
