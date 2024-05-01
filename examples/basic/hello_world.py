import fog_x

# 🦊 Dataset Creation 
# from distributed dataset storage 
dataset = fog_x.Dataset(
    name="demo_ds",
    path="~/test_dataset", # can be AWS S3, Google Bucket! 
)  

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