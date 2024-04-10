import sys
import pytest

# import fog_x


def test_import():
    # each test runs on cwd to its temp dir
    import fog_x


def test_dataset_create():
    import fog_x

    dataset = fog_x.Dataset(
        name="test_fog_x",
        path="/tmp/test_fog_x",
    )


def test_episode_create():
    import fog_x

    dataset = fog_x.Dataset(
        name="test_fog_x",
        path="/tmp/test_fog_x",
    )
    trajectory = dataset.new_episode()
    trajectory.add(feature="hello", value=1.0)
    trajectory.add(feature="world", value=2.0)
    trajectory.close()


def test_dataset_read():
    import fog_x

    dataset = fog_x.Dataset(
        name="test_fog_x",
        path="/tmp/test_fog_x",
    )
    for episode in dataset.read_by(
        pandas_metadata=dataset.get_metadata_as_pandas_df()
    ):
        print(episode)


def test_dataset_export():
    import fog_x

    dataset = fog_x.Dataset(
        name="test_fog_x",
        path="/tmp/test_fog_x",
    )
    dataset.export(
        "/tmp/test_fog_x_export",
        format="rtx",
        obs_keys=["hello"],
        act_keys=["world"],
    )


def test_rtx_example_load():
    import fog_x

    dataset = fog_x.Dataset(
        name="test_fog_x",
        path="/tmp/test_fog_x",
    )

    dataset.load_rtx_episodes(
        name="berkeley_autolab_ur5",
        split="train[:1]",
    )

    dataset.export("/tmp/rtx_export", format="rtx")


def test_rtx_example_merge():
    import fog_x

    dataset = fog_x.Dataset(
        name="test_fog_x",
        path="/tmp/test_fog_x",
    )
    dataset.load_rtx_episodes(
        name="berkeley_autolab_ur5",
        split="train[:2]",
        additional_metadata={
            "collector": "User 1",
            "custom_tag": "Partition_1",
        },
    )

    dataset.load_rtx_episodes(
        name="berkeley_autolab_ur5",
        split="train[3:5]",
        additional_metadata={
            "collector": "User 2",
            "custom_tag": "Partition_2",
        },
    )


def test_rtx_example_query():
    dataset = fog_x.Dataset(
        name="test_fog_x",
        path="/tmp/test_fog_x",
    )
    metadata = dataset.get_metadata_as_pandas_df()
    print(metadata)
    metadata = metadata.filter(metadata["custom_tag"] == "Partition_1")
    episodes = dataset.read_by(metadata)
    for episode in episodes:
        print(episode)
