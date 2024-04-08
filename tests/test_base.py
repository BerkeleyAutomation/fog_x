import sys
import pytest

# import fog_rtx


def test_import():
    # each test runs on cwd to its temp dir
    import fog_rtx


def test_dataset_create():
    import fog_rtx

    dataset = fog_rtx.Dataset(
        name="test_fog_rtx",
        path="/tmp/test_fog_rtx",
    )


def test_episode_create():
    import fog_rtx

    dataset = fog_rtx.Dataset(
        name="test_fog_rtx",
        path="/tmp/test_fog_rtx",
    )
    trajectory = dataset.new_episode()
    trajectory.add(feature="hello", value=1.0)
    trajectory.add(feature="world", value=2.0)
    trajectory.close()


def test_dataset_read():
    import fog_rtx

    dataset = fog_rtx.Dataset(
        name="test_fog_rtx",
        path="/tmp/test_fog_rtx",
    )
    for episode in dataset.read_by(
        pandas_metadata=dataset.get_metadata_as_pandas_df()
    ):
        print(episode)


def test_dataset_export():
    import fog_rtx

    dataset = fog_rtx.Dataset(
        name="test_fog_rtx",
        path="/tmp/test_fog_rtx",
    )
    dataset.export(
        "/tmp/test_fog_rtx_export",
        format="rtx",
        obs_keys=["hello"],
        act_keys=["world"],
    )


def test_rtx_example_load():
    import fog_rtx

    dataset = fog_rtx.Dataset(
        name="test_fog_rtx",
        path="/tmp/test_fog_rtx",
    )

    dataset.load_rtx_episodes(
        name="berkeley_autolab_ur5",
        split="train[:1]",
    )

    dataset.export("/tmp/rtx_export", format="rtx")


def test_rtx_example_merge():
    import fog_rtx

    dataset = fog_rtx.Dataset(
        name="test_fog_rtx",
        path="/tmp/test_fog_rtx",
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
    dataset = fog_rtx.Dataset(
        name="test_fog_rtx",
        path="/tmp/test_fog_rtx",
    )
    metadata = dataset.get_metadata_as_pandas_df()
    print(metadata)
    metadata = metadata.filter(metadata["custom_tag"] == "Partition_1")
    episodes = dataset.read_by(metadata)
    for episode in episodes:
        print(episode)
