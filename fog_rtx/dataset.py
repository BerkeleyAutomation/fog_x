import io
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fog_rtx.database import (
    DatabaseConnector,
    DatabaseManager,
    PolarsConnector,
)
from fog_rtx.episode import Episode
from fog_rtx.feature import FeatureType

logger = logging.getLogger(__name__)


class Dataset:
    """
    Create or load from a new dataset.
    """

    def __init__(
        self,
        name: str,
        path: str = None,
        replace_existing: bool = False,
        features: Dict[
            str, FeatureType
        ] = {},  # features to be stored {name: FeatureType}
        enable_feature_inferrence=True,  # whether additional features can be inferred
        episode_connector: DatabaseConnector = None, 
        step_connector: DatabaseConnector = None, 
        storage: Optional[str] = None,
    ) -> None:
        self.name = name
        self.path = path
        if path is None: 
            raise ValueError("Path is required")
        # create the folder if path doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)

        self.replace_existing = replace_existing
        self.features = features
        self.enable_feature_inferrence = enable_feature_inferrence

        if episode_connector is None:
            episode_connector = PolarsConnector(f"{path}/")
        if step_connector is None:
            if not os.path.exists(f"{path}/steps"):
                os.makedirs(f"{path}/steps")
            step_connector = PolarsConnector(f"{path}/steps")
        self.db_manager = DatabaseManager(episode_connector, step_connector)
        self.db_manager.initialize_dataset(self.name, features)

        self.storage = storage
        self.obs_keys = []
        self.act_keys = []
        self.step_keys = []

    def new_episode(
        self, metadata: Optional[Dict[str, Any]] = None
    ) -> Episode:
        """
        Create a new episode / trajectory.
        TODO #1: support multiple processes writing to the same episode
        TODO #2: close the previous episode if not closed
        """
        return Episode(
            metadata=metadata,
            features=self.features,
            enable_feature_inferrence=self.enable_feature_inferrence,
            db_manager=self.db_manager,
        )

    def query(self, query: str) -> List[Episode]:
        """
        Query the dataset.
        """
        return self.db_manager.query(query)

    def _get_tf_feature_dicts(
        self, obs_keys: List[str], act_keys: List[str], step_keys: List[str]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Get the tensorflow feature dictionaries.
        """
        observation_tf_dict = {}
        action_tf_dict = {}
        step_tf_dict = {}

        for k in obs_keys:
            observation_tf_dict[k] = self.features[k].to_tf_feature_type()

        for k in act_keys:
            action_tf_dict[k] = self.features[k].to_tf_feature_type()

        for k in step_keys:
            step_tf_dict[k] = self.features[k].to_tf_feature_type()

        return observation_tf_dict, action_tf_dict, step_tf_dict

    def export(
        self,
        export_path: str,
        format: str,
        max_episodes_per_file: int = 1,
        version: str = "0.0.1",
        obs_keys=[],
        act_keys=[],
        step_keys=[],
    ) -> None:
        """
        Export the dataset.
        """
        if format == "rtx":
            import dm_env
            import tensorflow as tf
            import tensorflow_datasets as tfds
            from envlogger import step_data
            from tensorflow_datasets.core.features import Tensor

            from fog_rtx.rlds.writer import CloudBackendWriter

            (
                observation_tf_dict,
                action_tf_dict,
                step_tf_dict,
            ) = self._get_tf_feature_dicts(
                self.obs_keys + obs_keys,
                self.act_keys + act_keys,
                self.step_keys + step_keys,
            )

            logger.info("Exporting dataset as RT-X format")
            logger.info(f"Observation keys: {observation_tf_dict}")
            logger.info(f"Action keys: {action_tf_dict}")
            logger.info(f"Step keys: {step_tf_dict}")

            # generate tensorflow configuration file
            ds_config = tfds.rlds.rlds_base.DatasetConfig(
                name=self.name,
                description="",
                homepage="",
                citation="",
                version=tfds.core.Version("0.0.1"),
                release_notes={
                    "0.0.1": "Initial release.",
                },
                observation_info=observation_tf_dict,
                action_info=action_tf_dict,
                reward_info=(
                    step_tf_dict["reward"]
                    if "reward" in step_tf_dict
                    else Tensor(shape=(), dtype=tf.float32)
                ),
                discount_info=(
                    step_tf_dict["discount"]
                    if "discount" in step_tf_dict
                    else Tensor(shape=(), dtype=tf.float32)
                ),
            )

            ds_identity = tfds.core.dataset_info.DatasetIdentity(
                name=ds_config.name,
                version=tfds.core.Version(version),
                data_dir=export_path,
                module_name="",
            )
            writer = CloudBackendWriter(
                data_directory=export_path,
                ds_config=ds_config,
                ds_identity=ds_identity,
                max_episodes_per_file=max_episodes_per_file,
            )

            # export the dataset
            episodes = self.get_episodes_from_metadata()
            for episode in episodes:
                for step in episode.rows(named=True):
                    observationd = {}
                    actiond = {}
                    stepd = {}
                    for k, v in step.items():
                        logger.info(f"key: {k}")
                        if k not in self.features:
                            logger.info(
                                f"Feature {k} not found in the dataset features."
                            )
                            continue
                        feature_spec = self.features[k].to_tf_feature_type()
                        if (
                            isinstance(feature_spec, tfds.core.features.Tensor)
                            and feature_spec.shape != ()
                        ):
                            # reverse the process
                            value = np.load(io.BytesIO(v)).astype(
                                feature_spec.np_dtype
                            )
                        elif (
                            isinstance(feature_spec, tfds.core.features.Tensor)
                            and feature_spec.shape == ()
                        ):
                            value = np.array(v, dtype=feature_spec.np_dtype)
                        elif isinstance(
                            feature_spec, tfds.core.features.Image
                        ):
                            value = np.load(io.BytesIO(v)).astype(
                                feature_spec.np_dtype
                            )
                        else:
                            value = v

                        if k in obs_keys:
                            observationd[k] = value
                        elif k in act_keys:
                            actiond[k] = value
                        else:
                            stepd[k] = value

                    logger.info(
                        f"Step: {stepd}"
                        f"Observation: {observationd}"
                        f"Action: {actiond}"
                    )
                    timestep = dm_env.TimeStep(
                        step_type=dm_env.StepType.FIRST,
                        reward=np.float32(
                            0.0
                        ),  # stepd["reward"] if "reward" in step else np.float32(0.0),
                        discount=np.float32(
                            0.0
                        ),  # stepd["discount"] if "discount" in step else np.float32(0.0),
                        observation=observationd,
                    )
                    stepdata = step_data.StepData(
                        timestep=timestep, action=actiond, custom_data=None
                    )
                    writer._record_step(stepdata, is_new_episode=True)

            pass
        else:
            raise ValueError("Unsupported export format")

    def get_episode_info(self):
        """
        Return the metadata as pandas dataframe.
        """
        return self.db_manager.get_episode_table()

    def load_rtx_episodes(
        self,
        name: str,
        split: Optional[str],
        additional_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Load the dataset.
        """

        # this is only required if rtx format is used
        import tensorflow_datasets as tfds
        from tensorflow_datasets.core.features import (
            FeaturesDict,
            Image,
            Scalar,
            Tensor,
            Text,
        )

        from fog_rtx.rlds.utils import dataset2path

        b = tfds.builder_from_directory(builder_dir=dataset2path(name))
        ds = b.as_dataset(split=split)

        data_type = b.info.features["steps"]

        for tf_episode in ds:
            logger.info(tf_episode)
            fog_epsiode = self.new_episode(
                metadata=additional_metadata,
            )
            for step in tf_episode["steps"]:
                for k, v in step.items():
                    if k == "observation" or k == "action":
                        for k2, v2 in v.items():
                            # TODO: abstract this to feature.py

                            if (
                                isinstance(data_type[k][k2], Tensor)
                                and data_type[k][k2].shape != ()
                            ):
                                memfile = io.BytesIO()
                                np.save(memfile, v2.numpy())
                                value = memfile.getvalue()
                            elif isinstance(data_type[k][k2], Image):
                                memfile = io.BytesIO()
                                np.save(memfile, v2.numpy())
                                value = memfile.getvalue()
                            else:
                                value = v2.numpy()

                            fog_epsiode.add(
                                feature=str(k2),
                                value=value,
                                feature_type=FeatureType(
                                    tf_feature_spec=data_type[k][k2]
                                ),
                            )
                            if k == "observation":
                                self.obs_keys.append(k2)
                            elif k == "action":
                                self.act_keys.append(k2)
                    else:
                        fog_epsiode.add(
                            feature=str(k),
                            value=v.numpy(),
                            feature_type=FeatureType(
                                tf_feature_spec=data_type[k]
                            ),
                        )
                        self.step_keys.append(k)
            fog_epsiode.close()

    def read_by(self, episode_info: Any = None):
        episode_ids = list(episode_info["episode_id"])
        logger.info(f"Reading episodes as order: {episode_ids}")
        episodes = []
        for episode_id in episode_ids:
            if episode_id == None:
                continue
            episodes.append(self.db_manager.get_step_table(episode_id))
        return episodes

    def get_episodes_from_metadata(self, metadata: Any = None):
        # Assume we use get_metadata_as_pandas_df to retrieve episodes metadata
        if metadata is None:
            metadata_df = self.get_metadata_as_pandas_df()
        else:
            metadata_df = metadata
        episodes = self.read_by(metadata_df)
        return episodes

    def pytorch_dataset_builder(self, metadata=None, **kwargs):
        import torch
        from torch.utils.data import Dataset

        class PyTorchDataset(Dataset):
            def __init__(self, episodes, features):
                """
                Initialize the dataset with the episodes and features.
                :param episodes: A list of episodes loaded from the database.
                :param features: A dictionary of features to be included in the dataset.
                """
                self.episodes = episodes
                self.features = features

            def __len__(self):
                """
                Return the total number of episodes in the dataset.
                """
                return len(self.episodes)

            def __getitem__(self, idx):
                """
                Retrieve the idx-th episode from the dataset.
                Depending on the structure, you may need to process the episode
                and its features here.
                """
                print("Retrieving episode at index", idx)
                episode = self.episodes[idx]
                # Process the episode and its features here
                # For simplicity, let's assume we're just returning the episode
                return episode

        episodes = self.get_episodes_from_metadata(metadata)

        # Initialize the PyTorch dataset with the episodes and features
        pytorch_dataset = PyTorchDataset(episodes, self.features)

        return pytorch_dataset

    def tensorflow_dataset_builder(
        self, metadata=None, batch_size=32, shuffle_buffer_size=None, **kwargs
    ):
        # TODO: doesn't work yet
        """
        Build a TensorFlow dataset.

        :param batch_size: The size of the batches of data.
        :param shuffle_buffer_size: The buffer size for shuffling the data. If None, no shuffling will be performed.
        :param kwargs: Additional arguments for TensorFlow data transformations.
        :return: A tf.data.Dataset object.
        """
        import tensorflow as tf

        if metadata is None:
            metadata_df = self.get_metadata_as_pandas_df()
        else:
            metadata_df = metadata

        episodes = self.read_by(metadata_df)
        print(episodes)
        # Convert data into tensors
        # Assuming episodes_data is a list of numpy arrays or a similar format that can be directly converted to tensors.
        # This might require additional processing depending on the data format and the features.
        episodes_tensors = [
            tf.convert_to_tensor(episode.drop(columns=["Timestamp", "index"]))
            for episode in episodes
        ]

        # Create a tf.data.Dataset from tensors
        dataset = tf.data.Dataset.from_tensor_slices(episodes_tensors)

        # Shuffle the dataset if shuffle_buffer_size is provided
        if shuffle_buffer_size:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

        # Batch the dataset
        dataset = dataset.batch(batch_size)

        # Apply any additional transformations provided in kwargs
        for transformation, parameters in kwargs.items():
            if hasattr(dataset, transformation):
                dataset = getattr(dataset, transformation)(**parameters)

        return dataset
