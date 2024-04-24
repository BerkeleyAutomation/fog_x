import io
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
import subprocess
import numpy as np
import polars
import pandas

from fog_x.database import (
    DatabaseConnector,
    DatabaseManager,
    DataFrameConnector,
    LazyFrameConnector,
    PolarsConnector,
)
from fog_x.episode import Episode
from fog_x.feature import FeatureType

logger = logging.getLogger(__name__)



def convert_to_h264(input_file, output_file):
    
    # FFmpeg command to convert video to H.264
    command = [
        'ffmpeg',
        '-i', input_file,    # Input file
        '-loglevel', 'error', # Suppress the logs
        '-vcodec', 'h264', # Specify the codec
        output_file          # Output file
    ]
    subprocess.run(command)

def create_cloud_bucket_if_not_exist(provider, bucket_name, dir_name):
    logger.info(f"Creating bucket '{bucket_name}' in cloud provider '{provider}' with folder '{dir_name}'...")
    if provider == "s3":
        import boto3
        s3_client = boto3.client('s3')
        # s3_client.create_bucket(Bucket=bucket_name)
        s3_client.put_object(Bucket=bucket_name, Key=f"{dir_name}/")
        logger.info(f"Bucket '{bucket_name}' created in AWS S3.")
    elif provider == "gs":
        from google.cloud import storage
        """Create a folder in a Google Cloud Storage bucket if it does not exist."""
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # Ensure the folder name ends with a '/'
        if not dir_name.endswith('/'):
            dir_name += '/'

        # Check if folder exists by trying to list objects with the folder prefix
        blobs = storage_client.list_blobs(bucket_name, prefix=dir_name, delimiter='/')
        exists = any(blob.name == dir_name for blob in blobs)

        if not exists:
            # Create an empty blob to simulate a folder
            blob = bucket.blob(dir_name)
            blob.upload_from_string('')
            print(f"Folder '{dir_name}' created.")
        else:
            print(f"Folder '{dir_name}' already exists.")
    else:
        raise ValueError(f"Unsupported cloud provider '{provider}'.")

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
        enable_feature_inference=True,  # whether additional features can be inferred
        episode_info_connector: DatabaseConnector = None,
        step_data_connector: DatabaseConnector = None,
        storage: Optional[str] = None,
    ) -> None:
        """

        Args:
            name (str): Name of this dataset. Used as the directory name when exporting.
            path (str): Required. Local path of where this dataset should be stored.
            features (optional Dict[str, FeatureType]): Description of `param1`.
            enable_feature_inference (bool): enable inferring additional FeatureTypes

        Example:
        ```
        >>> dataset = fog_x.Dataset('my_dataset', path='~/fog_x/my_dataset`)
        ```

        TODO:
            * is replace_existing actually used anywhere?
        """
        self.name = name
        path = os.path.expanduser(path).removesuffix("/")
        logger.info(f"Dataset path: {path}")
        self.path = path
        if path is None:
            raise ValueError("Path is required")
        # create the folder if path doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)

        self.replace_existing = replace_existing
        self.features = features
        self.enable_feature_inference = enable_feature_inference
        if episode_info_connector is None:
                episode_info_connector = DataFrameConnector(f"{path}")
        
        if step_data_connector is None:
            if not os.path.exists(f"{path}/{name}"):
                os.makedirs(f"{path}/{name}")
            try: 
                step_data_connector = LazyFrameConnector(f"{path}/{name}")
            except:
                logger.info(f"Path does not exist. ({path}/{name})")
                cloud_provider = path[:2]
                bucket_name = path[5:]
                create_cloud_bucket_if_not_exist(cloud_provider, bucket_name, f"{name}/")
                step_data_connector = LazyFrameConnector(f"{path}/{name}") 
        self.db_manager = DatabaseManager(episode_info_connector, step_data_connector)
        self.db_manager.initialize_dataset(self.name, features)

        self.storage = storage
        self.obs_keys = []
        self.act_keys = []
        self.step_keys = []

    def new_episode(self, metadata: Optional[Dict[str, Any]] = None) -> Episode:
        """
        Create a new episode / trajectory.

        Returns:
            Episode

        TODO:
            * support multiple processes writing to the same episode
            * close the previous episode if not closed
        """
        return Episode(
            metadata=metadata,
            features=self.features,
            enable_feature_inference=self.enable_feature_inference,
            db_manager=self.db_manager,
        )

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
        export_path: Optional[str] = None,
        format: str = "rtx",
        max_episodes_per_file: int = 1,
        version: str = "0.0.1",
        obs_keys=[],
        act_keys=[],
        step_keys=[],
    ) -> None:
        """
        Export the dataset.

        Args:
            export_path (optional str): location of exported data. Uses dataset.path/export by default.
            format (str): Supported formats are `rtx`, `open-x`, and `rlds`.
        """
        if format == "rtx" or format == "open-x" or format == "rlds":
            self.export_rtx(export_path, max_episodes_per_file, version, obs_keys, act_keys, step_keys)
        else:
            raise ValueError("Unsupported export format")

    def export_rtx(        
        self,
        export_path: Optional[str] = None,
        max_episodes_per_file: int = 1,
        version: str = "0.0.1",
        obs_keys=[],
        act_keys=[],
        step_keys=[]
    ):
        if export_path == None:
            export_path = self.path + "/export"
        if not os.path.exists(export_path):
            os.makedirs(export_path)

        import dm_env
        import tensorflow as tf
        import tensorflow_datasets as tfds
        from envlogger import step_data
        from tensorflow_datasets.core.features import Tensor

        from fog_x.rlds.writer import CloudBackendWriter

        self.obs_keys += obs_keys
        self.act_keys += act_keys
        self.step_keys += step_keys

        (
            observation_tf_dict,
            action_tf_dict,
            step_tf_dict,
        ) = self._get_tf_feature_dicts(
            self.obs_keys,
            self.act_keys,
            self.step_keys,
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
            steps = episode.collect().rows(named=True)
            for i in range(len(steps)):
                step = steps[i]
                observationd = {}
                actiond = {}
                stepd = {}
                for k, v in step.items():
                    # logger.info(f"key: {k}")
                    if k not in self.features:
                        if k != "episode_id" and k != "Timestamp":
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

                    if k in self.obs_keys:
                        observationd[k] = value
                    elif k in self.act_keys:
                        actiond[k] = value
                    else:
                        stepd[k] = value

                # logger.info(
                #     f"Step: {stepd}"
                #     f"Observation: {observationd}"
                #     f"Action: {actiond}"
                # )
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
                if i < len(steps) - 1:
                    writer._record_step(stepdata, is_new_episode=False)
                else:
                    writer._record_step(stepdata, is_new_episode=True)


    def load_rtx_episodes(
        self,
        name: str,
        split: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = dict(),
    ):
        """
        Load robot data from Tensorflow Datasets.

        Args:
            name (str): Name of RT-X episodes, which can be found at [Tensorflow Datasets](https://www.tensorflow.org/datasets/catalog) under the Robotics category
            split (optional str): the portion of data to load, see [Tensorflow Split API](https://www.tensorflow.org/datasets/splits)
            additional_metadata (optional Dict[str, Any]): additional metadata to be associated with the loaded episodes

        Example:
            ```
            >>> dataset.load_rtx_episodes(name="berkeley_autolab_ur5)
            >>> dataset.load_rtx_episodes(name="berkeley_autolab_ur5", split="train[:10]", additional_metadata={"data_collector": "Alice", "custom_tag": "sample"})
            ```
        """

        # this is only required if rtx format is used
        import tensorflow_datasets as tfds

        from fog_x.rlds.utils import dataset2path

        b = tfds.builder_from_directory(builder_dir=dataset2path(name))

        ds = b.as_dataset(split=split)

        data_type = b.info.features["steps"]

        for tf_episode in ds:
            logger.info(tf_episode)
            fog_episode = self.new_episode(
                    metadata=additional_metadata,
                )
            for step in tf_episode["steps"]:
                ret = self._load_rtx_step_data_from_tf_step(
                    step, data_type, 
                )
                for r in ret:
                    fog_episode.add(**r)

            fog_episode.close()


    def _prepare_rtx_metadata(
        self,
        name: str,
        export_path: Optional[str] = None,
        sample_size = 20,
        shuffle = False,
        seed = 42,
    ):

        # this is only required if rtx format is used
        import tensorflow_datasets as tfds
        from fog_x.rlds.utils import dataset2path
        import cv2

        b = tfds.builder_from_directory(builder_dir=dataset2path(name))
        ds = b.as_dataset(split="all")
        if shuffle:
            ds = ds.shuffle(sample_size, seed=seed)
        data_type = b.info.features["steps"]
        counter = 0

        if export_path == None:
            export_path = self.path + "/" + self.name + "_viz"
        if not os.path.exists(export_path):
            os.makedirs(export_path)

        
        for tf_episode in ds:
            video_writers = {}

            additional_metadata = {
                "load_from": name,
                "load_index": f"all, {shuffle}, {seed}, {counter}",
            }
            
            logger.info(tf_episode)
            fog_episode = self.new_episode()
            
            for step in tf_episode["steps"]:
                ret = self._load_rtx_step_data_from_tf_step(
                    step, data_type, 
                )

                for r in ret:
                    feature_name = r["feature"]
                    if "image" in feature_name and "depth" not in feature_name:
                        image = np.load(io.BytesIO(r["value"]))

                        # convert from RGB to BGR
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                        if feature_name not in video_writers:
                            
                            output_filename = f"{self.name}_{counter}_{feature_name}"
                            tmp_vid_output_path = f"/tmp/{output_filename}.mp4"
                            output_path = f"{export_path}/{output_filename}"

                            frame_size = (image.shape[1], image.shape[0])

                            # save the initial image
                            cv2.imwrite(f"{output_path}.jpg", image)
                            # save the video
                            video_writers[feature_name] = cv2.VideoWriter(
                                tmp_vid_output_path,
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                10,
                                frame_size
                            )
                            

                        video_writers[r["feature"]].write(image)

                    if "instruction" in r["feature"]:
                        natural_language_instruction = r["value"].decode("utf-8")
                        additional_metadata["natural_language_instruction"] = natural_language_instruction

                    r["metadata_only"] = True
                    fog_episode.add(**r)
            
            for feature_name, video_writer in video_writers.items():
                video_writer.release()
                # need to convert to h264 to properly display over chrome / vscode 
                output_filename = f"{self.name}_{counter}_{feature_name}"
                tmp_vid_output_path = f"/tmp/{output_filename}.mp4"
                vid_output_path = f"{export_path}/{output_filename}.mp4"
                convert_to_h264(tmp_vid_output_path, vid_output_path)
                additional_metadata[f"video_path_{feature_name}"] = output_filename
                if os.path.isfile(tmp_vid_output_path):
                    os.remove(tmp_vid_output_path)
                
            video_writers = {}
            fog_episode.close(save_data = False, additional_metadata = additional_metadata)
            counter += 1
            if counter > sample_size:
                break

    def _load_rtx_step_data_from_tf_step(
            self, 
            step: Dict[str, Any],
            data_type: Dict[str, Any] = {},
    ): 
        from tensorflow_datasets.core.features import (
            FeaturesDict,
            Image,
            Scalar,
            Tensor,
            Text,
        )
        ret = []

        for k, v in step.items():
            # logger.info(f"k {k} , v {v}")
            if isinstance(v, dict): #and (k == "observation" or k == "action"):
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
                    
                    ret.append(
                        {
                            "feature": str(k2),
                            "value": value,
                            "feature_type": FeatureType(
                                tf_feature_spec=data_type[k][k2]
                            ),
                        }
                    )
                    # fog_episode.add(
                    #     feature=str(k2),
                    #     value=value,
                    #     feature_type=FeatureType(
                    #         tf_feature_spec=data_type[k][k2]
                    #     ),
                    # )
                    if k == "observation":
                        self.obs_keys.append(k2)
                    elif k == "action":
                        self.act_keys.append(k2)
            else:
                # fog_episode.add(
                #     feature=str(k),
                #     value=v.numpy(),
                #     feature_type=FeatureType(tf_feature_spec=data_type[k]),
                # )
                ret.append(
                    {
                        "feature": str(k),
                        "value": v.numpy(),
                        "feature_type": FeatureType(
                            tf_feature_spec=data_type[k]
                        ),
                    }
                )
                self.step_keys.append(k)
        return ret
        

    def get_episode_info(self) -> pandas.DataFrame:
        """
        Returns:
            metadata of all episodes as `pandas.DataFrame`
        """
        return self.db_manager.get_episode_info_table()

    def get_step_data(self) -> polars.LazyFrame:
        """
        Returns:
            step data of all episodes
        """
        return self.db_manager.get_step_table_all()

    def get_step_data_by_episode_ids(
        self, episode_ids: List[int], as_lazy_frame=True
    ):
        """
        Args:
            episode_ids (List[int]): list of episode ids
            as_lazy_frame (bool): whether to return polars.LazyFrame or polars.DataFrame

        Returns:
            step data of each episode
        """
        episodes = []
        for episode_id in episode_ids:
            if episode_id == None:
                continue
            if as_lazy_frame:
                episodes.append(self.db_manager.get_step_table(episode_id))
            else:
                episodes.append(self.db_manager.get_step_table(episode_id).collect())
        return episodes

    def read_by(self, episode_info: Any = None) -> List[polars.LazyFrame]:
        """
        To be used with `Dataset.get_episode_info`.

        Args:
            episode_info (pandas.DataFrame): episode metadata information to determine which episodes to read

        Returns:
            episodes filtered by `episode_info`
        """
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
            metadata_df = self.get_episode_info()
        else:
            metadata_df = metadata
        episodes = self.read_by(metadata_df)
        return episodes

    def pytorch_dataset_builder(self, metadata=None, **kwargs):
        """
        Used for loading current dataset as a PyTorch dataset.
        To be used with `torch.utils.data.DataLoader`.
        """

        import torch
        from torch.utils.data import Dataset
        episodes = self.get_episodes_from_metadata(metadata)

        # Initialize the PyTorch dataset with the episodes and features
        pytorch_dataset = PyTorchDataset(episodes, self.features)

        return pytorch_dataset

    def get_as_huggingface_dataset(self):
        """
        Load current dataset as a HuggingFace dataset.

        TODO:
            * currently the support for huggingg face dataset is limited.
                it only shows its capability of easily returning a hf dataset
            * add features from the episode metadata
            * allow selecting episodes based on queries.
                doing so requires creating a new copy of the dataset on disk
        """
        import datasets

        dataset_path = self.path + "/" + self.name
        parquet_files = [
            os.path.join(dataset_path, f) for f in os.listdir(dataset_path)
        ]

        hf_dataset = datasets.load_dataset("parquet", data_files=parquet_files)
        return hf_dataset
    
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
        episode = self.episodes[idx].collect().to_pandas()
        # Process the episode and its features here
        # For simplicity, let's assume we're just returning the episode
        return episode
