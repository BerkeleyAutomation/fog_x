from fractions import Fraction
import logging
import time
from typing import Any, Dict, List, Optional, Text
import av
import numpy as np
import os
from fog_x import FeatureType
import pickle
import h5py

logger = logging.getLogger(__name__)

logging.getLogger("libav").setLevel(logging.CRITICAL)


def _flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class Trajectory:
    def __init__(
        self,
        path: Text,
        mode = "r",
        cache_dir: Optional[Text] = "/tmp/fog_x/cache/",
        num_pre_initialized_h264_streams: int = 5,
        feature_name_separator: Text = "/",
    ) -> None:
        """
        Args:
            path (Text): path to the trajectory file
            mode (Text, optional):  mode of the file, "r" for read and "w" for write
            num_pre_initialized_h264_streams (int, optional):
                Number of pre-initialized H.264 video streams to use when adding new features.
                we pre initialize a configurable number of H.264 video streams to avoid the overhead of creating new streams for each feature.
                otherwise we need to remux everytime
            . Defaults to 5.
            feature_name_separator (Text, optional):
                Delimiter to separate feature names in the container file.
                Defaults to "/".
        """
        self.path = path
        self.feature_name_separator = feature_name_separator
        # self.cache_file_name = "/tmp/fog_" + os.path.basename(self.path) + ".cache"
        # use hex hash of the path for the cache file name
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        hex_hash = hex(abs(hash(self.path)))[2:]
        self.cache_file_name = cache_dir + hex_hash + ".cache"
        self.feature_name_to_stream = {}  # feature_name: stream
        self.feature_name_to_feature_type = {}  # feature_name: feature_type
        self.trajectory_data = None  # trajectory_data
        self.start_time = time.time()
        self.num_pre_initialized_h264_streams = num_pre_initialized_h264_streams
        self.pre_initialized_image_streams = (
            []
        )  # a list of pre-initialized h264 streams
        self.mode = mode

        # check if the path exists
        # if not, create a new file and start data collection
        if self.mode == "w":
            if not os.path.exists(self.path):
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
            try:
                self.container_file = av.open(self.path, mode="w", format="matroska")
            except Exception as e:
                logger.error(f"error creating the trajectory file: {e}")
                raise
            self._pre_initialize_h264_streams(num_pre_initialized_h264_streams)
        elif self.mode == "r":
            if not os.path.exists(self.path):
                raise FileNotFoundError(f"{self.path} does not exist")
        else:
            raise ValueError(f"Invalid mode {self.mode}, must be 'r' or 'w'")

    def _get_current_timestamp(self):
        current_time = (time.time() - self.start_time) * 1000
        return current_time

    def __len__(self):
        raise NotImplementedError

    def _pre_initialize_h264_streams(self, num_streams: int):
        """
        Pre-initialize a configurable number of H.264 video streams.
        """
        for i in range(num_streams):
            encoding = "libx264"
            stream = self.container_file.add_stream(encoding)
            stream.time_base = Fraction(1, 1000)
            stream.pix_fmt = "yuv420p"
            self.pre_initialized_image_streams.append(stream)

    def __getitem__(self, key):
        """
        get the value of the feature
        return hdf5-ed data
        """
        if self.trajectory_data is None:
            self.trajectory_data = self.load()
        

        return self.trajectory_data[key]

    def close(self):
        """
        close the container file
        """
        try:
            ts = self._get_current_timestamp()
            for stream in self.container_file.streams:
                try:
                    packets = stream.encode(None)
                    for packet in packets:
                        packet.pts = ts
                        packet.dts = ts
                        self.container_file.mux(packet)
                except Exception as e:
                    logger.error(f"Error flushing stream {stream}: {e}")
            logger.debug("Flushing the container file")
        except av.error.EOFError:
            pass  # This exception is expected and means the encoder is fully flushed

        self.container_file.close()

    def load(self):
        """
        load the container file

        workflow:
        - check if a cached mmap/hdf5 file exists
        - if exists, load the file
        - otherwise: load the container file with entire vla trajctory
        """

        if os.path.exists(self.cache_file_name):
            return self._load_from_cache()
        else:
            return self._load_from_container()

    def init_feature_streams(self, feature_spec: Dict):
        """
        initialize the feature stream with the feature name and its type
        args:
            feature_dict: dictionary of feature name and its type
        """
        for feature, feature_type in feature_spec.items():
            encoding = self._get_encoding_of_feature(None, feature_type)
            self.feature_name_to_stream[feature] = self._add_stream_to_container(
                self.container_file, feature, encoding, feature_type
            )

    def add(
        self,
        feature: str,
        data: Any,
        timestamp: Optional[int] = None,
    ) -> None:
        """
        add one value to container file

        Args:
            feature (str): name of the feature
            value (Any): value associated with the feature; except dictionary
            timestamp (optional int): nanoseconds since the Epoch.
                If not provided, the current time is used.

        Examples:
            >>> trajectory.add('feature1', 'image1.jpg')

        Logic:
        - check the feature name
        - if the feature name is not in the container, create a new stream

        - check the type of value
        - if value is numpy array, create a frame and encode it
        - if it is a string or int, create a packet and encode it
        - else raise an error

        Exceptions:
            raise an error if the value is a dictionary
        """

        if type(data) == dict:
            raise ValueError("Use add_by_dict for dictionary")

        feature_type = FeatureType.from_data(data)
        encoding = self._get_encoding_of_feature(data, None)
        self.feature_name_to_feature_type[feature] = feature_type

        # check if the feature is already in the container
        # if not, create a new stream
        # Check if the feature is already in the container
        if feature not in self.feature_name_to_stream:
            self._on_new_stream(feature, encoding, feature_type)

        # get the stream
        stream = self.feature_name_to_stream[feature]

        # get the timestamp
        if timestamp is None:
            timestamp = self._get_current_timestamp()

        # encode the frame
        packets = self._encode_frame(data, stream, timestamp)

        # write the packet to the container
        for packet in packets:
            self.container_file.mux(packet)

    def add_by_dict(
        self,
        data: Dict[str, Any],
        timestamp: Optional[int] = None,
    ) -> None:
        """
        add one value to container file
        data might be nested dictionary of values for each feature

        Args:
            data (Dict[str, Any]): dictionary of feature name and value
            timestamp (optional int): nanoseconds since the Epoch.
                If not provided, the current time is used.
                assume the timestamp is same for all the features within the dictionary

        Examples:
            >>> trajectory.add_by_dict({'feature1': 'image1.jpg'})

        Logic:
        - check the data see if it is a dictionary
        - if dictionary, need to flatten it and add each feature separately
        """
        if type(data) != dict:
            raise ValueError("Use add for non-dictionary data, type is ", type(data))

        _flatten_dict_data = _flatten_dict(data, sep=self.feature_name_separator)
        timestamp = self._get_current_timestamp() if timestamp is None else timestamp
        for feature, value in _flatten_dict_data.items():
            self.add(feature, value, timestamp)

    @classmethod
    def from_list_of_dicts(cls, data: List[Dict[str, Any]], path: Text) -> "Trajectory":
        """
        Create a Trajectory object from a list of dictionaries.

        args:
            data (List[Dict[str, Any]]): list of dictionaries
            path (Text): path to the trajectory file

        Example:
        original_trajectory = [
            {"feature1": "value1", "feature2": "value2"},
            {"feature1": "value3", "feature2": "value4"},
        ]

        trajectory = Trajectory.from_list_of_dicts(original_trajectory, path="/tmp/fog_x/output.vla")
        """
        traj = cls(path, mode="w")
        for step in data:
            traj.add_by_dict(step)
        traj.close()
        return traj

    @classmethod
    def from_dict_of_lists(
        cls, data: Dict[str, List[Any]], path: Text, feature_name_separator: Text = "/"
    ) -> "Trajectory":
        """
        Create a Trajectory object from a dictionary of lists.

        Args:
            data (Dict[str, List[Any]]): dictionary of lists. Assume list length is the same for all features.
            path (Text): path to the trajectory file

        Returns:
            Trajectory: _description_

        Example:
        original_trajectory = {
            "feature1": ["value1", "value3"],
            "feature2": ["value2", "value4"],
        }

        trajectory = Trajectory.from_dict_of_lists(original_trajectory, path="/tmp/fog_x/output.vla")
        """
        traj = cls(path, feature_name_separator=feature_name_separator, mode="w")
        # flatten the data such that all data starts and put feature name with separator
        _flatten_dict_data = _flatten_dict(data, sep=traj.feature_name_separator)

        # Check if all lists have the same length
        list_lengths = [len(v) for v in _flatten_dict_data.values()]
        if len(set(list_lengths)) != 1:
            raise ValueError(
                "All lists must have the same length",
                [(k, len(v)) for k, v in _flatten_dict_data.items()],
            )

        for i in range(list_lengths[0]):
            step = {k: v[i] for k, v in _flatten_dict_data.items()}
            traj.add_by_dict(step)
        traj.close()
        return traj

    def _load_from_cache(self):
        """
        load the cached file with entire vla trajctory
        """
        h5_cache = h5py.File(self.cache_file_name, "r")
        for feature_name, feature_data in h5_cache.items():
            self.feature_name_to_stream[feature_name] = None
            self.feature_name_to_feature_type[feature_name] = FeatureType.from_str(
                feature_data.attrs["FEATURE_TYPE"]
            )
        return h5_cache

    def _load_from_container(self):
        """

        load the container file with entire vla trajctory

        workflow:
        - get schema of the container file
        - preallocate decoded streams
        - decode frame by frame and store in the preallocated memory

        """

        container = av.open(self.path, mode="r", format="matroska")
        h5_cache = h5py.File(self.cache_file_name, "w")
        streams = container.streams

        # preallocate memory for the streams in h5
        for stream in streams:
            if stream.metadata.get("FEATURE_NAME") is None:
                logger.debug(f"Skipping stream without FEATURE_NAME: {stream}")
                continue
            feature_name = stream.metadata["FEATURE_NAME"]
            feature_type = FeatureType.from_str(stream.metadata["FEATURE_TYPE"])
            self.feature_name_to_stream[feature_name] = stream
            self.feature_name_to_feature_type[feature_name] = feature_type
            # Preallocate arrays with the shape [None, X, Y, Z]
            # where X, Y, Z are the dimensions of the feature

            logger.debug(
                f"creating a cache for {feature_name} with shape {feature_type.shape}"
            )

            if feature_type.dtype == "string":
                # strings are not supported in h5py, so we store them as objects
                h5_cache.create_dataset(
                    feature_name,
                    (0,) + feature_type.shape,
                    maxshape=(None,) + feature_type.shape,
                    dtype=h5py.special_dtype(vlen=str),
                )
            else:
                h5_cache.create_dataset(
                    feature_name,
                    (0,) + feature_type.shape,
                    maxshape=(None,) + feature_type.shape,
                    dtype=feature_type.dtype,
                )

        # decode the frames and store in the preallocated memory

        for packet in container.demux(list(streams)):
            if packet.stream.metadata.get("FEATURE_NAME") is None:
                logger.debug(f"Skipping packet without FEATURE_NAME: {packet}")
                continue
            feature_name = packet.stream.metadata["FEATURE_NAME"]
            feature_type = self.feature_name_to_feature_type[feature_name]
            logger.debug(
                f"Decoding {feature_name} with shape {feature_type.shape} and dtype {feature_type.dtype} with time {packet.dts}"
            )
            feature_codec = packet.stream.codec_context.codec.name
            if feature_codec == "h264":
                frames = packet.decode()
                for frame in frames:
                    data = frame.to_ndarray(format="rgb24").reshape(feature_type.shape)
                    h5_cache[feature_name].resize(
                        h5_cache[feature_name].shape[0] + 1, axis=0
                    )
                    h5_cache[feature_name][-1] = data
            else:
                packet_in_bytes = bytes(packet)
                if packet_in_bytes:
                    # decode the packet
                    data = pickle.loads(packet_in_bytes)
                    h5_cache[feature_name].resize(
                        h5_cache[feature_name].shape[0] + 1, axis=0
                    )
                    h5_cache[feature_name][-1] = data
                else:
                    logger.debug(f"Skipping empty packet: {packet}")

        container.close()
        return h5_cache

    def _encode_frame(self, data: Any, stream: Any, timestamp: int) -> List[av.Packet]:
        """
        encode the frame and write it to the stream file, return the packet
        args:
            data: data frame to be encoded
            stream: stream to write the frame
            timestamp: timestamp of the frame
        return:
            packet: encoded packet
        """
        encoding = self._get_encoding_of_feature(data, None)
        feature_type = FeatureType.from_data(data)
        if encoding == "libx264":
            if feature_type.dtype == "float32":
                frame = self._create_frame_depth(data, stream)
            else:
                frame = self._create_frame(data, stream)
            frame.pts = timestamp
            frame.dts = timestamp
            frame.time_base = stream.time_base
            packets = stream.encode(frame)
        else:
            packet = av.Packet(pickle.dumps(data))
            packet.dts = timestamp
            packet.pts = timestamp
            packet.time_base = stream.time_base
            packet.stream = stream

            packets = [packet]

        for packet in packets:
            packet.pts = timestamp
            packet.dts = timestamp
            packet.time_base = stream.time_base
        return packets

    def _on_new_stream(self, new_feature, new_encoding, new_feature_type):
        if new_feature in self.feature_name_to_stream:
            return

        if new_encoding == "libx264":
            # use pre-initialized h264 streams
            if self.pre_initialized_image_streams:
                stream = self.pre_initialized_image_streams.pop()
                stream.metadata["FEATURE_NAME"] = new_feature
                stream.metadata["FEATURE_TYPE"] = str(new_feature_type)
                self.feature_name_to_stream[new_feature] = stream
                return
            else:
                raise ValueError("No pre-initialized h264 streams available")

        if not self.feature_name_to_stream:
            logger.debug(f"Creating a new stream for the first feature {new_feature}")
            self.feature_name_to_stream[new_feature] = self._add_stream_to_container(
                self.container_file, new_feature, new_encoding, new_feature_type
            )
        else:
            logger.debug(f"Adding a new stream for the feature {new_feature}")
            # Following is a workaround because we cannot add new streams to an existing container
            # Close current container
            self.close()

            # Move the original file to a temporary location
            temp_path = self.path + ".temp"
            os.rename(self.path, temp_path)

            # Open the original container for reading
            original_container = av.open(temp_path, mode="r", format="matroska")
            original_streams = list(original_container.streams)

            # Create a new container
            new_container = av.open(self.path, mode="w", format="matroska")

            # reset the pre-initialized h264 streams
            self.pre_initialized_image_streams = []
            # preinitialize  h264 streams
            for i in range(self.num_pre_initialized_h264_streams):
                encoding = "libx264"
                stream = new_container.add_stream(encoding)
                stream.time_base = Fraction(1, 1000)
                self.pre_initialized_image_streams.append(stream)

            # Add existing streams to the new container
            d_original_stream_id_to_new_container_stream = {}
            for stream in original_streams:
                stream_feature = stream.metadata.get("FEATURE_NAME")
                if stream_feature is None:
                    logger.debug(f"Skipping stream without FEATURE_NAME: {stream}")
                    continue
                stream_encoding = self._get_encoding_of_feature(
                    None, self.feature_name_to_feature_type[stream_feature]
                )
                stream_feature_type = self.feature_name_to_feature_type[stream_feature]
                stream_in_updated_container = self._add_stream_to_container(
                    new_container, stream_feature, stream_encoding, stream_feature_type
                )
                # new_stream.options = stream.options
                for key, value in stream.metadata.items():
                    stream_in_updated_container.metadata[key] = value
                d_original_stream_id_to_new_container_stream[stream.index] = (
                    stream_in_updated_container
                )

            # Add new feature stream
            new_stream = self._add_stream_to_container(
                new_container, new_feature, new_encoding, new_feature_type
            )
            d_original_stream_id_to_new_container_stream[new_stream.index] = new_stream

            # Remux existing packets
            for packet in original_container.demux(original_streams):

                def is_packet_valid(packet):
                    return packet.pts is not None and packet.dts is not None

                if is_packet_valid(packet):
                    packet.stream = d_original_stream_id_to_new_container_stream[
                        packet.stream.index
                    ]
                    new_container.mux(packet)
                else:
                    pass

            original_container.close()
            os.remove(temp_path)

            # Reopen the new container for writing new data
            self.container_file = new_container
            self.feature_name_to_stream[new_feature] = new_stream

    def _add_stream_to_container(self, container, feature_name, encoding, feature_type):
        stream = container.add_stream(encoding)
        if encoding == "libx264":
            stream.width = feature_type.shape[0]
            stream.height = feature_type.shape[1]
        stream.metadata["FEATURE_NAME"] = feature_name
        stream.metadata["FEATURE_TYPE"] = str(feature_type)
        stream.time_base = Fraction(1, 1000)
        return stream

    def _create_frame(self, image_array, stream):
        frame = av.VideoFrame.from_ndarray(np.array(image_array, dtype=np.uint8))
        frame.pict_type = "NONE"
        return frame

    def _create_frame_depth(self, image_array, stream):
        image_array = np.array(image_array)
        # if float, convert to uint8
        # TODO: this is a hack, need to fix it
        if image_array.dtype == np.float32:
            image_array = (image_array * 255).astype(np.uint8)
        # if 3 dim, convert to 2 dim
        if len(image_array.shape) == 3:
            image_array = image_array[:, :, 0]
        frame = av.VideoFrame.from_ndarray(image_array, format="gray")
        frame.pict_type = "NONE"
        frame.time_base = stream.time_base
        return frame

    def _get_encoding_of_feature(
        self, feature_value: Any, feature_type: Optional[FeatureType]
    ) -> Text:
        """
        get the encoding of the feature value
        args:
            feature_value: value of the feature
            feature_type: type of the feature
        return:
            encoding of the feature in string
        """
        if feature_type is None:
            feature_type = FeatureType.from_data(feature_value)
        data_shape = feature_type.shape
        if len(data_shape) >= 2 and data_shape[0] >= 100 and data_shape[1] >= 100:
            vid_coding = "libx264"
        else:
            vid_coding = "rawvideo"
        return vid_coding
