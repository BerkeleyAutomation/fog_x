import logging
import time
from typing import Any, Dict, List, Optional, Text
import av 
import numpy as np
import os 
from fog_x import FeatureType 
import pickle
logger = logging.getLogger(__name__)
from fractions import Fraction

class Trajectory:
    def __init__(self, 
                 path: Text) -> None:
        self.path = path

        # check if the path exists 
        # if exists, load the data
        # if not, create a new file
        if os.path.exists(self.path):
            self.container_file = av.open(self.path, mode='r')
        else:
            logger.info(f"creating a new trajectory at {self.path}")
            try:
                # os.makedirs(os.path.dirname(self.path), exist_ok=True)
                # self.container_file = av.open(self.path, mode='w', format = "matroska")
                self.container_file = av.open(self.path, mode='w')
            except Exception as e:
                logger.error(f"error creating the trajectory file: {e}")
                raise
            
        self.feature_name_to_stream = {} # feature_name: stream
        
        self.start_time = time.time()
    
    def _get_current_timestamp(self):
        current_time = (time.time() - self.start_time) * 1000
        return current_time
    
    def __len__(self):
        raise NotImplementedError

    def __iter___(self):
        raise NotImplementedError
    
    def __next__(self):
        raise NotImplementedError

    def close(self):
        """
        close the container file
        """
        try:
            ts = self._get_current_timestamp()
            for stream in self.container_file.streams:
                print(stream)
                try:
                    packets = stream.encode(None) 
                    for packet in packets:
                        packet.pts = ts
                        packet.dts = ts
                        self.container_file.mux(packet)
                except Exception as e:
                    print(e)
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
        self._load_from_container()
        
        
    
    def _load_from_cache(self):
        raise NotImplementedError
    
    def _load_from_container(self):
        """
        
        load the container file with entire vla trajctory
        
        workflow:
        - get schema of the container file
        - preallocate decoded streams 
        - decode frame by frame and store in the preallocated memory

        Raises:
            NotImplementedError: _description_
        """
        
        container = av.open(self.path)
        streams = container.streams
        
        for packet in container.demux(list(streams)):
            print(packet.stream.metadata)
            for frame in packet.decode():
                print(frame)
        
        container.close()

    def init_feature_stream(self, feature_dict: Dict):
        """
        initialize the feature stream with the feature name and its type
        args:
            feature_dict: dictionary of feature name and its type
        """
        for feature, feature_type in feature_dict.items():
            encoding = self.get_encoding_of_feature(None, feature_type)
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
        add one value to video container file 

        Args:
            feature (str): name of the feature
            value (Any): value associated with the feature
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
        """
        # logger.info("Adding Feature name: %s", feature)

        feature_type = FeatureType.from_data(data)
        encoding = self.get_encoding_of_feature(data, None)
        
        # check if the feature is already in the container
        # if not, create a new stream
        # Check if the feature is already in the container
        if feature not in self.feature_name_to_stream:
            self._on_new_stream(feature, encoding, feature_type)
        
        # get the stream
        stream = self.feature_name_to_stream[feature]
        print(f"stream: {stream}")

        # get the timestamp
        if timestamp is None:
            timestamp = self._get_current_timestamp()
        else:
            logger.warning("Using custom timestamp, may cause misalignment")
            
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
        raise NotImplementedError
    
    def _encode_frame(self, 
                      data: Any, 
                      stream: Any, 
                      timestamp: int) -> List[av.Packet]:
        """
        encode the frame and write it to the stream file, return the packet
        args:
            data: data frame to be encoded
            stream: stream to write the frame
            timestamp: timestamp of the frame
        return:
            packet: encoded packet
        """
        encoding = self.get_encoding_of_feature(data, None)
        feature_type = FeatureType.from_data(data)
        if encoding == "libx264":
            if feature_type.dtype == np.float32:
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
            
            print(packet.stream)
            packets = [packet]
            
        for packet in packets:
            packet.pts = timestamp
            packet.dts = timestamp
            packet.time_base = stream.time_base
        return packets

    def _on_new_stream(self, new_feature, encoding, feature_type):
        if new_feature in self.feature_name_to_stream:
            return
        
        if not self.feature_name_to_stream:
            logger.info(f"Creating a new stream for the first feature {new_feature}")
            self.feature_name_to_stream[new_feature] = self._add_stream_to_container(
                self.container_file, new_feature, encoding, feature_type
            )
        else:
            logger.info(f"Adding a new stream for the feature {new_feature}")
            # a workaround because we cannot add new streams to an existing container
            # Close current container
            self.close()
            
            # Move the original file to a temporary location
            temp_path = self.path + ".temp"
            os.rename(self.path, temp_path)
            
            # Open the original container for reading
            original_container = av.open(temp_path, mode='r')
            original_streams = list(original_container.streams)
            print(original_streams)
            
            # Create a new container
            new_container = av.open(self.path, mode='w')
            
            # Add existing streams to the new container
            stream_map = {}
            for stream in original_streams:
                new_stream = new_container.add_stream(template=stream)
                stream_map[stream.index] = new_stream

            # Add new feature stream
            new_stream = self._add_stream_to_container(new_container, new_feature, encoding, feature_type)
            stream_map[new_stream.index] = new_stream
            
            # Remux existing packets
            for packet in original_container.demux(original_streams):
                packet.stream = stream_map[packet.stream.index]
                print(packet)
                def is_packet_valid(packet):
                    return packet.pts is not None and packet.dts is not None
                if is_packet_valid(packet):
                    new_container.mux(packet)
                else:
                    logger.warning(f"Invalid packet: {packet}")
            
            original_container.close()
            os.remove(temp_path)
            
            # Reopen the new container for writing new data
            self.container_file = new_container
            self.feature_name_to_stream[new_feature] = new_stream
            print(self.feature_name_to_stream)
        
    def _add_stream_to_container(self, container, feature_name, encoding, feature_type):
        stream = container.add_stream(encoding)
        if encoding == "libx264":
            stream.width = feature_type.shape[0]
            stream.height = feature_type.shape[1]
        stream.metadata['feature_name'] = feature_name
        stream.metadata['feature_type'] = str(feature_type)
        stream.time_base = Fraction(1, 1000)
        return stream
        
    
    def _create_frame(self, image_array, stream):
        frame = av.VideoFrame.from_ndarray(np.array(image_array, dtype=np.uint8), format='rgb24')
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
            image_array = image_array[:,:,0]
        frame = av.VideoFrame.from_ndarray(image_array, format='gray')
        frame.pict_type = 'NONE'
        frame.time_base = stream.time_base
        return frame

    def get_encoding_of_feature(self, feature_value : Any, feature_type: Optional[FeatureType]) -> Text:
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
        

