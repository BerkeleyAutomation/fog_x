import logging
import time
from typing import Any, Dict, List, Optional, Text
import av 
import numpy as np
import os 
from fog_x import FeatureType 
import pickle
logger = logging.getLogger(__name__)


class Trajectory:
    def __init__(self, 
                 path: Text) -> None:
        self.path = path
        self.container_file = av.open(self.path, mode='w')
        
        # check if the path exists 
        # if exists, load the data
        # if not, create a new file
        # if os.path.exists(self.path):
        #     self.container_file = av.open(self.path, mode='w', format = "matroska")
        # else:
        #     logger.info(f"creating a new trajectory at {self.path}")
        #     try:
        #         # os.makedirs(os.path.dirname(self.path), exist_ok=True)
        #         io_handle = open(self.path, 'w')
        #         self.container_file = av.open(self.path, mode='w', format = "matroska", io_open = io_handle)
        #     except Exception as e:
        #         logger.error(f"error creating the trajectory file: {e}")
        #         raise
            
        self.feature_name_to_stream = {} # feature_name: stream
    
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
            for stream in self.container_file.streams:
                for packet in stream.encode():
                    self.container_file.mux(packet)
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
            for frame in packet.decode():
                print(frame)
        
        raise NotImplementedError


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

        feature_type = FeatureType.from_data(data)
        encoding = self.get_encoding_of_feature(data, None)
        
        # check if the feature is already in the container
        # if not, create a new stream
        if feature not in self.feature_name_to_stream:
            logger.info("Adding Feature name: %s, Feature type: %s, Encoding: %s", feature, feature_type, encoding)
            stream = self.container_file.add_stream(
                encoding, 
                rate=1
            )
            stream.metadata['feature_name'] = feature
            stream.metadata['feature_type'] = str(feature_type)
            self.feature_name_to_stream[feature] = stream
            
        
        # get the stream
        stream = self.feature_name_to_stream[feature]

        # get the timestamp
        if timestamp is None:
            timestamp = time.time_ns()
        
        # encode the frame
        packet = self._encode_frame(data, stream, timestamp)

        # write the packet to the container
        if packet:
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
                      timestamp: int) -> av.Packet:
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
                frame = self._create_frame(data, stream, timestamp)
            frame.dts = timestamp
            packet = stream.encode(frame)
        else:
            packet = av.Packet(pickle.dumps(data))
            packet.stream = stream
            packet.dts = timestamp
        return packet

    def _create_frame(self, image_array, stream, frame_index):
        frame = av.VideoFrame.from_ndarray(np.array(image_array, dtype=np.uint8), format='rgb24')
        return frame
    
    def _create_frame_depth(self, image_array, stream):
        image_array = np.array(image_array)
        # if float, convert to uint8
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
        

