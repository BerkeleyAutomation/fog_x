import logging
import time
from typing import Any, Dict, List, Optional, Text
import av 
import numpy as np
import os 
from fog_x import FeatureType 

logger = logging.getLogger(__name__)


class Trajectory:
    def __init__(self, 
                 path: Text) -> None:
        self.path = path
        
        # check if the path exists 
        # if exists, load the data
        # if not, create a new file
        if os.path.exists(self.path):
            self.vid_output_file = av.open(path, mode='r')
        else:
            self._create_container_file()
    
    def __len__(self):
        raise NotImplementedError

    def __iter___(self):
        raise NotImplementedError
    
    def __next__(self):
        raise NotImplementedError

    def _create_container_file(self):
        self.vid_output_file = av.open(self.path, mode='w')
        self.features = {} # feature_name: feature_type


    def add(
        self,
        feature: str,
        value: Any,
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

        # check if the feature is already in the container
        # if not, create a new stream
        if feature not in self.features:
            self.features[feature] = self.vid_output_file.add_stream(
                feature, 
                rate=1
            )




        # if isinstance(type_val, np.ndarray):
                
        #     if structure.vid_coding == "libx264":
        #         if type_val.dtype == np.float32:
        #             frame = self._create_frame_depth(step_val, streams_d)
        #         else:
        #             frame = self._create_frame(step_val, streams_d, ts)
        #         frame.dts = ts
        #         packet = streams_d.encode(frame)
        #     else:
        #         packet = av.Packet(pickle.dumps(step_val))
        #         packet.stream = streams_d
        #         packet.dts = ts
                
        #     self.vid_output.mux(packet)
        # else:
        #     raise ValueError(f"{type(type_val)} not supported")
    
    def add_by_dict(
        self,
        data: Dict[str, Any],
        timestamp: Optional[int] = None,
    ) -> None:
        raise NotImplementedError
    

    def load(self):
        raise NotImplementedError

    def create_frame(self, image_array, stream, frame_index):
        frame = av.VideoFrame.from_ndarray(np.array(image_array), format='rgb24')
        frame.pict_type = 'NONE'
        frame.time_base = stream.time_base
        frame.dts = frame_index
        return frame
    
    # Function to create a frame from numpy array
    def create_frame_depth(self, image_array, stream):
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
