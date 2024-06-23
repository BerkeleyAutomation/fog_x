import os
import av
import sys
import time
import pickle
import numpy as np
from logging import getLogger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow_datasets as tfds
import tensorflow as tf

PATH = os.path.expanduser("~")
sys.path.append(PATH + "/fog_x_fork")
PATH += "/datasets/"


class BaseLoader():
    def __init__(self, path):
        super(BaseLoader, self).__init__()
        self.path = path
        self.logger = getLogger(__name__)

    def __len__(self):
        raise NotImplementedError

    def __iter___(self):
        raise NotImplementedError


class BaseWriter():
    def __init__(self):
        super(BaseWriter, self).__init__()
        self.logger = getLogger(__name__)

    def export(self, loader, path):
        raise NotImplementedError


class TFLoader(BaseLoader):
    def __init__(self, path, split):
        super(TFLoader, self).__init__(path)

        builder = tfds.builder_from_directory(path)
        self.ds = builder.as_dataset(split)

    def __len__(self):
        return len(self.ds)
    
    def __iter__(self):
        return self.ds.__iter__()


class TFWriter(BaseWriter):
    def __init__(self):
        super(TFWriter, self).__init__()
        self.wr = tf.io.TFRecordWriter

    def export(self, loader, path):
        for i, data in enumerate(loader): 
            data_list = list(data["steps"].as_numpy_iterator())

            with self.wr(f"{path}/TFWriter_{i}.tfrecord") as writer:
                for dl in data_list:
                    writer.write(dl)

    def export_temp(self, data_list, path):
        path += "/temp.tfrecord"

        with self.wr(path) as writer:
            for dl in data_list:
                writer.write(dl)
        os.remove(path)


N = 51
MB = 1024 * 1024

def get_size(obj):
    if isinstance(obj, dict):
        obj = list(obj.values())
    if isinstance(obj, list):
        return sum(get_size(v) for v in obj)
    elif isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, bytes):
        return len(obj)
    if np.isscalar(obj):
        return obj.itemsize
    raise TypeError(type(obj))


def iterate_dataset(loader, writer, n):
    read_time, write_time, data_size = 0, 0, 0

    for i, data in enumerate(loader): 
        stop = time.time()
        data_list = list(data["steps"].as_numpy_iterator())
        read_time += time.time() - stop

        data_size += get_size(data_list)

        stop = time.time()
        writer.export_temp(data_list, PATH)
        write_time += time.time() - stop

        if i == n: break
    return read_time, write_time, data_size / MB


tf_loader = TFLoader(PATH + "berkeley_autolab_ur5", f"train[:{N}]")
tf_writer = TFWriter()
iterate_dataset(tf_loader, N)
print("do we even get here")
exit()

class MKVExporter(BaseWriter):
    def __init__(self):
        super(MKVExporter, self).__init__()

    # Function to create a frame from numpy array
    def create_frame(self, image_array, stream):
        frame = av.VideoFrame.from_ndarray(np.array(image_array), format='rgb24')
        frame.pict_type = 'NONE'
        frame.time_base = stream.time_base
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

    def export(self, loader: BaseLoader, output_path: str):
        # Create an output container
        i = -1
        for traj_tensor in loader:
            i += 1
            trajectory = dict(traj_tensor)
            output = av.open(f'{output_path}/output_{i}.mkv', mode='w')
            # Define video streams (assuming images are 640x480 RGB)
            video_stream_1 = output.add_stream('libx264', rate=1)
            video_stream_1.width = 640
            video_stream_1.height = 480
            video_stream_1.pix_fmt = 'yuv420p'

            video_stream_2 = output.add_stream('libx264', rate=1)
            video_stream_2.width = 640
            video_stream_2.height = 480
            video_stream_2.pix_fmt = 'yuv420p'

            # Define custom data stream for vectors
            depth_stream = output.add_stream('libx264', rate=1)

            data_stream = output.add_stream('rawvideo', rate=1)

            ts = 0
            # convert step data to stream
            for step_tensor in trajectory["steps"]:
                step = dict(step_tensor)
                obesrvation = step_tensor["observation"].copy()
                obesrvation.pop("image")
                obesrvation.pop("hand_image")
                obesrvation.pop("image_with_depth")
                non_image_data_step = step.copy()
                non_image_data_step["observation"] = obesrvation

                non_image_data_bytes = pickle.dumps(non_image_data_step)
                packet = av.Packet(non_image_data_bytes)
                packet.stream = data_stream
                packet.pts = ts
                output.mux(packet)


                image =np.array(step["observation"]["image"])
                # Create a frame from the numpy array
                frame = self.create_frame(image, video_stream_1)
                frame.pts = ts
                packet = video_stream_1.encode(frame)
                
                output.mux(packet)

                hand_image =np.array(step["observation"]["hand_image"])
                # Create a frame from the numpy array
                frame = self.create_frame(hand_image, video_stream_2)
                frame.pts = ts
                packet = video_stream_2.encode(frame)
                output.mux(packet)

                # # Create a frame from the numpy array
                frame = self.create_frame_depth(step["observation"]["image_with_depth"], depth_stream)
                # frame.pts = ts
                # Encode the frame
                packet = depth_stream.encode(frame)
                # Write the packet to the output file
                output.mux(packet)

                ts += 1

            
            # Flush the remaining frames
            for packet in video_stream_1.encode():
                output.mux(packet)

            for packet in video_stream_2.encode():
                output.mux(packet)

            for packet in depth_stream.encode():
                output.mux(packet)
            output.close()


class MKVLoader(BaseLoader):
    def __init__(self, data_path):
        super(MKVLoader, self).__init__(data_path)
        self.files = [data_path + f for f in os.listdir(data_path) if f.endswith('.mkv')]
        self.index = 0

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return idx
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.files):
            result = self.files[self.index]
            self.index += 1
            return self._parse_mkv_file(result)
        else:
            raise StopIteration
    
    def _parse_mkv_file(self, filename):
        print(filename)
        input_container = av.open(filename)
        video_stream1 = input_container.streams.video[0] 
        video_stream1.thread_type = 'AUTO'
        video_stream2 = input_container.streams.video[1] 
        video_stream2.thread_type = 'AUTO'
        depth_stream = input_container.streams.video[2] 
        depth_stream.thread_type = 'AUTO'
        data_stream = input_container.streams[3] 

        decoded_stream_1 = []
        decoded_stream_2 = []
        decoded_stream_depth = []
        decoded_stream_data = []

        pkt_counter = 0
        for packet in input_container.demux(video_stream1, video_stream2, depth_stream, data_stream):
            pkt_counter += 1
            if packet.stream.index == video_stream1.index: 
                frame = packet.decode()
                if frame:
                    for f in frame:
                        image = f.to_ndarray(format='rgb24')
                        decoded_stream_1.append(image)
            elif packet.stream.index == video_stream2.index:
                frame = packet.decode()
                if frame:
                    for f in frame:
                        image = f.to_ndarray(format='rgb24')
                        decoded_stream_2.append(image)
            elif packet.stream.index == depth_stream.index:
                frame = packet.decode()
                if frame:
                    for f in frame:
                        image = f.to_ndarray(format='gray')
                        decoded_stream_depth.append(image)
            elif packet.stream.index == data_stream.index:
                packet_in_bytes = bytes(packet)
                if packet_in_bytes:
                    non_dict = pickle.loads(packet_in_bytes)
                    decoded_stream_data.append(non_dict)
            else:
                print("Unknown stream")
        print(pkt_counter, len(decoded_stream_1), len(decoded_stream_2), len(decoded_stream_depth), len(decoded_stream_data))
        input_container.close()


exporter = MKVExporter()
output_path = os.path.expanduser("~") + "/fog_x/examples/dataloader/mkv_output/"
exporter.export(rtx_loader, output_path)
mkv_loader = MKVLoader(output_path)
def iterate_dataset(loader: BaseLoader, number_of_samples = 50):
    for i, data in enumerate(loader): 
        if i == number_of_samples:
            break
iterate_dataset(mkv_loader, number_of_samples)