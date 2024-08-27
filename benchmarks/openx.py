import fog_x
import os
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor
import glob
import time
import numpy as np
from fog_x.loader import RLDSLoader, VLALoader, HDF5Loader
import tensorflow as tf  # this prevents tensorflow printed logs
import pandas as pd
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Constants
DEFAULT_EXP_DIR = "/home/kych/datasets/fog_x/"
DEFAULT_NUMBER_OF_TRAJECTORIES = 1024
DEFAULT_DATASET_NAMES = ["berkeley_autolab_ur5", "bridge", "berkeley_cable_routing", "nyu_door_opening_surprising_effectiveness"]
# DEFAULT_NUMBER_OF_TRAJECTORIES = 1000
# DEFAULT_DATASET_NAMES = ["berkeley_autolab_ur5"]
CACHE_DIR = "/tmp/fog_x/cache/"


class DatasetHandler:
    """Base class to handle dataset-related operations."""

    DATA_URL_TEMPLATE = "gs://gresearch/robotics/{dataset_name}/0.1.0/{dataset_name}-train.tfrecord-{index:05d}-of-{total_trajectories:05d}"
    LS_URL_TEMPLATE = "gs://gresearch/robotics/{dataset_name}/0.1.0/{dataset_name}-train.tfrecord-*"
    LOCAL_FILE_TEMPLATE = "{exp_dir}/{dataset_type}/{dataset_name}/{dataset_name}-train.tfrecord-{index:05d}-of-{total_trajectories:05d}"
    FEATURE_JSON_URL_TEMPLATE = (
        "gs://gresearch/robotics/{dataset_name}/0.1.0/features.json"
    )
    DATASET_INFO_JSON_URL_TEMPLATE = (
        "gs://gresearch/robotics/{dataset_name}/0.1.0/dataset_info.json"
    )

    def __init__(self, exp_dir, dataset_name, num_trajectories, dataset_type):
        self.exp_dir = exp_dir
        self.dataset_name = dataset_name
        self.total_trajectories = self._get_total_number_of_trajectories()
        self.num_trajectories = num_trajectories if num_trajectories <= self.total_trajectories else self.total_trajectories
        self.dataset_type = dataset_type
        self.dataset_dir = os.path.join(exp_dir, dataset_type, dataset_name)
        
    def _get_total_number_of_trajectories(self):
        """Gets the total number of trajectories in the dataset."""
        # use gsutil to get a trajectory file name and extract the total number of trajectories
        data_url = self.LS_URL_TEMPLATE.format(
            dataset_name=self.dataset_name, index=0,
            total_trajectories="*"
        )
        output = subprocess.run(
            ["gsutil", "ls", data_url], stdout=subprocess.PIPE, check=True
        )
        total_trajectories = int(output.stdout.decode().split("-")[-1])
        
        return total_trajectories
    def clear_cache(self):
        """Clears the cache directory."""
        if os.path.exists(CACHE_DIR):
            subprocess.run(["rm", "-rf", CACHE_DIR], check=True)

    def clear_os_cache(self):
        """Clears the OS cache."""
        subprocess.run(["sync"], check=True)
        subprocess.run(["echo", "3", ">", "/proc/sys/vm/drop_caches"], check=True)

    def check_and_download_file(self, url, local_path):
        """Checks if a file is already downloaded; if not, downloads it."""
        if not os.path.exists(local_path):
            subprocess.run(["gsutil", "-m", "cp", url, local_path], check=True)
        else:
            print(f"File {local_path} already exists. Skipping download.")

    def check_and_download_trajectory(self, trajectory_index):
        """Checks if a trajectory and associated JSON files are already downloaded; if not, downloads them."""
        os.makedirs(self.dataset_dir, exist_ok=True)

        # Check and download the trajectory files
        local_file_pattern = self.LOCAL_FILE_TEMPLATE.format(
            exp_dir=self.exp_dir,
            dataset_type=self.dataset_type,
            dataset_name=self.dataset_name,
            index=trajectory_index,
            total_trajectories = self.total_trajectories
        )

        # Ensure no files with .gstmp postfix are considered valid
        valid_files_exist = any(
            os.path.exists(file) and not file.endswith(".gstmp")
            for file in glob.glob(local_file_pattern)
        )

        if not valid_files_exist:
            data_url = self.DATA_URL_TEMPLATE.format(
                dataset_name=self.dataset_name, index=trajectory_index,
                total_trajectories=self.total_trajectories
            )
            subprocess.run(
                ["gsutil", "-m", "cp", data_url, self.dataset_dir], check=True
            )
        else:
            print(
                f"Trajectory {trajectory_index} of dataset {self.dataset_name} already exists in {self.dataset_dir}. Skipping download."
            )

        # Check and download the feature.json file
        feature_json_local_path = os.path.join(self.dataset_dir, "features.json")
        feature_json_url = self.FEATURE_JSON_URL_TEMPLATE.format(
            dataset_name=self.dataset_name,
        )
        self.check_and_download_file(feature_json_url, feature_json_local_path)

        # Check and download the dataset_info.json file
        dataset_info_json_local_path = os.path.join(
            self.dataset_dir, "dataset_info.json"
        )
        dataset_info_json_url = self.DATASET_INFO_JSON_URL_TEMPLATE.format(
            dataset_name=self.dataset_name
        )
        self.check_and_download_file(
            dataset_info_json_url, dataset_info_json_local_path
        )

    def download_data(self):
        """Downloads the specified number of trajectories from the dataset concurrently if not already downloaded."""
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.check_and_download_trajectory, i)
                for i in range(self.num_trajectories)
            ]
            for future in futures:
                future.result()

    def measure_file_size(self):
        """Calculates the total size of all files in the dataset directory."""
        total_size = sum(
            os.path.getsize(os.path.join(dirpath, f))
            for dirpath, dirnames, filenames in os.walk(self.dataset_dir)
            for f in filenames
        )
        return total_size

    def measure_file_size_per_trajectory(self):
        """Calculates the size of each trajectory file in the dataset directory."""
        trajectory_sizes = []
        for dirpath, dirnames, filenames in os.walk(self.dataset_dir):
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                file_size = os.path.getsize(file_path)
                trajectory_sizes.append(file_size)
        return trajectory_sizes

class RLDSHandler(DatasetHandler):
    """Handles RLDS dataset operations, including loading and measuring loading times."""

    def __init__(self, exp_dir, dataset_name, num_trajectories):
        super().__init__(exp_dir, dataset_name, num_trajectories, dataset_type="rlds")

    def measure_loading_time(self):
        """Measures the time taken to load data into memory using RLDSLoader."""
        start_time = time.time()
        loader = RLDSLoader(self.dataset_dir, split=f"train[:{self.num_trajectories}]")
        for data in loader:
            print("length of loaded data", len(data))

        end_time = time.time()
        loading_time = end_time - start_time
        print(
            f"Loaded {len(loader)} trajectories in {loading_time:.2f} seconds start time {start_time} end time {end_time}"
        )
        return loading_time, len(loader)

    def measure_loading_time_per_trajectory(self):
        """Measures the time taken to load each trajectory separately."""
        times = []
        loader = RLDSLoader(self.dataset_dir, split=f"train[:{self.num_trajectories}]")
        for data in loader:
            start_time = time.time()
            l = list(data)
            for i in l:
                # recursively load all data
                def _recursively_load_data(data):
                    for key in data.keys():
                        if isinstance(data[key], dict):
                            _recursively_load_data(data[key])
                        else:
                            (key, np.array(data[key]))
                            (key, np.array(data[key]).shape)
                _recursively_load_data(i)
            # print("length of loaded data", len(l))
            end_time = time.time()
            loading_time = end_time - start_time
            times.append(loading_time)
            print(
                f"Loaded 1 trajectory in {loading_time:.2f} seconds start time {start_time} end time {end_time}"
            )
        return times

class VLAHandler(DatasetHandler):
    """Handles VLA dataset operations, including loading, converting, and measuring loading times."""

    def __init__(self, exp_dir, dataset_name, num_trajectories):
        super().__init__(exp_dir, dataset_name, num_trajectories, dataset_type="vla")
        self.trajectories_objects = []

    def convert_data_to_vla_format(self, loader):
        """Converts data to VLA format and saves it to the same directory."""
        for index, data_traj in enumerate(loader):
            output_path = os.path.join(self.dataset_dir, f"output_{index}.vla")
            fog_x.Trajectory.from_list_of_dicts(data_traj, path=output_path)

    def measure_loading_time_per_trajectory(self, save_trajectorie_objects=False, mode = "no_cache"):
        """Measures the time taken to load each trajectory separately using VLALoader."""
        times = []
        loader = VLALoader(self.dataset_dir, cache_dir=CACHE_DIR)
        for data in loader:
            start_time = time.time()
            self._recursively_load_h5_data(data.load(mode = mode))
            if save_trajectorie_objects:
                self.trajectories_objects.append(data)
            end_time = time.time()
            loading_time = end_time - start_time
            times.append(loading_time)
            print(
                f"Loaded 1 trajectory in {loading_time:.2f} seconds start time {start_time} end time {end_time}"
            )
        return times
    def _recursively_load_h5_data(self, data):
        for key in data.keys():
            if isinstance(data[key], dict):
                self._recursively_load_h5_data(data[key])
            else:
                (key, np.array(data[key]))
                (key, np.array(data[key]).shape)

class HDF5Handler(DatasetHandler):
    """Handles HDF5 dataset operations, including conversion and measuring file sizes."""

    def __init__(self, exp_dir, dataset_name, num_trajectories):
        super().__init__(exp_dir, dataset_name, num_trajectories, dataset_type="hdf5")
        self.hdf5_dir = os.path.join(exp_dir, "hdf5", dataset_name)
        if not os.path.exists(self.hdf5_dir):
            os.makedirs(self.hdf5_dir)

    def convert_data_to_hdf5(self, trajectories_objects):
        """Converts data to HDF5 format and saves it to the same directory."""
        print(f"Converting {len(trajectories_objects)} trajectories to HDF5 format.")
        for index, trajectory in enumerate(trajectories_objects):
            trajectory.to_hdf5(path=f"{self.hdf5_dir}/output_{index}.h5")

    def measure_file_size(self):
        """Calculates the total size of all files in the HDF5 directory."""
        total_size = sum(
            os.path.getsize(os.path.join(dirpath, f))
            for dirpath, dirnames, filenames in os.walk(self.hdf5_dir)
            for f in filenames
        )
        return total_size

    def measure_loading_time(self):
        """Measures the time taken to load data into memory using HDF5Loader."""
        start_time = time.time()
        loader = HDF5Loader(path=os.path.join(self.hdf5_dir, "*.h5"))

        def _recursively_load_h5_data(data):
            for key in data.keys():
                if isinstance(data[key], dict):
                    _recursively_load_h5_data(data[key])
                else:
                    (key, np.array(data[key]))
                    (key, np.array(data[key]).shape)

        count = 0
        for data in loader:
            # recursively load all data
            _recursively_load_h5_data(data)
            count += 1

        end_time = time.time()
        loading_time = end_time - start_time
        print(
            f"Loaded {count} trajectories in {loading_time:.2f} seconds start time {start_time} end time {end_time}"
        )
        return loading_time, count


    def measure_loading_time_per_trajectory(self):
        """Measures the time taken to load each trajectory separately using HDF5Loader."""
        times = []
        loader = HDF5Loader(path=os.path.join(self.hdf5_dir, "*.h5"))
        for data in loader:
            start_time = time.time()
            self._recursively_load_h5_data(data)
            end_time = time.time()
            loading_time = end_time - start_time
            times.append(loading_time)
            print(
                f"Loaded 1 trajectory in {loading_time:.2f} seconds start time {start_time} end time {end_time}"
            )
        return times

    def _recursively_load_h5_data(self, data):
        for key in data.keys():
            if isinstance(data[key], dict):
                self._recursively_load_h5_data(data[key])
            else:
                (key, np.array(data[key]))
                (key, np.array(data[key]).shape)
                
def prepare():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Download, process, and read RLDS data."
    )
    parser.add_argument(
        "--exp_dir", type=str, default=DEFAULT_EXP_DIR, help="Experiment directory."
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=DEFAULT_NUMBER_OF_TRAJECTORIES,
        help="Number of trajectories to download.",
    )
    parser.add_argument(
        "--dataset_names",
        nargs="+",
        default=DEFAULT_DATASET_NAMES,
        help="List of dataset names to download.",
    )
    args = parser.parse_args()

    for dataset_name in args.dataset_names:
        print(f"Processing dataset: {dataset_name}")

        # Clear the cache directory
        cache_dir = CACHE_DIR
        if os.path.exists(cache_dir):
            subprocess.run(["rm", "-rf", cache_dir], check=True)

        # Process RLDS data
        rlds_handler = RLDSHandler(args.exp_dir, dataset_name, args.num_trajectories)
        rlds_handler.download_data()

        # Prepare VLA data
        vla_handler = VLAHandler(args.exp_dir, dataset_name, args.num_trajectories)
        loader = RLDSLoader(
            rlds_handler.dataset_dir, split=f"train[:{args.num_trajectories}]"
        )
        vla_handler.convert_data_to_vla_format(loader)


def evaluation():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Download, process, and read RLDS data."
    )
    parser.add_argument(
        "--exp_dir", type=str, default=DEFAULT_EXP_DIR, help="Experiment directory."
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=DEFAULT_NUMBER_OF_TRAJECTORIES,
        help="Number of trajectories to download.",
    )
    parser.add_argument(
        "--dataset_names",
        nargs="+",
        default=DEFAULT_DATASET_NAMES,
        help="List of dataset names to download.",
    )
    args = parser.parse_args()

    results = []

    for dataset_name in args.dataset_names:
        print(f"Processing dataset: {dataset_name}")

        # Clear the cache directory
        cache_dir = CACHE_DIR
        if os.path.exists(cache_dir):
            subprocess.run(["rm", "-rf", cache_dir], check=True)

        # Process RLDS data
        rlds_handler = RLDSHandler(args.exp_dir, dataset_name, args.num_trajectories)
        rlds_sizes = rlds_handler.measure_file_size_per_trajectory()
        rlds_handler.clear_os_cache()
        rlds_loading_times = rlds_handler.measure_loading_time_per_trajectory()

        for i, (size, time) in enumerate(zip(rlds_sizes, rlds_loading_times)):
            results.append({
                'Dataset': dataset_name,
                'Format': 'RLDS',
                'Trajectory': i,
                'LoadingTime(s)': time,
                'FileSize(MB)': size / (1024 * 1024),
                'Throughput(traj/s)': 1 / time if time > 0 else 0
            })

        # Process VLA data
        vla_handler = VLAHandler(args.exp_dir, dataset_name, args.num_trajectories)
        vla_sizes = vla_handler.measure_file_size_per_trajectory()
        
        # first, no cache test, directly reading everything to memory
        # no side effect 
        vla_handler.clear_os_cache()
        vla_loading_times = vla_handler.measure_loading_time_per_trajectory(save_trajectorie_objects=False, mode = "no_cache")

        for i, (size, time) in enumerate(zip(vla_sizes, vla_loading_times)):
            results.append({
                'Dataset': dataset_name,
                'Format': 'VLA-NoCache',
                'Trajectory': i,
                'LoadingTime(s)': time,
                'FileSize(MB)': size / (1024 * 1024),
                'Throughput(traj/s)': 1 / time if time > 0 else 0
            })
        
        
        
        vla_handler.clear_os_cache()
        vla_loading_times = vla_handler.measure_loading_time_per_trajectory(save_trajectorie_objects=True, mode = "cache")

        for i, (size, time) in enumerate(zip(vla_sizes, vla_loading_times)):
            results.append({
                'Dataset': dataset_name,
                'Format': 'VLA-ColdCache',
                'Trajectory': i,
                'LoadingTime(s)': time,
                'FileSize(MB)': size / (1024 * 1024),
                'Throughput(traj/s)': 1 / time if time > 0 else 0
            })
        
        vla_handler.clear_os_cache()
        # hot cache test
        vla_loading_times = vla_handler.measure_loading_time_per_trajectory(save_trajectorie_objects=False, mode = "cache")

        for i, (size, time) in enumerate(zip(vla_sizes, vla_loading_times)):
            results.append({
                'Dataset': dataset_name,
                'Format': 'VLA-HotCache',
                'Trajectory': i,
                'LoadingTime(s)': time,
                'FileSize(MB)': size / (1024 * 1024),
                'Throughput(traj/s)': 1 / time if time > 0 else 0
            })
        

        # Convert VLA to HDF5 and benchmark
        hdf5_handler = HDF5Handler(args.exp_dir, dataset_name, args.num_trajectories)
        hdf5_handler.convert_data_to_hdf5(vla_handler.trajectories_objects)
        hdf5_sizes = hdf5_handler.measure_file_size_per_trajectory()
        hdf5_handler.clear_os_cache()
        hdf5_loading_times = hdf5_handler.measure_loading_time_per_trajectory()

        for i, (size, time) in enumerate(zip(hdf5_sizes, hdf5_loading_times)):
            results.append({
                'Dataset': dataset_name,
                'Format': 'HDF5',
                'Trajectory': i,
                'LoadingTime(s)': time,
                'FileSize(MB)': size / (1024 * 1024),
                'Throughput(traj/s)': 1 / time if time > 0 else 0
            })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('trajectory_results.csv', index=False)
    print("Results written to trajectory_results.csv")



if __name__ == "__main__":
    prepare()
    exit()
    evaluation()
