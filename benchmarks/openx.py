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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Constants
DEFAULT_EXP_DIR = "/tmp/fog_x"
DEFAULT_NUMBER_OF_TRAJECTORIES = 1
DEFAULT_DATASET_NAMES = ["berkeley_autolab_ur5"]
CACHE_DIR = "/tmp/fog_x/cache/"


class DatasetHandler:
    """Base class to handle dataset-related operations."""

    DATA_URL_TEMPLATE = "gs://gresearch/robotics/{dataset_name}/0.1.0/{dataset_name}-train.tfrecord-{index:05d}-*"
    LOCAL_FILE_TEMPLATE = "{exp_dir}/{dataset_type}/{dataset_name}/{dataset_name}-train.tfrecord-{index:05d}-*"
    FEATURE_JSON_URL_TEMPLATE = (
        "gs://gresearch/robotics/{dataset_name}/0.1.0/features.json"
    )
    DATASET_INFO_JSON_URL_TEMPLATE = (
        "gs://gresearch/robotics/{dataset_name}/0.1.0/dataset_info.json"
    )

    def __init__(self, exp_dir, dataset_name, num_trajectories, dataset_type):
        self.exp_dir = exp_dir
        self.dataset_name = dataset_name
        self.num_trajectories = num_trajectories
        self.dataset_type = dataset_type
        self.dataset_dir = os.path.join(exp_dir, dataset_type, dataset_name)

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
        )

        # Ensure no files with .gstmp postfix are considered valid
        valid_files_exist = any(
            os.path.exists(file) and not file.endswith(".gstmp")
            for file in glob.glob(local_file_pattern)
        )

        if not valid_files_exist:
            data_url = self.DATA_URL_TEMPLATE.format(
                dataset_name=self.dataset_name, index=trajectory_index
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
            dataset_name=self.dataset_name
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


class VLAHandler(DatasetHandler):
    """Handles VLA dataset operations, including loading, converting, and measuring loading times."""

    def __init__(self, exp_dir, dataset_name, num_trajectories):
        super().__init__(exp_dir, dataset_name, num_trajectories, dataset_type="vla")
        self.trajectories_objects = []

    def measure_loading_time(self, is_add_to_trajectories=False):
        """Measures the time taken to load data into memory using VLALoader."""

        def _recursively_load_h5_data(data):
            for key in data.keys():
                if isinstance(data[key], dict):
                    _recursively_load_h5_data(data[key])
                else:
                    (key, np.array(data[key]))
                    (key, np.array(data[key]).shape)

        start_time = time.time()
        loader = VLALoader(self.dataset_dir, cache_dir=CACHE_DIR)
        for data in loader:
            _recursively_load_h5_data(data.load())
            if is_add_to_trajectories:
                self.trajectories_objects.append(data)

        end_time = time.time()
        loading_time = end_time - start_time
        print(
            f"Loaded {len(loader)} trajectories in {loading_time:.2f} seconds start time {start_time} end time {end_time}"
        )
        return loading_time, len(loader)

    def convert_data_to_vla_format(self, loader):
        """Converts data to VLA format and saves it to the same directory."""
        for index, data_traj in enumerate(loader):
            output_path = os.path.join(self.dataset_dir, f"output_{index}.vla")
            fog_x.Trajectory.from_list_of_dicts(data_traj, path=output_path)


class HDF5Handler:
    """Handles HDF5 dataset operations, including conversion and measuring file sizes."""

    def __init__(self, exp_dir, dataset_name):
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

    for dataset_name in args.dataset_names:
        print(f"Processing dataset: {dataset_name}")

        # Clear the cache directory
        cache_dir = CACHE_DIR
        if os.path.exists(cache_dir):
            subprocess.run(["rm", "-rf", cache_dir], check=True)

        # Process RLDS data
        rlds_handler = RLDSHandler(args.exp_dir, dataset_name, args.num_trajectories)
        rlds_file_size = rlds_handler.measure_file_size()
        rlds_loading_time, num_loaded_rlds = rlds_handler.measure_loading_time()

        print(f"Total RLDS file size: {rlds_file_size / (1024 * 1024):.2f} MB")
        print(
            f"RLDS format loading time for {num_loaded_rlds} trajectories: {rlds_loading_time:.2f} seconds"
        )
        print(
            f"RLDS format throughput: {num_loaded_rlds / rlds_loading_time:.2f} trajectories per second"
        )

        # # Process VLA data
        vla_handler = VLAHandler(args.exp_dir, dataset_name, args.num_trajectories)
        vla_loading_time, num_loaded_vla = vla_handler.measure_loading_time(
            is_add_to_trajectories=True
        )
        vla_file_size = vla_handler.measure_file_size()
        print(f"Total VLA file size: {vla_file_size / (1024 * 1024):.2f} MB")
        print(
            f"VLA format loading time for {num_loaded_vla} trajectories: {vla_loading_time:.2f} seconds"
        )
        print(
            f"VLA format throughput: {num_loaded_vla / vla_loading_time:.2f} trajectories per second\n"
        )

        vla_handler.clear_os_cache()
        # hot cache VLA loading time
        vla_loading_time, num_loaded_vla = vla_handler.measure_loading_time(
            is_add_to_trajectories=False
        )
        print(
            f"VLA format loading time for {num_loaded_vla} trajectories: {vla_loading_time:.2f} seconds"
        )
        print(
            f"VLA format throughput: {num_loaded_vla / vla_loading_time:.2f} trajectories per second\n"
        )

        # Convert VLA to HDF5 and benchmark
        hdf5_handler = HDF5Handler(args.exp_dir, dataset_name)
        hdf5_handler.convert_data_to_hdf5(vla_handler.trajectories_objects)
        hdf5_file_size = hdf5_handler.measure_file_size()
        print(f"Total HDF5 file size: {hdf5_file_size / (1024 * 1024):.2f} MB")

        vla_handler.clear_os_cache()
        # Measure HDF5 loading time
        hdf5_loading_time, num_loaded_hdf5 = hdf5_handler.measure_loading_time()
        print(
            f"HDF5 format loading time for {num_loaded_hdf5} trajectories: {hdf5_loading_time:.2f} seconds"
        )
        print(
            f"HDF5 format throughput: {num_loaded_hdf5 / hdf5_loading_time:.2f} trajectories per second\n"
        )


if __name__ == "__main__":
    # prepare()
    # exit()
    evaluation()
