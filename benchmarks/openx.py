import os
import subprocess
import argparse
import time
import numpy as np
from fog_x.loader import RLDSLoader, VLALoader, HDF5Loader
import tensorflow as tf
import pandas as pd
import fog_x

# Constants
DEFAULT_EXP_DIR = "/mnt/data/fog_x/"
DEFAULT_NUMBER_OF_TRAJECTORIES = -1 # Load all trajectories
DEFAULT_DATASET_NAMES = ["nyu_door_opening_surprising_effectiveness"]
CACHE_DIR = "/mnt/data/fog_x/cache/"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class DatasetHandler:
    def __init__(self, exp_dir, dataset_name, num_trajectories, dataset_type):
        self.exp_dir = exp_dir
        self.dataset_name = dataset_name
        self.num_trajectories = num_trajectories
        self.dataset_type = dataset_type
        self.dataset_dir = os.path.join(exp_dir, dataset_type, dataset_name)

    def measure_folder_size(self):
        """Calculates the total size of all files in the dataset directory."""
        total_size = sum(
            os.path.getsize(os.path.join(dirpath, f))
            for dirpath, dirnames, filenames in os.walk(self.dataset_dir)
            for f in filenames
        )
        return total_size / (1024 * 1024)  # Convert to MB

    def clear_cache(self):
        """Clears the cache directory."""
        if os.path.exists(CACHE_DIR):
            subprocess.run(["rm", "-rf", CACHE_DIR], check=True)

    def clear_os_cache(self):
        """Clears the OS cache."""
        subprocess.run(["sync"], check=True)
        subprocess.run(["echo", "3", ">", "/proc/sys/vm/drop_caches"], check=True)

    def _recursively_load_data(self, data):
        if isinstance(data, dict):
            for key, value in data.items():
                self._recursively_load_data(value)
        elif isinstance(data, (list, tuple)):
            for item in data:
                self._recursively_load_data(item)
        else:
            _ = np.array(data)

class RLDSHandler(DatasetHandler):
    def __init__(self, exp_dir, dataset_name, num_trajectories):
        super().__init__(exp_dir, dataset_name, num_trajectories, dataset_type="rlds")

    def measure_loading_time(self):
        start_time = time.time()
        if self.num_trajectories == -1:
            loader = RLDSLoader(self.dataset_dir, split="train")
        else:
            loader = RLDSLoader(self.dataset_dir, split=f"train[:{self.num_trajectories}]")
        for data in loader:
            self._recursively_load_data(data)
        end_time = time.time()
        return end_time - start_time

class VLAHandler(DatasetHandler):
    def __init__(self, exp_dir, dataset_name, num_trajectories):
        super().__init__(exp_dir, dataset_name, num_trajectories, dataset_type="vla")

    def measure_loading_time(self, mode="no_cache"):
        start_time = time.time()
        loader = VLALoader(self.dataset_dir, cache_dir=CACHE_DIR)
        for data in loader:
            if self.num_trajectories != -1 and loader.index >= self.num_trajectories:
                break
            try:
                self._recursively_load_data(data.load(mode=mode))
            except Exception as e:
                print(f"Failed to load data: {e}")
        end_time = time.time()
        return end_time - start_time

class HDF5Handler(DatasetHandler):
    def __init__(self, exp_dir, dataset_name, num_trajectories):
        super().__init__(exp_dir, dataset_name, num_trajectories, dataset_type="hdf5")

    def measure_loading_time(self):
        start_time = time.time()
        loader = HDF5Loader(path=os.path.join(self.dataset_dir, "*.h5"))
        for data in loader:
            if self.num_trajectories != -1 and loader.index >= self.num_trajectories:
                break
            self._recursively_load_data(data)
        end_time = time.time()
        return end_time - start_time

def prepare(args):
    # Clear the cache directory
    if os.path.exists(CACHE_DIR):
        subprocess.run(["rm", "-rf", CACHE_DIR], check=True)

def evaluation(args):
    results = []
    
    for dataset_name in args.dataset_names:
        print(f"Evaluating dataset: {dataset_name}")

        handlers = [
            RLDSHandler(args.exp_dir, dataset_name, args.num_trajectories),
            VLAHandler(args.exp_dir, dataset_name, args.num_trajectories),
            HDF5Handler(args.exp_dir, dataset_name, args.num_trajectories)
        ]

        for handler in handlers:
            handler.clear_cache()
            handler.clear_os_cache()

            folder_size = handler.measure_folder_size()
            loading_time = handler.measure_loading_time()

            results.append({
                'Dataset': dataset_name,
                'Format': handler.dataset_type.upper(),
                'FolderSize(MB)': folder_size,
                'LoadingTime(s)': loading_time,
            })

            print(f"{handler.dataset_type.upper()} - Folder Size: {folder_size:.2f} MB, Loading Time: {loading_time:.2f} s")

        # Additional VLA measurements
        vla_handler = handlers[1]
        vla_handler.clear_cache()
        vla_handler.clear_os_cache()
        cold_cache_time = vla_handler.measure_loading_time(mode="cache")
        hot_cache_time = vla_handler.measure_loading_time(mode="cache")

        results.append({
            'Dataset': dataset_name,
            'Format': 'VLA-ColdCache',
            'FolderSize(MB)': folder_size,
            'LoadingTime(s)': cold_cache_time,
        })

        results.append({
            'Dataset': dataset_name,
            'Format': 'VLA-HotCache',
            'FolderSize(MB)': folder_size,
            'LoadingTime(s)': hot_cache_time,
        })
        print(f"VLA-ColdCache - Folder Size: {folder_size:.2f} MB, Loading Time: {cold_cache_time:.2f} s")
        print(f"VLA-HotCache - Folder Size: {folder_size:.2f} MB, Loading Time: {hot_cache_time:.2f} s")

    results_df = pd.DataFrame(results)
    results_df.to_csv('format_comparison_results.csv', index=False)
    print("Results written to format_comparison_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and evaluate loading times and folder sizes for RLDS, VLA, and HDF5 formats.")
    parser.add_argument("--exp_dir", type=str, default=DEFAULT_EXP_DIR, help="Experiment directory.")
    parser.add_argument("--num_trajectories", type=int, default=DEFAULT_NUMBER_OF_TRAJECTORIES, help="Number of trajectories to evaluate.")
    parser.add_argument("--dataset_names", nargs="+", default=DEFAULT_DATASET_NAMES, help="List of dataset names to evaluate.")
    parser.add_argument("--prepare", action="store_true", help="Prepare the datasets before evaluation.")
    args = parser.parse_args()

    if args.prepare:
        prepare(args)
    evaluation(args)