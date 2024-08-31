import os
import subprocess
import argparse
import time
import numpy as np
from fog_x.loader import RLDSLoader, VLALoader, HDF5Loader
import tensorflow as tf
import pandas as pd
import fog_x
import csv
import stat
from fog_x.loader.lerobot import LeRobotLoader
from fog_x.loader.pytorch_vla import get_vla_dataloader

# Constants
DEFAULT_EXP_DIR = "/mnt/data/fog_x/"
DEFAULT_NUMBER_OF_TRAJECTORIES = -1 # Load all trajectories
# DEFAULT_DATASET_NAMES = ["nyu_door_opening_surprising_effectiveness", "berkeley_cable_routing", "berkeley_autolab_ur5", "bridge"]
DEFAULT_DATASET_NAMES = ["berkeley_cable_routing"]
CACHE_DIR = "/tmp/fog_x/cache/"
DEFAULT_LOG_FREQUENCY = 20

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class DatasetHandler:
    def __init__(self, exp_dir, dataset_name, num_trajectories, dataset_type, batch_size, log_frequency=DEFAULT_LOG_FREQUENCY):
        self.exp_dir = exp_dir
        self.dataset_name = dataset_name
        self.num_trajectories = num_trajectories
        self.dataset_type = dataset_type
        self.dataset_dir = os.path.join(exp_dir, dataset_type, dataset_name)
        self.batch_size = batch_size
        # Resolve the symbolic link if the dataset_dir is a soft link
        self.dataset_dir = os.path.realpath(self.dataset_dir)
        self.log_frequency = log_frequency
        self.results = []

    def measure_average_trajectory_size(self):
        """Calculates the average size of trajectory files in the dataset directory."""
        total_size = 0
        file_count = 0
        for dirpath, dirnames, filenames in os.walk(self.dataset_dir):
            for f in filenames:
                if f.endswith(self.file_extension):
                    file_path = os.path.join(dirpath, f)
                    total_size += os.path.getsize(file_path)
                    file_count += 1
        if file_count == 0:
            return 0
        return (total_size / file_count) / (1024 * 1024)  # Convert to MB

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

    def write_result(self, format_name, elapsed_time, index):
        result = {
            'Dataset': self.dataset_name,
            'Format': format_name,
            'AverageTrajectorySize(MB)': self.measure_average_trajectory_size(),
            'LoadingTime(s)': elapsed_time,
            'Index': index,
            'BatchSize': self.batch_size
        }
        
        csv_file = f'{self.dataset_name}_results.csv'
        file_exists = os.path.isfile(csv_file)
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(result)

class RLDSHandler(DatasetHandler):
    def __init__(self, exp_dir, dataset_name, num_trajectories, batch_size, log_frequency=DEFAULT_LOG_FREQUENCY):
        super().__init__(exp_dir, dataset_name, num_trajectories, dataset_type="rlds", batch_size=batch_size, log_frequency=log_frequency)
        self.file_extension = ".tfrecord"

    def measure_random_loading_time(self, num_loads):
        start_time = time.time()
        loader = RLDSLoader(self.dataset_dir, split="train", batch_size=self.batch_size)
        
        for i in range(num_loads):
            for batch_num, data in enumerate(loader):
                self._recursively_load_data(data)
                
                elapsed_time = time.time() - start_time
                self.write_result(f"RLDS-RandomLoad", elapsed_time, batch_num)
                if batch_num % self.log_frequency == 0:
                    print(f"RLDS-RandomLoad - Loaded {batch_num} random batches, Time: {elapsed_time:.2f} s")
            
        return time.time() - start_time

class VLAHandler(DatasetHandler):
    def __init__(self, exp_dir, dataset_name, num_trajectories, batch_size, log_frequency=DEFAULT_LOG_FREQUENCY):   
        super().__init__(exp_dir, dataset_name, num_trajectories, dataset_type="vla", batch_size=batch_size, log_frequency=log_frequency)
        self.file_extension = ".vla"

    def measure_random_loading_time(self, num_loads, save_to_cache=True):
        start_time = time.time()
        dataloader = get_vla_dataloader(self.dataset_dir, batch_size=self.batch_size, cache_dir=CACHE_DIR)
        
        for i in range(num_loads):
            for batch_num, batch in enumerate(dataloader):
                self._recursively_load_data(batch)
                elapsed_time = time.time() - start_time 
                self.write_result(f"VLA-RandomLoad", elapsed_time, batch_num)
                if batch_num % self.log_frequency == 0:
                    print(f"VLA-RandomLoad - Loaded {batch_num} random batches, Time: {elapsed_time:.2f} s")

        return time.time() - start_time

class HDF5Handler(DatasetHandler):
    def __init__(self, exp_dir, dataset_name, num_trajectories, batch_size, log_frequency=DEFAULT_LOG_FREQUENCY):
        super().__init__(exp_dir, dataset_name, num_trajectories, dataset_type="hdf5", batch_size=batch_size, log_frequency=log_frequency)
        self.file_extension = ".h5"

    def measure_random_loading_time(self, num_loads):
        start_time = time.time()
        loader = HDF5Loader(path=os.path.join(self.dataset_dir, "*.h5"))
        
        for i in range(num_loads):
            for batch_num, data in enumerate(loader):
                self._recursively_load_data(data)
                elapsed_time = time.time() - start_time
                self.write_result(f"HDF5-RandomLoad", elapsed_time, batch_num)
                if batch_num % self.log_frequency == 0:
                    print(f"HDF5-RandomLoad - Loaded {batch_num} random batches, Time: {elapsed_time:.2f} s")
            
        return time.time() - start_time

class LeRobotHandler(DatasetHandler):
    def __init__(self, exp_dir, dataset_name, num_trajectories, batch_size, log_frequency=DEFAULT_LOG_FREQUENCY):
        super().__init__(exp_dir, dataset_name, num_trajectories, dataset_type="hf", batch_size=batch_size, log_frequency=log_frequency)
        self.file_extension = ""  # LeRobot datasets don't have a specific file extension

    def measure_random_loading_time(self, num_loads):
        start_time = time.time()
        path = os.path.join(self.exp_dir, "hf")
        loader = LeRobotLoader(path, self.dataset_name, batch_size=self.batch_size)
        
        dataset_len = len(loader)
        
        for i in range(num_loads):
            for batch_num, data in enumerate(loader):
                if batch_num >= dataset_len:
                    break
                self._recursively_load_data(data)
                elapsed_time = time.time() - start_time
                self.write_result(f"LeRobot-RandomLoad", elapsed_time, batch_num)
                if batch_num % self.log_frequency == 0:
                    print(f"LeRobot-RandomLoad - Loaded {batch_num} random batches, Time: {elapsed_time:.2f} s")
            
        return time.time() - start_time

def prepare(args):
    # Clear the cache directory
    if os.path.exists(CACHE_DIR):
        subprocess.run(["rm", "-rf", CACHE_DIR], check=True)

def evaluation(args):
    
    csv_file = 'format_comparison_results.csv'
    
    if os.path.exists(csv_file):
        existing_results = pd.read_csv(csv_file).to_dict('records')
    else:
        existing_results = []
    
    new_results = []
    for dataset_name in args.dataset_names:
        print(f"Evaluating dataset: {dataset_name}")

        handlers = [
            # RLDSHandler(args.exp_dir, dataset_name, args.num_trajectories, args.batch_size, args.log_frequency),
            VLAHandler(args.exp_dir, dataset_name, args.num_trajectories, args.batch_size, args.log_frequency),
            # HDF5Handler(args.exp_dir, dataset_name, args.num_trajectories, args.batch_size, args.log_frequency),
            # FFV1Handler(args.exp_dir, dataset_name, args.num_trajectories, args.log_frequency, args.batch_size)
            # LeRobotHandler(args.exp_dir, dataset_name, args.num_trajectories, args.batch_size, args.log_frequency),
        ]

        for handler in handlers:
            handler.clear_cache()
            handler.clear_os_cache()

            avg_traj_size = handler.measure_average_trajectory_size()
            # loading_time = handler.measure_loading_time()

            # new_results.append({
            #     'Dataset': dataset_name,
            #     'Format': handler.dataset_type.upper(),
            #     'AverageTrajectorySize(MB)': avg_traj_size,
            #     'LoadingTime(s)': loading_time,
            # })

            # print(f"{handler.dataset_type.upper()} - Average Trajectory Size: {avg_traj_size:.2f} MB, Loading Time: {loading_time:.2f} s")

            random_load_time = handler.measure_random_loading_time(args.random_loads)
            new_results.append({
                'Dataset': dataset_name,
                'Format': f"{handler.dataset_type.upper()}-RandomLoad",
                'AverageTrajectorySize(MB)': avg_traj_size,
                'LoadingTime(s)': random_load_time,
            })
            print(f"{handler.dataset_type.upper()}-RandomLoad - Average Trajectory Size: {avg_traj_size:.2f} MB, Loading Time: {random_load_time:.2f} s")

        # # Additional VLA measurements
        # vla_handler = handlers[1]
        # vla_handler.clear_cache()
        # vla_handler.clear_os_cache()
        # cold_cache_time = vla_handler.measure_loading_time(mode="cache")
        # hot_cache_time = vla_handler.measure_loading_time(mode="cache")

        # new_results.append({
        #     'Dataset': dataset_name,
        #     'Format': 'VLA-ColdCache',
        #     'AverageTrajectorySize(MB)': avg_traj_size,
        #     'LoadingTime(s)': cold_cache_time,
        # })

        # new_results.append({
        #     'Dataset': dataset_name,
        #     'Format': 'VLA-HotCache',
        #     'AverageTrajectorySize(MB)': avg_traj_size,
        #     'LoadingTime(s)': hot_cache_time,
        # })
        # print(f"VLA-ColdCache - Average Trajectory Size: {avg_traj_size:.2f} MB, Loading Time: {cold_cache_time:.2f} s")
        # print(f"VLA-HotCache - Average Trajectory Size: {avg_traj_size:.2f} MB, Loading Time: {hot_cache_time:.2f} s")

        # Combine existing and new results
        all_results = existing_results + new_results

        # Write all results to CSV
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(csv_file, index=False)
        print(f"Results appended to {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and evaluate loading times and folder sizes for RLDS, VLA, and HDF5 formats.")
    parser.add_argument("--exp_dir", type=str, default=DEFAULT_EXP_DIR, help="Experiment directory.")
    parser.add_argument("--num_trajectories", type=int, default=DEFAULT_NUMBER_OF_TRAJECTORIES, help="Number of trajectories to evaluate.")
    parser.add_argument("--dataset_names", nargs="+", default=DEFAULT_DATASET_NAMES, help="List of dataset names to evaluate.")
    parser.add_argument("--prepare", action="store_true", help="Prepare the datasets before evaluation.")
    parser.add_argument("--log_frequency", type=int, default=DEFAULT_LOG_FREQUENCY, help="Frequency of logging results.")
    parser.add_argument("--random_loads", type=int, default=2, help="Number of random loads to perform for each loader.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for loaders.")
    args = parser.parse_args()

    if args.prepare:
        prepare(args)
    evaluation(args)