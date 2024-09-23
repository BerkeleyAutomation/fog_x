import os
import subprocess
import argparse
import time
import numpy as np
from fog_x.loader import RLDSLoader, VLALoader, HDF5FrameLoader, HDF5EpisodeLoader
import tensorflow as tf
import pandas as pd
import fog_x
import csv
import stat
from fog_x.loader.lerobot import LeRobotLoader_ByFrame
from fog_x.loader.vla import get_vla_dataloader
from fog_x.loader.hdf5 import get_hdf5_dataloader

# Constants
DEFAULT_EXP_DIR = "/mnt/data/fog_x/"
DEFAULT_NUMBER_OF_TRAJECTORIES = -1  # Load all trajectories
DEFAULT_DATASET_NAMES = [
    "nyu_door_opening_surprising_effectiveness",
    "berkeley_cable_routing",
    "berkeley_autolab_ur5",
    "bridge",
]
# DEFAULT_DATASET_NAMES = ["bridge"]
# CACHE_DIR = "/tmp/fog_x/cache/"
CACHE_DIR  = "/mnt/data/fog_x/cache/"
DEFAULT_LOG_FREQUENCY = 20

# suppress tensorflow warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
logger = logging.getLogger(__name__)

class DatasetHandler:
    def __init__(
        self,
        exp_dir,
        dataset_name,
        num_batches,
        dataset_type,
        batch_size,
        log_frequency=DEFAULT_LOG_FREQUENCY,
    ):
        self.exp_dir = exp_dir
        self.dataset_name = dataset_name
        self.num_batches = num_batches
        self.dataset_type = dataset_type
        self.dataset_dir = os.path.join(exp_dir, dataset_type, dataset_name)
        self.batch_size = batch_size
        # Resolve the symbolic link if the dataset_dir is a soft link
        self.dataset_dir = os.path.realpath(self.dataset_dir)
        self.log_frequency = log_frequency
        self.results = []
        self.log_level = "debug"
        self.unit = "frame"

    def measure_average_trajectory_size(self):
        """Calculates the average size of trajectory files in the dataset directory."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.dataset_dir):
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                total_size += os.path.getsize(file_path)
        
        logger.debug(f"total_size: {total_size} of directory {self.dataset_dir}")
        # trajectory number 
        traj_num = 0
        if self.dataset_name == "nyu_door_opening_surprising_effectiveness":
            traj_num = 435
        if self.dataset_name == "berkeley_cable_routing":
            traj_num = 1482
        if self.dataset_name == "bridge":
            traj_num = 25460
        if self.dataset_name == "berkeley_autolab_ur5":
            traj_num = 896
        return (total_size / traj_num) / (1024 * 1024)  # Convert to MB

    def clear_cache(self):
        """Clears the cache directory."""
        if os.path.exists(CACHE_DIR):
            logger.info(f"Clearing cache directory: {CACHE_DIR}")
            subprocess.run(["rm", "-rf", CACHE_DIR], check=True)

    def clear_os_cache(self):
        """Clears the OS cache."""
        subprocess.run(["sync"], check=True)
        subprocess.run(["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=True)
        logger.info(f"Cleared OS cache")
        
    def _recursively_load_data(self, data):
        logger.debug(f"Data summary for loader {self.dataset_type.upper()}")
        if None in data:
            logger.warning(f"None value found in data")
        def summarize_trajectory(trajectory):
            def summarize_value(value):
                if isinstance(value, np.ndarray):
                    return value.shape
                elif isinstance(value, (list, tuple)):
                    if len(value) > 0 and isinstance(value[0], np.ndarray):
                        return [v.shape for v in value]
                    return len(value)
                elif isinstance(value, dict):
                    return {k: summarize_value(v) for k, v in value.items()}
                elif isinstance(value, str):
                    return value
                else:
                    logger.warning(f"Unknown type: {type(value)}")
                    return type(value).__name__

            return {key: summarize_value(value) for key, value in trajectory.items()}

        trajectory_summaries = [summarize_trajectory(trajectory) for trajectory in data]

        log_func = logger.debug if self.log_level == 'debug' else logger.info
        for i, summary in enumerate(trajectory_summaries):
            log_func(f"Trajectory {i + 1}:")
            for feature, dimension in summary.items():
                if isinstance(dimension, dict):
                    log_func(f"  {feature}:")
                    for sub_feature, sub_dimension in dimension.items():
                        log_func(f"    {sub_feature}: {sub_dimension}")
                else:
                    log_func(f"  {feature}: {dimension}")

        log_func(f"Total number of trajectories: {len(trajectory_summaries)}")

    def write_result(self, format_name, elapsed_time, index):
        result = {
            "Dataset": self.dataset_name,
            "Format": format_name,
            "AverageTrajectorySize(MB)": self.measure_average_trajectory_size(),
            "LoadingTime(s)": elapsed_time,
            "AverageLoadingTime(s)": elapsed_time / (index + 1),
            "Index": index,
            "BatchSize": self.batch_size,
        }

        csv_file = f"{self.dataset_name}_results.csv"
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(result)

    def measure_random_loading_time(self):
        start_time = time.time()
        loader = self.get_loader()
        last_batch_time = time.time()
        for batch_num, data in enumerate(loader):
            if batch_num >= self.num_batches:
                break
            self._recursively_load_data(data)
            current_batch_time = time.time()
            elapsed_time = current_batch_time - last_batch_time
            last_batch_time = current_batch_time

            self.write_result(
                f"{self.dataset_type.upper()}", elapsed_time, batch_num
            )
            if batch_num % self.log_frequency == 0:
                logger.info(
                    f"{self.dataset_type.upper()} - Loaded {batch_num} random {self.batch_size} batches from {self.dataset_name}, Time: {elapsed_time:.2f} s, Total Average Time: {(current_batch_time - start_time) / (batch_num + 1):.2f} s, Batch Average Time: {elapsed_time / self.batch_size:.2f} s"
                )

        return time.time() - start_time

    def get_loader(self):
        raise NotImplementedError("Subclasses must implement get_loader method")


class RLDSHandler(DatasetHandler):
    def __init__(
        self,
        exp_dir,
        dataset_name,
        num_batches,
        batch_size,
        log_frequency=DEFAULT_LOG_FREQUENCY,
    ):
        super().__init__(
            exp_dir,
            dataset_name,
            num_batches,
            dataset_type="rlds",
            batch_size=batch_size,
            log_frequency=log_frequency,
        )
        self.file_extension = ".tfrecord"

    def get_loader(self):
        return RLDSLoader(self.dataset_dir, split="train", batch_size=self.batch_size)

    def _recursively_load_data(self, data):
        log_level = self.log_level
        # rlds returns a list of dictionaries
        log_func = logger.debug if log_level == 'debug' else logger.info
        log_func(f"Data summary for loader {self.dataset_type.upper()}")
        for i, trajectory in enumerate(data):
            log_func(f"Trajectory {i + 1}:")
            # each trajectory is a list of dictionaries
            for j, step in enumerate(trajectory):
                log_func(f"  Step {j + 1}:")
                for key, value in step.items():
                    if isinstance(value, np.ndarray):
                        log_func(f"    {key}: {value.shape}")
                    elif isinstance(value, dict):
                        log_func(f"    {key}:")
                        for sub_key, sub_value in value.items():
                            log_func(f"      {sub_key}: {sub_value.shape}")
                    else:
                        log_func(f"    {key}: {type(value).__name__}")
        log_func(f"Total number of trajectories: {len(data)}")

class VLAHandler(DatasetHandler):
    def __init__(
        self,
        exp_dir,
        dataset_name,
        num_batches,
        batch_size,
        log_frequency=DEFAULT_LOG_FREQUENCY,
    ):
        super().__init__(
            exp_dir,
            dataset_name,
            num_batches,
            dataset_type="vla",
            batch_size=batch_size,
            log_frequency=log_frequency,
        )
        self.file_extension = ".vla"

    def get_loader(self):
        if self.unit == "frame":
            return get_vla_dataloader(
                self.dataset_dir, batch_size=1, cache_dir=CACHE_DIR, 
                unit = self.unit,
                slice_size=self.batch_size,
            )
        else:
            return get_vla_dataloader(
                self.dataset_dir, batch_size=self.batch_size, cache_dir=CACHE_DIR, 
                unit = self.unit,
            )


class HDF5Handler(DatasetHandler):
    def __init__(
        self,
        exp_dir,
        dataset_name,
        num_batches,
        batch_size,
        log_frequency=DEFAULT_LOG_FREQUENCY,
    ):
        super().__init__(
            exp_dir,
            dataset_name,
            num_batches,
            dataset_type="hdf5",
            batch_size=batch_size,
            log_frequency=log_frequency,
        )
        self.file_extension = ".h5"

    def get_loader(self):
        if self.unit == "frame":
            return get_hdf5_dataloader(
                path=os.path.join(self.dataset_dir, "*.h5"),
                batch_size=1,
                num_workers=0,  # You can adjust this if needed
                unit = self.unit,
                slice_size=self.batch_size,
            )
        else:
            return get_hdf5_dataloader(
                path=os.path.join(self.dataset_dir, "*.h5"),
                batch_size=self.batch_size,
                num_workers=0,  # You can adjust this if needed
                unit = self.unit,
            )


class LeRobotHandler(DatasetHandler):
    def __init__(
        self,
        exp_dir,
        dataset_name,
        num_batches,
        batch_size,
        log_frequency=DEFAULT_LOG_FREQUENCY,
    ):
        super().__init__(
            exp_dir,
            dataset_name,
            num_batches,
            dataset_type="hf",
            batch_size=batch_size,
            log_frequency=log_frequency,
        )
        self.file_extension = (
            ""  # LeRobot datasets don't have a specific file extension
        )

    def get_loader(self):
        path = os.path.join(self.exp_dir, "hf")
        return LeRobotLoader_ByFrame(path, self.dataset_name, batch_size=1, slice_length=self.batch_size)

    def _recursively_load_data(self, data):
        import torch
        log_level = self.log_level
        # LeRobot returns a list of lists
        log_func = logger.debug if log_level == 'debug' else logger.info
        log_func(f"Data summary for loader {self.dataset_type.upper()}")
        for i, trajectory in enumerate(data):
            log_func(f"Trajectory {i + 1}:")
            # each trajectory is a list of dictionaries
            for j, step in enumerate(trajectory):
                log_func(f"  Step {j + 1}:")
                for key, value in step.items():
                    if isinstance(value, np.ndarray):
                        log_func(f"    {key}: {value.shape}")
                    elif isinstance(value, dict):
                        log_func(f"    {key}:")
                        for sub_key, sub_value in value.items():
                            log_func(f"      {sub_key}: {sub_value.shape}")
                    elif isinstance(value, torch.Tensor):
                        log_func(f"    {key}: {value.shape}")
                    else:
                        log_func(f"    {key}: {type(value).__name__}")
        log_func(f"Total number of trajectories: {len(data)}")

class FFV1Handler(DatasetHandler):
    def __init__(self, exp_dir, dataset_name, num_batches, batch_size, log_frequency=DEFAULT_LOG_FREQUENCY):
        super().__init__(exp_dir, dataset_name, num_batches, dataset_type="ffv1", batch_size=batch_size, log_frequency=log_frequency)
        self.file_extension = ".vla"

    def get_loader(self):
        return VLALoader(self.dataset_dir, batch_size=self.batch_size)


def evaluation(args):

    csv_file = "format_comparison_results.csv"

    if os.path.exists(csv_file):
        existing_results = pd.read_csv(csv_file).to_dict("records")
    else:
        existing_results = []

    new_results = []
    for dataset_name in args.dataset_names:
        logger.debug(f"Evaluating dataset: {dataset_name}")
        
        handlers = [
            VLAHandler(
                args.exp_dir,
                dataset_name,
                args.num_batches,
                args.batch_size,
                args.log_frequency,
            ),
            HDF5Handler(
                args.exp_dir,
                dataset_name,
                args.num_batches,
                args.batch_size,
                args.log_frequency,
            ),
            LeRobotHandler(
                args.exp_dir,
                dataset_name,
                args.num_batches,
                args.batch_size,
                args.log_frequency,
            ),
            # RLDSHandler(
            #     args.exp_dir,
            #     dataset_name,
            #     args.num_batches,
            #     args.batch_size,
            #     args.log_frequency,
            # ),
            # FFV1Handler(
            #     args.exp_dir,
            #     dataset_name,
            #     args.num_batches,
            #     args.batch_size,
            #     args.log_frequency,
            # ),
        ]

        for handler in handlers:
            handler.clear_cache()
            handler.clear_os_cache()

            avg_traj_size = handler.measure_average_trajectory_size()
            random_load_time = handler.measure_random_loading_time()
            new_results.append(
                {
                    "Dataset": dataset_name,
                    "Format": f"{handler.dataset_type.upper()}",
                    "AverageTrajectorySize(MB)": avg_traj_size,
                    "LoadingTime(s)": random_load_time,
                    "AverageLoadingTime(s)": random_load_time / (args.num_batches + 1),
                    "Index": args.num_batches,
                    "BatchSize": args.batch_size,
                }
            )
            logger.debug(
                f"{handler.dataset_type.upper()} - Average Trajectory Size: {avg_traj_size:.2f} MB, Loading Time: {random_load_time:.2f} s"
            )

        # Combine existing and new results
        all_results = existing_results + new_results

        # Write all results to CSV
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(csv_file, index=False)
        logger.debug(f"Results appended to {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare and evaluate loading times and folder sizes for RLDS, VLA, and HDF5 formats."
    )
    parser.add_argument(
        "--exp_dir", type=str, default=DEFAULT_EXP_DIR, help="Experiment directory."
    )
    parser.add_argument(
        "--dataset_names",
        nargs="+",
        default=DEFAULT_DATASET_NAMES,
        help="List of dataset names to evaluate.",
    )

    parser.add_argument(
        "--log_frequency",
        type=int,
        default=DEFAULT_LOG_FREQUENCY,
        help="Frequency of logging results.",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=1000,
        help="Number of batches to load for each loader.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for loaders."
    )
    args = parser.parse_args()

    evaluation(args)
