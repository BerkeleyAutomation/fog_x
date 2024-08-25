import fog_x
import os
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor
import glob
import time
import numpy as np
from fog_x.loader import RLDSLoader
from fog_x.loader import VLALoader

# Constants
DEFAULT_EXP_DIR = "/tmp/fog_x"
DEFAULT_NUMBER_OF_TRAJECTORIES = 20
DEFAULT_DATASET_NAMES = ["berkeley_autolab_ur5"]
DATA_URL_TEMPLATE = "gs://gresearch/robotics/{dataset_name}/0.1.0/{dataset_name}-train.tfrecord-{index:05d}-*"
LOCAL_FILE_TEMPLATE = (
    "{exp_dir}/{dataset_name}/{dataset_name}-train.tfrecord-{index:05d}-*"
)
FEATURE_JSON_URL_TEMPLATE = "gs://gresearch/robotics/{dataset_name}/0.1.0/features.json"
DATASET_INFO_JSON_URL_TEMPLATE = (
    "gs://gresearch/robotics/{dataset_name}/0.1.0/dataset_info.json"
)


def check_and_download_file(url, local_path):
    """Checks if a file is already downloaded; if not, downloads it."""
    if not os.path.exists(local_path):
        subprocess.run(["gsutil", "-m", "cp", url, local_path], check=True)
    else:
        print(f"File {local_path} already exists. Skipping download.")


def check_and_download_trajectory(exp_dir, dataset_name, trajectory_index):
    """Checks if a trajectory and associated JSON files are already downloaded; if not, downloads them."""
    # Create a directory for each dataset
    dataset_dir = os.path.join(exp_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Check and download the trajectory files
    local_file_pattern = LOCAL_FILE_TEMPLATE.format(
        exp_dir=exp_dir, dataset_name=dataset_name, index=trajectory_index
    )
    if not any(os.path.exists(file) for file in glob.glob(local_file_pattern)):
        data_url = DATA_URL_TEMPLATE.format(
            dataset_name=dataset_name, index=trajectory_index
        )
        subprocess.run(["gsutil", "-m", "cp", data_url, dataset_dir], check=True)
    else:
        print(
            f"Trajectory {trajectory_index} of dataset {dataset_name} already exists in {dataset_dir}. Skipping download."
        )

    # Check and download the feature.json file
    feature_json_local_path = os.path.join(dataset_dir, "features.json")
    feature_json_url = FEATURE_JSON_URL_TEMPLATE.format(dataset_name=dataset_name)
    check_and_download_file(feature_json_url, feature_json_local_path)

    # Check and download the dataset_info.json file
    dataset_info_json_local_path = os.path.join(dataset_dir, "dataset_info.json")
    dataset_info_json_url = DATASET_INFO_JSON_URL_TEMPLATE.format(
        dataset_name=dataset_name
    )
    check_and_download_file(dataset_info_json_url, dataset_info_json_local_path)


def download_data(exp_dir, dataset_names, num_trajectories):
    """Downloads the specified number of trajectories from each dataset concurrently if not already downloaded."""
    with ThreadPoolExecutor() as executor:
        futures = []
        for dataset_name in dataset_names:
            for i in range(num_trajectories):
                futures.append(
                    executor.submit(
                        check_and_download_trajectory, exp_dir, dataset_name, i
                    )
                )
        for future in futures:
            future.result()  # Will raise an exception if any download failed


def measure_file_size(dataset_dir):
    """Calculates the total size of all files in the dataset directory."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(dataset_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def measure_loading_time(loader_func, path, num_trajectories):
    """Measures the time taken to load data into memory using a specified loader function."""
    start_time = time.time()
    loader = loader_func(path, split=f"train[:{num_trajectories}]")
    for data in loader:
        #  use np array to force loading
        data

    end_time = time.time()
    loading_time = end_time - start_time
    print(
        f"Loaded {len(loader)} trajectories in {loading_time:.2f} seconds start time {start_time} end time {end_time}"
    )
    return loading_time, num_trajectories


def convert_data_to_vla_format(loader, output_dir):
    """Converts data to VLA format and saves it to the specified output directory."""
    for index, data_traj in enumerate(loader):
        output_path = os.path.join(output_dir, f"output_{index}.vla")
        print(
            f"Converting trajectory {index} to VLA format and saving to {output_path} {len(data_traj)}"
        )
        fog_x.Trajectory.from_list_of_dicts(data_traj, path=output_path)


def read_data(output_dir, num_trajectories):
    """Reads the VLA data files and prints their action keys."""
    for i in range(num_trajectories):
        traj = fog_x.Trajectory(os.path.join(output_dir, f"output_{i}.vla"))
        print(traj["action"].keys())


def main():
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

    # Create output directory if it doesn't exist
    output_dir = os.path.join(args.exp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Download data concurrently
    download_data(args.exp_dir, args.dataset_names, args.num_trajectories)

    # Iterate through datasets and measure file size and loading time for both formats
    for dataset_name in args.dataset_names:
        dataset_dir = os.path.join(args.exp_dir, dataset_name)
        file_size = measure_file_size(dataset_dir)

        # Measure loading time for RLDS format
        rlds_loading_time, num_loaded_rlds = measure_loading_time(
            RLDSLoader, dataset_dir, args.num_trajectories
        )

        print(f"Dataset: {dataset_name}")
        print(f"Total file size: {file_size / (1024 * 1024):.2f} MB")
        print(
            f"RLDS format loading time for {num_loaded_rlds} trajectories: {rlds_loading_time:.2f} seconds"
        )
        print(
            f"RLDS format throughput: {num_loaded_rlds / rlds_loading_time:.2f} trajectories per second"
        )

        # Convert data to VLA format
        loader = RLDSLoader(path=dataset_dir, split=f"train[:{args.num_trajectories}]")
        convert_data_to_vla_format(loader, output_dir)

        # Measure loading time for VLA format
        vla_loading_time, num_loaded_vla = measure_loading_time(
            VLALoader, output_dir, args.num_trajectories
        )

        print(
            f"VLA format loading time for {num_loaded_vla} trajectories: {vla_loading_time:.2f} seconds"
        )
        print(
            f"VLA format throughput: {num_loaded_vla / vla_loading_time:.2f} trajectories per second\n"
        )


if __name__ == "__main__":
    main()
