import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from fog_x.loader import RLDSLoader
import fog_x

def process_data(data_traj, dataset_name, index, destination_dir, lossless):
    data_traj = data_traj[0]
    # try:
    if lossless:
        fog_x.Trajectory.from_list_of_dicts(
            data_traj, path=f"{destination_dir}/{dataset_name}/output_{index}.vla",
            lossy_compression=False
        )
    else:
        fog_x.Trajectory.from_list_of_dicts(
            data_traj, path=f"{destination_dir}/{dataset_name}/output_{index}.vla", 
            lossy_compression=True,
        )
    print(f"Processed data {index}")
    # except Exception as e:
    #     print(f"Failed to process data {index}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Process RLDS data and convert to VLA format.")
    parser.add_argument("--data_dir", required=True, help="Path to the data directory")
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset")
    parser.add_argument("--version", default="0.1.0", help="Dataset version")
    parser.add_argument("--destination_dir", required=True, help="Destination directory for output files")
    parser.add_argument("--split", default="train", help="Data split to use")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of worker processes")
    parser.add_argument("--lossless", action="store_true", help="Enable lossless compression for VLA format")

    args = parser.parse_args()

    loader = RLDSLoader(
        path=f"{args.data_dir}/{args.dataset_name}/{args.version}", split=args.split
    )

    # train[start:end]
    try:
        split_starting_index = int(args.split.split("[")[1].split(":")[0])
        print(f"Starting index: {split_starting_index}")
    except Exception as e:
        print(f"Failed to get starting index: {e}")
        split_starting_index = 0
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        try:
            for index, data_traj in enumerate(loader):
                index = index + split_starting_index
                futures.append(executor.submit(process_data, data_traj, args.dataset_name, index, args.destination_dir, args.lossless))
        except Exception as e:
            print(f"Failed to process data: {e}")

        for future in futures:
            future.result()

    # for index, data_traj in enumerate(loader):
    #     index = index + split_starting_index
    #     process_data(data_traj, args.dataset_name, index, args.destination_dir, args.lossless)

    print("All tasks completed.")

if __name__ == "__main__":
    main()