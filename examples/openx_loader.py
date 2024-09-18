import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from fog_x.loader import RLDSLoader
import fog_x
import threading
import time

def process_data(data_traj, dataset_name, index, destination_dir, lossless):
    try:
        data_traj = data_traj[0]
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
        return index, True
    except Exception as e:
        print(f"Failed to process data {index}: {e}")
        return index, False

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
        path=f"{args.data_dir}/{args.dataset_name}/{args.version}", split=args.split, shuffling = False
    )

    # train[start:end]
    try:
        split_starting_index = int(args.split.split("[")[1].split(":")[0])
        print(f"Starting index: {split_starting_index}")
    except Exception as e:
        print(f"Failed to get starting index: {e}")
        split_starting_index = 0
    
    max_concurrent_tasks = args.max_workers
    semaphore = threading.Semaphore(max_concurrent_tasks)

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        retry_queue = []
        try:
            from tqdm import tqdm
            for index, data_traj in tqdm(enumerate(loader), desc="Processing data", unit="trajectory"):
                if index < split_starting_index:
                    continue
                semaphore.acquire()
                future = executor.submit(process_data, data_traj, args.dataset_name, index, args.destination_dir, args.lossless)
                future.add_done_callback(lambda x: semaphore.release())
                futures.append(future)
        except Exception as e:
            print(f"Failed to process data: {e}")

        for future in as_completed(futures):
            try:
                index, success = future.result()
                if not success:
                    retry_queue.append((index, data_traj))
            except Exception as e:
                print(f"Error processing future: {e}")

    # Retry failed tasks
    if retry_queue:
        print(f"Retrying {len(retry_queue)} failed tasks...")
        with ProcessPoolExecutor(max_workers=args.max_workers) as retry_executor:
            retry_futures = []
            for index, data_traj in retry_queue:
                future = retry_executor.submit(process_data, data_traj, args.dataset_name, index, args.destination_dir, args.lossless)
                retry_futures.append(future)
            
            for future in as_completed(retry_futures):
                try:
                    index, success = future.result()
                    if not success:
                        print(f"Failed to process data {index} after retry")
                except Exception as e:
                    print(f"Error processing retry future: {e}")

    print("All tasks completed.")

if __name__ == "__main__":
    main()