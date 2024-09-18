import fog_x 
import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from tqdm import tqdm
import threading
from fog_x.loader import NonShuffleVLALoader
import h5py
import time

def process_data(trajectory, dataset_name, index, destination_dir):
    try:    
        print(f"Processing data {index}")
        if trajectory is None:
            print(f"Trajectory is None for index {index}")
            return index, False
        write_to_h5(trajectory, dataset_name, index, destination_dir)
        return index, True
    except Exception as e:
        print(f"Failed to process data {index}: {e}")
        return index, False

def write_to_h5(trajectory, dataset_name, index, destination_dir):
    print(trajectory.keys())
    try:
        with h5py.File(f"{destination_dir}/{dataset_name}/output_{index}.h5", "w") as f:
            for k in trajectory.keys():
                v = trajectory[k]
                print(k, v.shape)
                
                f.create_dataset(k, data=v, compression="gzip", compression_opts=9)
    except Exception as e:
        print(f"Failed to write to h5 {index}: {e}")

    # except Exception as e:
    #     print(f"Failed to process data {index}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert VLA data to HDF5 format.")
    parser.add_argument("--data_dir", required=True, help="Path to the VLA data directory")
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset")
    parser.add_argument("--destination_dir", required=True, help="Destination directory for output HDF5 files")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of worker processes")
    parser.add_argument("--timeout", type=int, default=20, help="Timeout for each task in seconds")

    args = parser.parse_args()

    vla_path = os.path.join(args.data_dir, args.dataset_name, "*.vla")
    cache_dir = os.path.join("/mnt/data/fog_x/cache/", args.dataset_name)
    print(vla_path, cache_dir)
    loader = NonShuffleVLALoader(vla_path, cache_dir=cache_dir)

    os.makedirs(os.path.join(args.destination_dir, args.dataset_name), exist_ok=True)

    max_concurrent_tasks = args.max_workers
    semaphore = threading.Semaphore(max_concurrent_tasks)

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        retry_queue = []
        try:
            for index, trajectory in tqdm(enumerate(loader), desc="Submitting tasks", unit="trajectory"):
                semaphore.acquire()
                future = executor.submit(process_data, trajectory, args.dataset_name, index, args.destination_dir)
                future.add_done_callback(lambda x: semaphore.release())
                futures.append(future)
        except Exception as e:
            print(f"Failed to submit tasks: {e}")

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tasks"):
            try:
                index, success = future.result(timeout=args.timeout)
                if not success:
                    retry_queue.append((index, trajectory))
            except TimeoutError:
                print(f"Task for index {index} timed out")
                retry_queue.append((index, trajectory))
            except Exception as e:
                print(f"Error processing future: {e}")

    # Retry failed tasks
    if retry_queue:
        print(f"Retrying {len(retry_queue)} failed tasks...")
        with ProcessPoolExecutor(max_workers=args.max_workers) as retry_executor:
            retry_futures = []
            for index, trajectory in retry_queue:
                future = retry_executor.submit(process_data, trajectory, args.dataset_name, index, args.destination_dir)
                retry_futures.append(future)
            
            for future in tqdm(as_completed(retry_futures), total=len(retry_futures), desc="Processing retry tasks"):
                try:
                    index, success = future.result(timeout=args.timeout)
                    if not success:
                        print(f"Failed to process data {index} after retry")
                except TimeoutError:
                    print(f"Retry task for index {index} timed out")
                except Exception as e:
                    print(f"Error processing retry future: {e}")

    print("All tasks completed.")

if __name__ == "__main__":
    main()
