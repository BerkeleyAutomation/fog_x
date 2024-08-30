import fog_x 
import os
import argparse
from concurrent.futures import ProcessPoolExecutor
from fog_x.loader import VLALoader

def process_data(trajectory, dataset_name, index, destination_dir):
    try:
        trajectory.to_hdf5(path=f"{destination_dir}/{dataset_name}/output_{index}.h5")
        print(f"processed data {index} to {destination_dir}/{dataset_name}/output_{index}.h5")
    except Exception as e:
        print(f"Failed to process data {index}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert VLA data to HDF5 format.")
    parser.add_argument("--data_dir", required=True, help="Path to the VLA data directory")
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset")
    parser.add_argument("--destination_dir", required=True, help="Destination directory for output HDF5 files")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of worker processes")

    args = parser.parse_args()

    vla_path = os.path.join(args.data_dir, args.dataset_name, "*.vla")
    cache_dir = os.path.join("/mnt/data/fog_x/cache/", args.dataset_name)
    loader = VLALoader(vla_path, cache_dir=cache_dir)

    os.makedirs(os.path.join(args.destination_dir, args.dataset_name), exist_ok=True)

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        try:
            for index, trajectory in enumerate(loader):
                futures.append(executor.submit(process_data, trajectory, args.dataset_name, index, args.destination_dir))
        except Exception as e:
            print(f"Failed to process data: {e}")

        for future in futures:
            future.result()

    print("All tasks completed.")

if __name__ == "__main__":
    main()
