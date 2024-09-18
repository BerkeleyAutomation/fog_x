import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from fog_x.loader import RLDSLoader
import fog_x
import time
def check_and_fix_conversion(file_path, data_traj, dataset_name, index, destination_dir, lossless):
    try:
        # Try to load the existing file
        fog_x.Trajectory(file_path).load()
        print(f"File {file_path} is valid.")
        return index, True
    except Exception as e:
        print(f"Failed to load {file_path}. Attempting to fix: {e}")
        
        # If loading fails, attempt to reconvert
        try:
            data_traj = data_traj[0]
            if lossless:
                fog_x.Trajectory.from_list_of_dicts(
                    data_traj, path=file_path,
                    lossy_compression=False
                )
            else:
                fog_x.Trajectory.from_list_of_dicts(
                    data_traj, path=file_path, 
                    lossy_compression=True,
                )
            print(f"Successfully fixed and reconverted data {index}")
            return index, True
        except Exception as e:
            print(f"Failed to fix data {index}: {e}")
            return index, False

def main():
    parser = argparse.ArgumentParser(description="Check and fix failed VLA conversions.")
    parser.add_argument("--data_dir", required=True, help="Path to the original data directory")
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset")
    parser.add_argument("--version", default="0.1.0", help="Dataset version")
    parser.add_argument("--destination_dir", required=True, help="Directory containing converted files")
    parser.add_argument("--split", default="train", help="Data split to use")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of worker processes")
    parser.add_argument("--lossless", action="store_true", help="Enable lossless compression for VLA format")

    args = parser.parse_args()

    loader = RLDSLoader(
        path=f"{args.data_dir}/{args.dataset_name}/{args.version}", split=args.split, shuffling=False
    )

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for index, data_traj in enumerate(loader):
            file_path = f"{args.destination_dir}/{args.dataset_name}/output_{index}.vla"
            if os.path.exists(file_path):
                future = executor.submit(check_and_fix_conversion, file_path, data_traj, args.dataset_name, index, args.destination_dir, args.lossless)
                futures.append(future)
        
        time.sleep(60)
        failed_conversions = []
        for future in as_completed(futures):
            index, success = future.result()
            if not success:
                failed_conversions.append(index)

    if failed_conversions:
        print(f"Failed to fix {len(failed_conversions)} conversions: {failed_conversions}")
    else:
        print("All existing conversions are valid or have been successfully fixed.")

if __name__ == "__main__":
    main()
