import fog_x
import os
import h5py
import argparse
import numpy as np
from fog_x.utils import recursively_read_hdf5_group, _flatten
from multiprocessing import Pool, cpu_count

def process_trajectory(args):
    h5_file_path, traj_name, output_dir = args
    
    with h5py.File(h5_file_path, 'r') as h5_file:
        print(f"Processing trajectory: {traj_name}")
        traj_group = h5_file[traj_name]
        
        traj_dict = _flatten(recursively_read_hdf5_group(traj_group))
        
        traj_len = len(traj_dict["observation/wrist_image_left"])
        
        for key in traj_dict:
            if "language_instruction" in key:
                traj_dict[key] = [np.array(traj_dict[key], dtype=object) for _ in range(traj_len)]
            if "language_embedding" in key:
                traj_dict[key] = [traj_dict[key] for _ in range(traj_len)]
            if "image" in key:
                image = np.frombuffer(traj_dict[key], dtype=np.uint8).reshape((-1, 320,180,3))
                print(image.shape)
                traj_dict[key] = image

    vla_path = os.path.join(output_dir, f"{traj_name}.vla")
    fog_x.Trajectory.from_dict_of_lists(traj_dict, path=vla_path, lossy_compression=False)

def convert_h5_to_vla_trajectories(h5_file_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    

    with h5py.File(h5_file_path, 'r') as h5_file:
        traj_names = list(h5_file.keys())
        for traj_name in traj_names:
            process_trajectory((h5_file_path, traj_name, output_dir))
        

    # num_processes = cpu_count()
    # with Pool(processes=num_processes) as pool:
    #     args_list = [(h5_file_path, traj_name, output_dir) for traj_name in traj_names]
    #     pool.map(process_trajectory, args_list)
    

def main():
    parser = argparse.ArgumentParser(description="Convert HDF5 file to VLA trajectories.")
    parser.add_argument("--h5_file", required=True, help="Path to the input HDF5 file")
    parser.add_argument("--output_dir", required=True, help="Directory to save VLA trajectories")

    args = parser.parse_args()

    convert_h5_to_vla_trajectories(args.h5_file, args.output_dir)
    print(f"Conversion completed. VLA trajectories saved to {args.output_dir}")

if __name__ == "__main__":
    main()