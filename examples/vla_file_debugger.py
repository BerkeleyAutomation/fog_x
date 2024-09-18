import os
import numpy as np
from fog_x.trajectory import Trajectory
from fog_x.utils import _flatten
import imageio
from fog_x.loader import RLDSLoader

def load_ffv1_trajectory(path):
    traj = Trajectory(path,)
    return _flatten(traj.load())

def load_vla_trajectory(path):
    traj = Trajectory(path)
    return _flatten(traj.load())

def load_rlds_trajectory(path, dataset_name, version, split, index):
    loader = RLDSLoader(path=f"{path}/{dataset_name}/{version}", split=split, shuffling=False)
    data_traj = loader[index]
    
    data = {}
    # convert from a list of dicts to a dict of lists
    traj_len = len(data_traj)
    for i in range(traj_len):
        data_traj[i] = _flatten(data_traj[i])
        for k, v in data_traj[i].items():
            if k == "observation/natural_language_instruction":
                print(v)
                continue
            if k not in data:
                data[k] = np.empty((traj_len, *v.shape))
            data[k][i] = v
    return data

def save_traj_images_to_dir(traj_data, dir_path):
    os.makedirs(dir_path, exist_ok=True)
    for i in range(len(traj_data["observation/image"])):
        imageio.imwrite(f"{dir_path}/{i}.png", traj_data["observation/image"][i].astype(np.uint8))

def compare_trajectories(ffv1_data, vla_data, rlds_data, file_name):
    print(f"\nComparing FFV1, VLA, and RLDS trajectories for {file_name}:")
    
    # Compare keys
    ffv1_keys = set(ffv1_data.keys())
    vla_keys = set(vla_data.keys())
    rlds_keys = set(rlds_data.keys())
    
    print(f"FFV1 keys: {ffv1_keys}")
    print(f"VLA keys: {vla_keys}")
    print(f"RLDS keys: {rlds_keys}")
    
    common_keys = ffv1_keys.intersection(vla_keys).intersection(rlds_keys)
    
    # Compare data for common keys
    for key in common_keys:
        if key == "observation/natural_language_instruction":
            continue
        ffv1_array = ffv1_data[key]
        vla_array = vla_data[key]
        rlds_array = rlds_data[key]
        
        print(f"\nComparing '{key}':")
        print(f"  FFV1 shape: {ffv1_array.shape}, dtype: {ffv1_array.dtype}")
        print(f"  VLA shape: {vla_array.shape}, dtype: {vla_array.dtype}")
        print(f"  RLDS shape: {rlds_array.shape}, dtype: {rlds_array.dtype}")
        
        if ffv1_array.shape == vla_array.shape == rlds_array.shape: #and ffv1_array.dtype == vla_array.dtype == rlds_array.dtype:
            if np.allclose(ffv1_array, vla_array) and np.allclose(ffv1_array, rlds_array):
                continue
            else:
                diff_ffv1_vla = np.abs(ffv1_array - vla_array)
                diff_ffv1_rlds = np.abs(ffv1_array - rlds_array)
                diff_vla_rlds = np.abs(vla_array - rlds_array)
                print(f"  Max difference FFV1-VLA: {np.max(diff_ffv1_vla)}")
                print(f"  Max difference FFV1-RLDS: {np.max(diff_ffv1_rlds)}")
                print(f"  Max difference VLA-RLDS: {np.max(diff_vla_rlds)}")
                print(f"  Mean difference FFV1-VLA: {np.mean(diff_ffv1_vla)}")
                print(f"  Mean difference FFV1-RLDS: {np.mean(diff_ffv1_rlds)}")
                print(f"  Mean difference VLA-RLDS: {np.mean(diff_vla_rlds)}")
                if key == "observation/image":
                    print("ffv1_array[0]: ", ffv1_array[0])
                    print("vla_array[0]: ", vla_array[0])
                    print("rlds_array[0]: ", rlds_array[0])
                    save_traj_images_to_dir(ffv1_data, f"{file_name}_ffv1")
                    save_traj_images_to_dir(vla_data, f"{file_name}_vla")
                    save_traj_images_to_dir(rlds_data, f"{file_name}_rlds")
        else:
            print("  Shape or dtype mismatch")
            print(f" ffv1: {np.sum(ffv1_array - np.array(rlds_array))}")
            print(f" vla: {np.sum(vla_array - np.array(rlds_array))}")

def main():
    # dataset_name = "bridge"
    dataset_name = "fractal20220817_data"
    base_path = f"/home/kych/datasets/{dataset_name}"
    # base_path = "/mnt/data/fog_x"
    ffv1_dir = os.path.join(base_path, "ffv1", dataset_name)
    vla_dir = os.path.join(base_path, "vla", dataset_name)
    rlds_dir = "/home/kych/datasets/rtx"
    version = "0.1.0"
    split = "train"

    # Get all .vla files in the ffv1 directory
    vla_files = ["output_{}.vla".format(i) for i in range(1)]

    for file_name in vla_files:
        ffv1_file = os.path.join(ffv1_dir, file_name)
        vla_file = os.path.join(vla_dir, file_name)
        index = int(file_name.split("_")[1].split(".")[0])

        if not os.path.exists(vla_file):
            print(f"Skipping {file_name}: VLA file not found")
            continue

        print(f"\nProcessing {file_name}")
        ffv1_data = load_ffv1_trajectory(ffv1_file)
        vla_data = load_vla_trajectory(vla_file)
        rlds_data = load_rlds_trajectory(rlds_dir, dataset_name, version, split, index)
        
        compare_trajectories(ffv1_data, vla_data, rlds_data, file_name)

if __name__ == "__main__":
    main()