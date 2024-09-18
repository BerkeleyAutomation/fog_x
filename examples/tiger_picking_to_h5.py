import fog_x
import os
import h5py
import argparse
from fog_x.loader import VLALoader
import numpy as np
from collections import defaultdict

epi_len_mapping_json = {}

def convert_vla_to_unified_h5(vla_dir, output_path):
    vla_files = [f for f in os.listdir(vla_dir) if f.endswith('.vla')]
    
    with h5py.File(output_path, 'w') as h5_file:
        for idx, vla_filename in enumerate(vla_files):
            print(f"Processing {vla_filename} ({idx + 1}/{len(vla_files)})")
            
            vla_path = os.path.join(vla_dir, vla_filename)
            traj = fog_x.Trajectory(vla_path)
            trajectory = traj.load()
            
            shape_dict = {}
            for key, value in trajectory.items():
                shape_dict[key] = value.shape
            print("shape_dict", shape_dict)
            traj_len = len(trajectory["observation/wrist_image_left"])
            epi_len_mapping_json[vla_filename] = traj_len
            print("traj_len", vla_filename, traj_len)
            
            # Create a group for each trajectory
            group_name = os.path.splitext(vla_filename)[0]
            group = h5_file.create_group(group_name)
            
            # Convert trajectory data to HDF5 format
            for key, value in trajectory.items():
                print(f"Saving {key} with shape {value.shape}")
                
                if "language_instruction" in key.lower():
                    # revert: data[key] = np.array([value for _ in range(traj_len)]).reshape(-1)
                    group.create_dataset(key, data=value[0])
                elif "language_embedding" in key.lower():
                    # revert: data[key] = np.array([value for _ in range(traj_len)]).reshape(traj_len, -1)
                    group.create_dataset(key, data=value[0])
                elif "image" in key.lower():
                    # image = np.frombuffer(value, dtype='uint8').reshape(-1,180,320,3)
                    # image_np_buffer = value.tobytes()
                    group.create_dataset(key, data=value)
                elif isinstance(value, (list, tuple)):
                    group.create_dataset(key, data=value)
                elif hasattr(value, 'shape'):
                    group.create_dataset(key, data=value)
                else:
                    group.attrs[key] = value

    import json 
    with open(os.path.join("/home/kych/output/", "epi_len_mapping.json"), 'w') as f:
        json.dump(epi_len_mapping_json, f)
        
def main():
    parser = argparse.ArgumentParser(description="Convert VLA files to a unified HDF5 file.")
    parser.add_argument("--vla_dir", required=True, help="Directory containing VLA files")
    parser.add_argument("--output_path", required=True, help="Path for the output unified HDF5 file")

    args = parser.parse_args()

    convert_vla_to_unified_h5(args.vla_dir, args.output_path)
    print(f"Conversion completed. Unified HDF5 file saved to {args.output_path}")

if __name__ == "__main__":
    main()
