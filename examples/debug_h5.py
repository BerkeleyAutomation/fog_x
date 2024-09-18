import h5py
import numpy as np
import argparse
import cv2
def compare_datasets(dataset1, dataset2):
    if "language" in  dataset1.name:
        return True, "Identical"
    if "image" in dataset1.name:
        print("image",dataset1.shape, dataset2.shape, "check manually")
        return True, "Identical"
    if dataset1.shape != dataset2.shape:
        return False, f"Shape mismatch: {dataset1.shape} vs {dataset2.shape}"
    
    if dataset1.dtype != dataset2.dtype:
        return False, f"Dtype mismatch: {dataset1.dtype} vs {dataset2.dtype}"
    
    if isinstance(dataset1[()], np.ndarray):
        if not np.allclose(dataset1[()], dataset2[()], equal_nan=True):
            return False, "Array contents differ"
    elif dataset1[()] != dataset2[()]:
        return False, f"Values differ: {dataset1[()]} vs {dataset2[()]}"

    return True, "Identical"

def compare_groups(group1, group2):
    keys1 = set(group1.keys())
    keys2 = set(group2.keys())
    
    if keys1 != keys2:
        return False, f"Group keys differ: {keys1} vs {keys2}"
    
    for key in keys1:
        if isinstance(group1[key], h5py.Group):
            if not isinstance(group2[key], h5py.Group):
                return False, f"Item type mismatch for {key}: Group vs {type(group2[key])}"
            is_identical, message = compare_groups(group1[key], group2[key])
            if not is_identical:
                return False, f"Subgroup {key}: {message}"
        elif isinstance(group1[key], h5py.Dataset):
            if not isinstance(group2[key], h5py.Dataset):
                return False, f"Item type mismatch for {key}: Dataset vs {type(group2[key])}"
            is_identical, message = compare_datasets(group1[key], group2[key])
            if not is_identical:
                return False, f"Dataset {key}: {message}"
        else:
            return False, f"Unsupported item type for {key}: {type(group1[key])}"
    
    return True, "Identical"

def get_common_groups(file1, file2):
    groups1 = set(file1.keys())
    groups2 = set(file2.keys())
    return groups1.intersection(groups2)

def compare_h5_files(file1_path, file2_path):
    with h5py.File(file1_path, 'r') as f1, h5py.File(file2_path, 'r') as f2:
        print(f1.keys())
        print(f2.keys())
        common_groups = get_common_groups(f1, f2)
        
        if not common_groups:
            print("No common groups found between the two files.")
            return

        for group_name in common_groups:
            print(f"\nComparing group: {group_name}")
            group1 = f1[group_name]
            group2 = f2[group_name]

            is_identical, message = compare_groups(group1, group2)
            
            if is_identical:
                print(f"The group '{group_name}' is identical in both files.")
            else:
                print(f"The group '{group_name}' differs: {message}")

def main():
    parser = argparse.ArgumentParser(description="Compare two HDF5 files.")
    parser.add_argument("file1", help="Path to the first HDF5 file")
    parser.add_argument("file2", help="Path to the second HDF5 file")
    
    args = parser.parse_args()
    
    compare_h5_files(args.file1, args.file2)

if __name__ == "__main__":
    main()
