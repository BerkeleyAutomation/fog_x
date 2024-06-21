import h5py
import pyarrow.parquet as pq

def parquet_to_hdf5(parquet_file, hdf5_file):
    # Read the Parquet file
    table = pq.read_table(parquet_file)
    columns = table.column_names

    # Create HDF5 file
    with h5py.File(hdf5_file, 'w') as h5_file:
        for column in columns:
            data = table[column].to_numpy()
            h5_file.create_dataset(column, data=data)
        print(f"Data successfully written to {hdf5_file}")

# Example usage
parquet_file = 'input_file.parquet'
hdf5_file = 'output_file.h5'
parquet_to_hdf5(parquet_file, hdf5_file)