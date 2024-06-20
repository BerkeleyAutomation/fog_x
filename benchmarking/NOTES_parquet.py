import os
import sys
import time
import polars as pl
import pyarrow.dataset as ds

PATH = os.path.expanduser("~")
sys.path.append(PATH + "/fogrtx")
from fog_x.dataset import Dataset

DEBUG = False
PATH += "/datasets/"
NAME = "test_convert"

"""###"""

if DEBUG: # Debugging code for call tracing fog_x, collapse this for readability

    dataset = Dataset(name=NAME, path=PATH)
    # dataset.load_rtx_episodes(name="berkeley_autolab_ur5", split="train[:51]")

    # Debugging: path = "/home/joshzhang/datasets"
        # Starts with "/" but does not end with it
        # Should not initialize any cloud buckets!

    """###"""

    # Retrieve episode-level metadata as a pandas DataFrame
    episode_info = dataset.get_episode_info()

    # Codebase map:
    db_manager  = dataset.db_manager                # DatabaseManager
    traj_info_c = db_manager.episode_info_connector # DataFrameC, child of (PolarsC)
    step_data_c = db_manager.step_data_connector    # LazyFrameC, child of (PolarsC)

    # Call tracing:
        # dataset.get_episode_info()
        # db_manager.get_episode_info_table()
        # episode_info_c.select_table(dataset_name)
        # -> episode_info_c.tables[dataset_name]

    episode_info = traj_info_c.tables[NAME]     # Same object!
    print(episode_info)

    """###"""

    # Obtain the actual episodes containing their step data
    episode_0 = dataset.read_by(episode_info)[0]

    # Call tracing:
        # dataset.read_by(episode_info)
        # :: i = list(episode_info["episode_id"])[0]
        # db_manager.get_step_table(i)
        # step_data_c.select_table(dataset_name_i)
        # -> step_data_c.tables[dataset_name_i]

    episode_0 = step_data_c.tables[NAME + "_0"] # Same object!
    print(episode_0.collect())

    """###"""

    # Retrieve Fog-X dataset step data as a Polars LazyFrame
    step_data = dataset.get_step_data()

    # Call tracing:
        # dataset.get_step_data()
        # db_manager.get_step_table_all()
        # step_data_c.get_dataset_table()
        # :: d = ds.dataset(dataset_path, format="parquet")
        # -> pl.scan_pyarrow_dataset(d)

    d = ds.dataset(PATH + NAME, format="parquet")
    step_data = pl.scan_pyarrow_dataset(d)      # Same object!
    print(step_data.collect())

"""###"""

N = 51
MB = 1024 * 1024
SDC = db_manager.step_data_connector

def measure_traj():
    read_time, write_time, data_size = 0, 0, 0

    for i in range(N):
        print(f"measuring traj {i}")

        start = time.time()
        traj = SDC.tables[f"{NAME}_{i}"].collect()
        read_time += time.time() - start

        start = time.time()
        traj.write_parquet(PATH + "temp.parquet")
        write_time += time.time() - start

        os.remove(PATH + "temp.parquet")
        data_size += traj.estimated_size()

    return read_time, write_time, data_size / MB


rt, wt, mb = measure_traj()

print(f"\nPolars: data size = {mb:.4f} MB")
print(f"Read:  latency = {rt:.4f} s; throughput = {mb / rt :.4f} MB/s")
print(f"Write: latency = {wt:.4f} s; throughput = {mb / wt :.4f} MB/s")

print(f"\nPolars: num. traj = {N}")
print(f"Read:  latency = {rt:.4f} s; throughput = {N / rt :.4f} traj/s")
print(f"Write: latency = {wt:.4f} s; throughput = {N / wt :.4f} traj/s")

# pd_dict = {"name": "Pandas",
#       "read_func": lambda path: pd.read_parquet(path),
#      "write_func": lambda path, data: data.to_parquet(path)
# }
# fp_dict = {"name": "FastParquet",
#       "read_func": lambda path: fp.ParquetFile(path).to_pandas(),
#      "write_func": lambda path, data: fp.write(path, data)
# }
# pq_dict = {"name": "PyArrow",
#       "read_func": lambda path: pq.read_table(path),
#      "write_func": lambda path, data: pq.write_table(data, path)
# }