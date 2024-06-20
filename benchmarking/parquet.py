import os
import sys
import time
import polars as pl
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.dataset as ds

PATH = os.path.expanduser("~")
sys.path.append(PATH + "/fog_x_fork")
from fog_x.dataset import Dataset

DEBUG = False
PATH += "/datasets/"
NAME = "test_convert"
dataset = Dataset(name=NAME, path=PATH)


if DEBUG: # Debugging code for call tracing thru <fog_x>, collapse for readability
    if False:
        dataset.load_rtx_episodes(name="berkeley_autolab_ur5", split="train[:51]")

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
        # traj_info_c.select_table(dataset_name)
        # -> traj_info_c.tables[dataset_name]

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


N = 51
MB = 1024 * 1024
SDC = dataset.db_manager.step_data_connector

def measure_traj(read_func, write_func):
    read_time, write_time, data_size = 0, 0, 0

    for i in range(N):
        print(f"Measuring trajectory {i}")
        path = f"{SDC.path}/{NAME}_{i}-0.parquet"

        stop = time.time()
        traj = read_func(path)
        read_time += time.time() - stop

        data_size += os.path.getsize(path)
        path = PATH + "temp.parquet"

        stop = time.time()
        write_func(path, traj)
        write_time += time.time() - stop

        os.remove(path)
    return read_time, write_time, data_size / MB


if __name__ == "__main__":

    fx_dict = {"name": "Fog-X",
        "read_func": lambda path: SDC.tables[path.split("/")[-1].split("-")[0]].collect(),
        "write_func": lambda path, data: data.write_parquet(path)
    }
    pl_dict = {"name": "Polars",
        "read_func": lambda path: pl.read_parquet(path),
        "write_func": lambda path, data: data.write_parquet(path)
    }
    pq_dict = {"name": "PyArrow",
        "read_func": lambda path: pl.from_arrow(pq.read_table(path)),
        "write_func": lambda path, data: pq.write_table(data.to_arrow(), path)
    }
    pd_dict = {"name": "Pandas",
        "read_func": lambda path: pd.read_parquet(path),
        "write_func": lambda path, data: data.to_parquet(path)
    }

    for lib in [fx_dict, pl_dict, pq_dict, pd_dict]:
        rt, wt, mb = measure_traj(lib["read_func"], lib["write_func"])

        print(f"\n{lib['name']}: \nData size = {mb:.4f} MB; Num. traj = {N}")
        print(f"Read:  latency = {rt:.4f} s; throughput = {mb / rt :.4f} MB/s, {N / rt :.4f} traj/s")
        print(f"Write: latency = {wt:.4f} s; throughput = {mb / wt :.4f} MB/s, {N / wt :.4f} traj/s")