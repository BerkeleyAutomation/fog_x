import os
import time
import pandas as pd
import pyarrow.ipc as ipc
import pyarrow.feather as fth

# taxi_trip_data.parquet: 472,757,547 bytes (486.9 MB on disk)

PATH = os.path.expanduser("~") + "/datasets/arrow/"
DATA = [PATH + p for p in os.listdir(PATH)]
MB = 1024 * 1024

def measure(read_func, write_func, lib):
    read_time, write_time, data_size = 0, 0, 0

    ext = ".arrow" if (lib == "ArrowIPC") else ".feather"
    lst = [d for d in DATA if d.endswith(ext)]

    for path in lst:
        start = time.time()
        read_data = read_func(path)
        read_time += time.time() - start

        start = time.time()
        write_func(f"{PATH}temp{ext}", read_data)
        write_time += time.time() - start

        os.remove(f"{PATH}temp{ext}")
        data_size += os.path.getsize(path)

    return read_time, write_time, data_size / MB

pd_dict = {"name": "Pandas",
      "read_func": lambda path: pd.read_feather(path),
     "write_func": lambda path, data: data.to_feather(path)
}
ipc_dict = {"name": "ArrowIPC",
       "read_func": lambda path: ipc.open_file(path).read_all(),
      "write_func": lambda path, data: ipc.new_file(path, data.schema).write(data)
}
fth_dict = {"name": "Feather",
       "read_func": lambda path: fth.read_table(path),
      "write_func": lambda path, data: fth.write_feather(data, path),
}

for lib in [pd_dict, ipc_dict, fth_dict]:
    rt, wt, ds = measure(lib["read_func"], lib["write_func"], lib["name"])

    print(f"\n{lib['name']}: data size = {ds:.4f} MB")
    print(f"Read:  latency = {rt:.4f} s; throughput = {ds / rt :.4f} MB/s")
    print(f"Write: latency = {wt:.4f} s; throughput = {ds / wt :.4f} MB/s")