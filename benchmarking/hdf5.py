import os
import time
import h5py as h5
import pandas as pd
import tables as tb

# malabar_datasets.h5: 1,699,090,853 bytes (1.71 GB on disk)

PATH = os.path.expanduser("~") + "/datasets/hdf5/"
DATA = [PATH + p for p in os.listdir(PATH)]
MB = 1024 * 1024

def measure(read_func, write_func):
    read_time, write_time, data_size = 0, 0, 0

    for path in DATA:
        start = time.time()
        read_data = read_func(path)
        read_time += time.time() - start

        start = time.time()
        write_func(PATH + "temp.h5", read_data)
        write_time += time.time() - start

        os.remove(PATH + "temp.h5")
        data_size += os.path.getsize(path)

    return read_time, write_time, data_size / MB

def read_h5py(path):
    with h5.File(path, 'r') as f:
        distances = h5.File(path, 'r')['distances'][:]
        seqid = h5.File(path, 'r')['seqid'][:]
        word = h5.File(path, 'r')['word'][:]
    return distances, seqid, word


D, S, W = "distances", "seqid", "word"

def write_pandas(path, data):
    d, s, w = data
    d.to_hdf(path, key=D, mode="w")
    s.to_hdf(path, key=S, append=True)
    w.to_hdf(path, key=W, append=True)

def write_h5py(path, data):
    d, s, w = data
    with h5.File(path, "w") as f:
        f.create_dataset(D, data=d)
        f.create_dataset(S, data=s)
        f.create_dataset(W, data=w)

def read_tables(path):
    with tb.open_file(path, "r").root as f:
        return [f.distances[:], f.seqid[:], f.word[:]]

def write_tables(path, data):
    d, s, w = data
    with tb.open_file(path, "w") as f:
        f.create_array(f.root, D, d)
        f.create_array(f.root, S, s)
        f.create_array(f.root, W, w)

pd_dict = {"name": "Pandas",
      "read_func": lambda path: [pd.read_hdf(path, i) for i in [D, S, W]],
     "write_func": write_pandas
}
h5_dict = {"name": "h5py",
      "read_func": lambda path: [h5.File(path, "r")[i][:] for i in [D, S, W]],
     "write_func": write_h5py
}
tb_dict = {"name": "Tables",
      "read_func": read_tables,
     "write_func": write_tables
}

for lib in [h5_dict]:
    rt, wt, ds = measure(lib["read_func"], lib["write_func"])

    print(f"\n{lib['name']}: data size = {ds:.4f} MB")
    print(f"Read:  latency = {rt:.4f} s; throughput = {ds / rt :.4f} MB/s")
    print(f"Write: latency = {wt:.4f} s; throughput = {ds / wt :.4f} MB/s")