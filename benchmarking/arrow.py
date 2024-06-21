import os
import sys
import time
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.feather as ft

PATH = os.path.expanduser("~")
sys.path.append(PATH + "/fog_x_fork")

PATH += "/datasets/"
NAME = "test_convert"
out_dir = PATH + "arrow_convert"


def convert_parquet(in_dir):
    table = pq.read_table(in_dir)
    file = in_dir.split("/")[-1].split("-")[0]
    
    with pa.ipc.new_file(f"{out_dir}/IPC/{file}.arrow", table.schema) as w:
        w.write_table(table)

    with pa.OSFile(f"{out_dir}/REG/{file}.arrow", "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as w:
            w.write_table(table)

    ft.write_feather(table, f"{out_dir}/fth/{file}.arrow", "uncompressed")


N = 51
MB = 1024 * 1024

if False:
    for i in range(N):
        convert_parquet(f"{PATH}/{NAME}/{NAME}_{i}-0.parquet")

if __name__ == "__main__":
    exit()