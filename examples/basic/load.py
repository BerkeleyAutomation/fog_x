import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

import fog_x

print(pl.scan_pyarrow_dataset(ds.dataset("~/test_dataset/steps")).collect())
