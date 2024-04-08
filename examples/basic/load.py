import fog_rtx




import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
import pyarrow.dataset as ds

print(pl.scan_pyarrow_dataset(ds.dataset("~/test_dataset/steps")).collect())