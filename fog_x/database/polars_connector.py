import logging
import os
from typing import List

import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from fog_x.database.utils import _datasets_dtype_to_pld
import smart_open
logger = logging.getLogger(__name__)


class PolarsConnector:
    def __init__(self, path: str):
        # In Polars, data is directly read into DataFrames from files, not through a database engine
        self.path = path
        self.tables = (
            {}
        )  # This will store table names as keys and DataFrames as values
        self.table_len = {}

    def list_tables(self):
        # Listing available DataFrame tables
        return list(self.tables.keys())

    def add_column(self, table_name: str, column_name: str, column_type):
        if column_name in self.tables[table_name].columns:
            logger.debug(
                f"Column {column_name} already exists in table {table_name}."
            )
            return
        # Add a new column to an existing DataFrame
        if table_name in self.tables:
            arrow_type = _datasets_dtype_to_pld(column_type)
            # self.tables[table_name] = self.tables[table_name].with_column(pl.lit(None).alias(column_name).cast(column_type))
            # self.tables[table_name] = self.tables[table_name].with_columns(pl.lit(None).alias(column_name).cast(arrow_type))
            self.tables[table_name] = self.tables[table_name].with_columns(
                pl.Series(
                    column_name, [None] * self.table_len[table_name]
                ).cast(arrow_type)
            )
            logger.debug(f"Column {column_name} added to table {table_name}.")
        else:
            logger.error(f"Table {table_name} does not exist.")

    def insert_data(self, table_name: str, data: dict):
        # Inserting a new row into the DataFrame and return the index of the new row
        if table_name in self.tables:
            # use the schema of the original table
            new_row = pl.DataFrame(
                [data], schema=self.tables[table_name].schema
            )

            index_of_new_row = self.table_len[table_name]
            logger.debug(
                f"Inserting data into {table_name}: {data} with {new_row} to table {self.tables[table_name]}"
            )
            self.tables[table_name] = pl.concat(
                [self.tables[table_name], new_row], how="align"
            )
            # logger.debug(f"table is now {self.tables[table_name]}")

            self.table_len[table_name] += 1

            return index_of_new_row  # Return the index of the inserted row
        else:
            logger.error(f"Table {table_name} does not exist.")
            return None  # Or raise an exception depending on your error handling strategy

    def update_data(self, table_name: str, index: int, data: dict):
        # update data
        for column_name, value in data.items():
            logger.debug(
                f"updating {column_name} with {value} at index {index}"
            )
            self.tables[table_name][index, column_name] = value

    def merge_tables_with_timestamp(
        self, tables: List[str], output_table: str, 
        clear_feature_tables: bool = False
    ):
        for table_name in self.tables.keys():
            if table_name not in tables:
                continue
            self.tables[table_name] = self.tables[table_name].set_sorted(
                "Timestamp"
            )

        # Merging tables using timestamps
        if len(tables) > 1:
            merged_df = self.tables[tables[0]].join_asof(
                self.tables[tables[1]].drop("episode_id"), on="Timestamp", strategy="nearest"
            )
            for table_name in tables[2:]:
                merged_df = merged_df.join_asof(
                    self.tables[table_name].drop("episode_id"),
                    on="Timestamp",
                    strategy="nearest",
                )
            logger.debug("Tables merged on Timestamp.")
        else:
            merged_df = self.tables[tables[0]]
            logger.debug("Only one table to merge.")
        self.tables[output_table] = merged_df

    def remove_tables(self, tables: List[str]):
        for table_name in tables:
            self.tables.pop(table_name)
            logger.debug(f"Table {table_name} removed.")

    def select_table(self, table_name: str):
        # Return the DataFrame
        if table_name in self.tables:
            return self.tables[table_name]
        else:
            logger.error(
                f"Table {table_name} does not exist, available tables are {self.tables.keys()}."
            )
            return None


class DataFrameConnector(PolarsConnector):
    def __init__(self, path: str):
        super().__init__(path)
        self.tables = {}
        self.table_len = {}

    def create_table(self, table_name: str):
        # Create a new DataFrame with specified columns
        # schema = {column_name: _datasets_dtype_to_arrow(column_type) for column_name, column_type in columns.items()}
        self.tables[table_name] = pl.DataFrame()
        logger.debug(f"Table {table_name} created with Polars.")
        self.table_len[table_name] = 0  # no entries yet


    def load_tables(self, table_names: List[str]):
        
        # load tables from the path
        for table_name in table_names:
            path = self.path.removesuffix("/")
            path = f"{path}/{table_name}.parquet"
            logger.info(f"Prepare to load table {table_name} loaded from {path}.")

            # use smart_open to handle different file systems
            path = os.path.expanduser(path)
            try:
                with smart_open.open(path):
                    self.tables[table_name] = pl.from_arrow(pq.read_table(path))
                    self.table_len[table_name] = len(self.tables[table_name])
                    logger.info(f"Table {table_name} loaded from {path}.")
            except Exception as e:
                logger.warn(f"Failed to load table {table_name} from {path}.")
                continue 



    def save_table(self, table_name: str):
        pq.write_table(self.tables[table_name].to_arrow(), f"{self.path}/{table_name}.parquet")


class LazyFrameConnector(PolarsConnector):
    def __init__(self, path: str):
        super().__init__(path)
        self.tables = {}
        self.table_len = {}
        self.dataset = pl.scan_pyarrow_dataset(
            ds.dataset(self.path, format="parquet")
        )

    def load_tables(self, episode_ids: List[str], table_names: List[str]):
        # rescan the dataset
        self.dataset = pl.scan_pyarrow_dataset(
            ds.dataset(self.path, format="parquet")
        )

        for table_name, episode_id in zip(table_names, episode_ids):
            self.tables[table_name] = self.dataset.filter(
                pl.col("episode_id") == episode_id
            )
            # logger.debug(f"Tables loaded from {self.tables}")

    def create_table(self, table_name: str):
        # Create a new DataFrame with specified columns
        # schema = {column_name: _datasets_dtype_to_arrow(column_type) for column_name, column_type in columns.items()}
        self.tables[table_name] = pl.DataFrame()
        logger.debug(f"Table {table_name} created with Polars.")
        self.table_len[table_name] = 0  # no entries yet

    def save_table(self, table_name: str):
        # for table_name in tables:
        #     table = self.tables[table_name].to_arrow()
        #     pq.write_to_dataset(table, root_path=self.path)
        dataset = self.tables[table_name].to_arrow()
        # dataset = [self.dataset.to_table(), ]
        basename_template = f"{table_name}-{{i}}.parquet"
        ds.write_dataset(
            dataset,
            base_dir=self.path,
            basename_template=basename_template,
            format="parquet",
            existing_data_behavior="overwrite_or_ignore",
        )

    def get_dataset_table(self, reload:bool = False):
        if reload:
            self.dataset = pl.scan_pyarrow_dataset(
                ds.dataset(self.path, format="parquet")
            )
        return self.dataset