import logging
from typing import List
import pyarrow as pa
import pyarrow.parquet as pq

import polars as pl

from fog_rtx.database.utils import _datasets_dtype_to_pld

logger = logging.getLogger(__name__)


class PolarsConnector:
    def __init__(self, path: str):
        # In Polars, data is directly read into DataFrames from files, not through a database engine
        self.path = path
        self.tables = (
            {}
        )  # This will store table names as keys and DataFrames as values
        self.table_len = {}

    def load_tables(self):
        # Load all tables from the path
        dataset = pq.ParquetDataset(self.path)
        for table in dataset.tables:
            self.tables[table.name] = table.to_polars()
            self.table_len[table.name] = len(self.tables[table.name])
        logger.info(f"Tables loaded from {self.path}.")

    def list_tables(self):
        # Listing available DataFrame tables
        return list(self.tables.keys())

    def save_tables(self, tables: List[str]):        
        for table_name in tables:
            table = self.tables[table_name].to_arrow()
            pq.write_to_dataset(table, root_path=self.path)


    def create_table(self, table_name: str, as_lazy_frame: bool = False):
        # Create a new DataFrame with specified columns
        # schema = {column_name: _datasets_dtype_to_arrow(column_type) for column_name, column_type in columns.items()}
        self.tables[table_name] = pl.DataFrame()
        if as_lazy_frame:
            self.tables[table_name] = self.tables[table_name].lazy()
        logger.info(f"Table {table_name} created with Polars.")
        logger.info(f"writing to {self.path}/{table_name}.parquet")
        self.table_len[table_name] = 0 # no entries yet

    def add_column(self, table_name: str, column_name: str, column_type):
        if column_name in self.tables[table_name].columns:
            logger.warning(
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
            logger.info(f"Column {column_name} added to table {table_name}.")
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

            self.table_len[table_name] += 1 

            return index_of_new_row  # Return the index of the inserted row
        else:
            logger.error(f"Table {table_name} does not exist.")
            return None  # Or raise an exception depending on your error handling strategy

    def update_data(self, table_name: str, index: int, data: dict):
        # update data
        for column_name, value in data.items():
            logger.info(
                f"updating {column_name} with {value} at index {index}"
            )
            self.tables[table_name][index, column_name] = value

    def merge_tables_with_timestamp(
        self, tables: List[str], output_table: str
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
                self.tables[tables[1]], on="Timestamp", strategy="nearest"
            )
            for table_name in tables[2:]:
                merged_df = merged_df.join_asof(
                    self.tables[table_name].drop("episode_id"), on="Timestamp", strategy="nearest"
                )
            logger.info("Tables merged on Timestamp.")
        else:
            merged_df = self.tables[tables[0]]
            logger.info("Only one table to merge.")
        self.tables[output_table] = merged_df

    def select_table(self, table_name: str):
        # Return the DataFrame
        if table_name in self.tables:
            return self.tables[table_name]
        else:
            logger.error(f"Table {table_name} does not exist.")
            return None

class DataFrameConnector(PolarsConnector):
    def __init__(self, path: str):
        super().__init__(path)
        self.tables = {}
        self.table_len = {}
    
    def load_tables(self): 
        pass

class LazyConnector(PolarsConnector):
    pass 