import logging
from typing import Any, List

import pandas as pd  # type: ignore
import sqlalchemy  # type: ignore
from sqlalchemy import MetaData  # type: ignore
from sqlalchemy import Table  # type: ignore
from sqlalchemy import text  # type: ignore
from sqlalchemy import Column, Integer, create_engine, inspect
from sqlalchemy.orm import declarative_base, sessionmaker  # type: ignore

from fog_x.database.utils import type_py2sql  # type: ignore

Base = declarative_base()
logger = logging.getLogger(__name__)


class DatabaseConnector:
    def __init__(self, path: str):
        self.engine = create_engine(f"sqlite:///{path}")
        Base.metadata.bind = self.engine
        DBSession = sessionmaker(bind=self.engine)
        self.session = DBSession()

    def add(self, key, value):
        # This method should be updated based on the specific use case.
        pass

    def query(self, query):
        # This should be updated to use SQLAlchemy's query capabilities.
        pass

    def close(self):
        self.session.close()

    def list_tables(self):
        inspector = inspect(self.engine)
        return inspector.get_table_names()

    def create_table(self, table_name: str, columns):
        metadata = MetaData()
        columns = [Column("id", Integer, primary_key=True)] + [
            Column(column_name, column_type)
            for column_name, column_type in columns.items()
        ]
        table = Table(table_name, metadata, *columns)
        metadata.create_all(self.engine)
        logger.info(f"Table {table_name} created.")

    def add_column(self, table_name, column):
        column_name = column.compile(dialect=self.engine.dialect)
        column_type = column.type.compile(self.engine.dialect)
        sql = text(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
        )
        self.engine.execute(sql)
        # self.engine.execute('ALTER TABLE %s ADD COLUMN %s %s' % (table_name, column_name, column_type))

    def insert_data(
        self,
        table_name: str,
        data: dict,
        create_new_column_if_not_exist: bool = False,
    ) -> int:
        metadata = MetaData()
        table = Table(table_name, metadata, autoload_with=self.engine)

        logger.info(f"Inserting data into {table_name} with data {data}")
        # Check for each key in data if it exists as a column
        if create_new_column_if_not_exist:
            for key, value in data.items():
                if table.c.get(key) is None:
                    logger.warn(
                        f"Creating new column {key} in table {table_name} with type {type(value)}"
                    )
                    self.add_column(
                        table_name,
                        Column(key, type_py2sql(type(value)), nullable=True),
                    )

        insert_result = self.engine.execute(table.insert(), data)
        logger.warn(
            f"Data inserted into {table_name} with index {insert_result.inserted_primary_key[0]}"
        )
        return insert_result.inserted_primary_key[0]

    def update_data(
        self,
        table_name: str,
        index: int,
        data: dict,
        create_new_column_if_not_exist: bool = False,
        is_partial_data: bool = False,
    ):
        metadata = MetaData()
        table = Table(table_name, metadata, autoload_with=self.engine)

        logger.info(f"Inserting data into {table_name} with data {data}")
        # Check for each key in data if it exists as a column
        if create_new_column_if_not_exist:
            for key, value in data.items():
                if table.c.get(key) is None:
                    logger.warn(
                        f"Creating new column {key} in table {table_name} with type {type(value)}"
                    )
                    # self.add_column(
                    #     table_name,
                    #     Column(key, type_py2sql(type(value)), nullable=True),
                    # )
                    column_type = type_py2sql(type(value))
                    self.add_column(
                        table_name, Column(key, column_type, nullable=True)
                    )
                    metadata.clear()
                    table = Table(
                        table_name, metadata, autoload_with=self.engine
                    )
                    logger.info(
                        f"Successfully added column {key} to {table_name}"
                    )

        # if is_partial_data:
        #     self.engine.execute(
        #         table.update().where(table.c.id == index).values(**data)
        #     )
        #     logger.info(data)

        self.engine.execute(
            table.update().where(table.c.id == index).values(data)
        )
        logger.debug(f"Data updated in {table_name} at index {index}")

    def merge_tables_with_timestamp(
        self, tables: List[str], output_table: str
    ):
        merged_df = pd.DataFrame()
        # This method should be updated based on the specific use case.
        for table in tables:
            if merged_df.empty:
                merged_df = pd.read_sql(
                    f"SELECT * FROM {table}", self.engine
                ).drop("id", axis=1)
            else:
                cur_df = pd.read_sql(
                    f"SELECT * FROM {table}", self.engine
                ).drop("id", axis=1)
                merged_df = pd.merge_asof(
                    cur_df, merged_df, on="Timestamp", direction="nearest"
                )
                logger.info(f"merged: {merged_df}")
        logger.warn(
            """currently using merge asof nearest policy on the timstamp, 
            better policies should be expected"""
        )
        # policy can be X fps timestamp, and use backward merge_asof
        merged_df.to_sql(output_table, self.engine, if_exists="replace")

    def select_table(self, table_name: str, format: str = "sql") -> Any:
        if format == "sql":
            table = Table(table_name, MetaData(), autoload_with=self.engine)
            return self.engine.execute(
                sqlalchemy.select([table])
            )  # TODO: this may not work with the connection
        elif format == "pandas":
            return pd.read_sql(f"SELECT * FROM {table_name}", self.engine)
        else:
            raise ValueError(f"Unsupported table select format {format}")
