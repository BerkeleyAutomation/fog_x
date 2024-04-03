import logging
from typing import Any

from sqlalchemy import (
    Column,
    Integer,
    MetaData,
    Table,  # type: ignore
    create_engine,
    inspect,
)
from sqlalchemy.orm import declarative_base, sessionmaker  # type: ignore
from sqlalchemy.sql import select  # type: ignore

from fog_rtx.database import DatabaseConnector

Base = declarative_base()
logger = logging.getLogger(__name__)


class SQLite(DatabaseConnector):
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

    def insert_data(self, table_name: str, data: dict) -> int:
        table = Table(table_name, MetaData(), autoload_with=self.engine)
        insert_result = self.engine.execute(table.insert(), data)
        logger.info(
            f"Data inserted into {table_name} with index {insert_result.inserted_primary_key[0]}"
        )
        return insert_result.inserted_primary_key[0]
