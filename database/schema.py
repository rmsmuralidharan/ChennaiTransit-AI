from database.connection import DatabaseConnection

from sqlalchemy import text


class DatabaseSchemaManager:
    def __init__(self):

        self.database_connection = DatabaseConnection()
        self.engine = self.database_connection.get_engine()

    def create_schemas(self):
        with self.engine.begin() as connection:
            connection.execute(
                text("CREATE SCHEMA IF NOT EXISTS raw")
            )
            connection.execute(
                text("CREATE SCHEMA IF NOT EXISTS transformed")
            )
            connection.execute(
                text("CREATE SCHEMA IF NOT EXISTS final")
            )
            