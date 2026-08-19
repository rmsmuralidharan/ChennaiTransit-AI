from sqlalchemy import create_engine

from ChennaiTransit_AI.utils.configuration import ConfigurationManager

class DatabaseConnection:
    def __init__(self):
        self.configuration_manager = ConfigurationManager()
        self.database_config = (
            self.configuration_manager.get_database_config()
        )

        self.connection_url = (
            f"postgresql+psycopg2://{self.database_config['username']}"
            f":{self.database_config['password']}"
            f"@{self.database_config['host']}"
            f":{self.database_config['port']}/"
            f"{self.database_config['name']}"
        )

        self.engine = create_engine(
            self.connection_url
        )

    def get_engine(self):

        return self.engine


