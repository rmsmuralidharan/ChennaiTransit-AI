from pathlib import Path
import yaml
import dotenv as de
import os

from ChennaiTransit_AI.exception.exception import ChennaiTransitAIException

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_FILE = PROJECT_ROOT / ".env"

CONFIG_FILE = PROJECT_ROOT / "config" / "config.yaml"

de.load_dotenv(ENV_FILE)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("ENV_FILE:", ENV_FILE)
print("ENV_FILE EXISTS:", ENV_FILE.exists())
print("USERNAME:", os.getenv("DATABASE_USERNAME"))
print("PASSWORD EXISTS:", os.getenv("DATABASE_PASSWORD") is not None)

class ConfigurationManager:
    def __init__(self):
        with open(CONFIG_FILE, 'r') as cf:
           self.config = yaml.safe_load(cf)

    def get_database_config(self):
        database_config = self.config['database']

        username = os.getenv('DATABASE_USERNAME')
        password = os.getenv('DATABASE_PASSWORD')

        if not username or password is None:
            raise ChennaiTransitAIException(
                "Database credentials are missing"
            )

        database_config['username'] = username
        database_config['password'] = password

        return {
            **database_config,
            "username": username,
            "password": password
        }

    def get_data_config(self):
        data_config = self.config['data']

        return data_config


    def get_etl_config(self):
        etl_config = self.config['ETL']

        return etl_config





           
