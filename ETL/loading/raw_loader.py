import pandas as pd

from pathlib import Path

from database.connection import DatabaseConnection

PROJECT_ROOT = Path(__file__).parents[2]

RAW_FILE_PATH = PROJECT_ROOT / 'data' / 'generated'/ 'demand.csv'

class RawDataLoader:
    def __init__(self):
        self.database = DatabaseConnection()

    def load_demand_data(Self):
        if not RAW_FILE_PATH.exists():
            raise FileNotFoundError(
                f"raw demand file not found: {RAW_FILE_PATH}"
            )

        demand_df = pd.read_csv(RAW_FILE_PATH)

        print(
            f"rows loaded from csv: {len(demand_df)}"
        )

        engine = Self.database.get_engine()

        demand_df.to_sql(
            name='demand',
            con=engine,
            schema='raw',
            if_exists='replace',
            index=False
        )

        print(
            'raw demand data loaded successfully into raw.demand'
        )

        return len(demand_df)


if __name__ == "__main__":
    loader = RawDataLoader()

    loader.load_demand_data()