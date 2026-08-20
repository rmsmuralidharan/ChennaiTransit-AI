import pandas as pd

from database.connection import DatabaseConnection

class DemandTransformer:
    def __init__(self):
        self.database = DatabaseConnection()

    def transform(self):
        engine = self.database.get_engine()

        demand_df = pd.read_sql(
            'select * from raw.demand',
            con = engine
        ) 

        print(
            f"Rows loaded for transformation: {len(demand_df)}"
        )

        ## transformation

        demand_df['timestamp'] = pd.to_datetime(
            demand_df['timestamp']
        )

        columns_to_keep = [
        "station_id",
        "station_name",
        "zone",
        "latitude",
        "longitude",
        "timestamp",
        "hour",
        "day_of_week",
        "is_weekend",
        "demand_profile",
        "weather_condition",
        "temperature",
        "humidity",
        "rainfall",
        "passenger_count"
    ]

        demand_df = demand_df[columns_to_keep]

        print("\nColumns after transformation:")

        print(demand_df.columns.tolist())

        demand_df.to_sql(
            name='demand',
            con=engine,
            schema='transformed',
            if_exists='replace',
            index=False
        )

        print(
            "Transformed demand data loaded successfully "
            "into transformed.demand"
        )

if __name__ == "__main__":

    transformer = DemandTransformer()
    transformer.transform()


