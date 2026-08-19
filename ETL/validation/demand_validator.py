import pandas as pd

from database.connection import DatabaseConnection

class DemandValidator:
    def __init__(self):
        self.database = DatabaseConnection()


    def validate(self):
        engine = self.database.get_engine()

        demand_df = pd.read_sql(
            "select * from raw.demand",
            con=engine
        )

        print(
            f"rows loaded for validation: {len(demand_df)}"
        )

        required_columns = [
            "station_id",
            "station_name",
            "zone",
            "latitude",
            "longitude",
            "timestamp",
            "passenger_count"
        ]

        missing_columns = set(required_columns) - set(
            demand_df.columns
        )

        if missing_columns:
            raise ValueError(
                f"missing required columns:{missing_columns}"
            )

        missing_values = demand_df[
            required_columns
        ].isnull().sum()

        print('Missing values:')
        print(missing_values)

        if missing_values.any():
            raise ValueError(
                'Missing values found in required columns'
            )

        duplicate_count = demand_df.duplicated(
            subset = ['station_id', 'timestamp']
        ).sum()

        print(
            f"Duplicate station-timestamp rows: {duplicate_count}"
        )

        if duplicate_count > 0:
            raise ValueError(
                'duplicatet records found'
            )

        if (
            demand_df['passenger_count'] < 0
        ).any():
            raise ValueError(
                'negative passengers found'
            )

        station_count = demand_df[
            'station_id'
        ].nunique()

        print(
            f"Unique stations: {station_count}"
        )

        if station_count != 50:
            raise ValueError(
                f"Expected 50 stations, found {station_count}"
            )

        if (
            ~demand_df["latitude"].between(-90, 90)
        ).any():
            raise ValueError(
                "Invalid latitude values found."
            )

        if (
            ~demand_df["longitude"].between(-180, 180)
        ).any():
            raise ValueError(
                "Invalid longitude values found."
            )

        print(
            "Demand data validation completed successfully."
        )

        return True


if __name__ == "__main__":

    validator = DemandValidator()

    validator.validate()