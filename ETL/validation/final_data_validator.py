import pandas as pd

from database.connection import DatabaseConnection


class FinalDataValidator:

    def __init__(self):

        self.db_connection = DatabaseConnection()


    def validate(self):

        engine = self.db_connection.get_engine()

        query = """
        SELECT *
        FROM final.demand_features
        """

        demand_df = pd.read_sql(
            query,
            engine
        )

        print(
            f"Rows loaded for validation: {len(demand_df)}"
        )

        print("\nMissing values:")

        print(
            demand_df.isnull().sum()
        )

        duplicate_count = demand_df.duplicated(
            subset=[
                "station_id",
                "timestamp"
            ]
        ).sum()

        print(
            f"\nDuplicate station-timestamp rows: {duplicate_count}"
        )

        print(
            f"\nUnique stations: "
            f"{demand_df['station_id'].nunique()}"
        )

        print("\nPassenger count statistics:")

        print(
            demand_df["passenger_count"].describe()
        )

        print(
            "\nFinal data validation completed successfully."
        )


if __name__ == "__main__":

    validator = FinalDataValidator()

    validator.validate()