import pandas as pd

from database.connection import DatabaseConnection

class FeatureEngineering:

    def __init__(self):
        self.database = DatabaseConnection()

    def load_data(self):

        connection = self.database.get_engine()

        demand_df = pd.read_sql(
            "select * from transformed.demand order by station_id, timestamp",
            con=connection
        )

        print(
            f"Rows loaded for feature engineering: {len(demand_df)}"
        )

        return demand_df

    def create_time_features(self):

        # Morning peak: 7 AM to 10 AM
        demand_df["is_morning_peak"] = (
            demand_df["hour"].between(7, 9)
        ).astype(int)

        # Evening peak: 5 PM to 9 PM
        demand_df["is_evening_peak"] = (
            demand_df["hour"].between(17, 20)
        ).astype(int)

        # Overall peak hour
        demand_df["is_peak_hour"] = (
            (
                demand_df["is_morning_peak"] == 1
            )
            |
            (
                demand_df["is_evening_peak"] == 1
            )
        ).astype(int)

        return demand_df

    def create_lag_features(self, demand_df):

        demand_df = demand_df.sort_values(
            by=["station_id", "timestamp"]
        )

        demand_df["lag_1_passenger_count"] = (
            demand_df
            .groupby("station_id")["passenger_count"]
            .shift(1)
        )

        return demand_df

    def handle_missing_lag_values(self, demand_df):

        demand_df["lag_1_passenger_count"] = (
            demand_df["lag_1_passenger_count"]
            .fillna(demand_df["passenger_count"])
        )

        return demand_df

    def create_rolling_features(self, demand_df):

        demand_df = demand_df.sort_values(
            by=["station_id", "timestamp"]
        )

        demand_df["rolling_avg_4"] = (
            demand_df
            .groupby("station_id")["passenger_count"]
            .transform(
                lambda x: x.rolling(
                    window=4,
                    min_periods=1
                ).mean()
            )
        )

        return demand_df

    def load_to_final(self, demand_df):

        connection = self.database.get_engine()

        from psycopg2.extras import execute_values

        raw_connection = connection.raw_connection()

        cursor = raw_connection.cursor()

        insert_query = """
        INSERT INTO final.demand_features (
            station_id,
            station_name,
            zone,
            latitude,
            longitude,
            year,
            month,
            day,
            hour,
            minutes,
            seconds,
            day_of_week,
            is_weekend,
            demand_profile,
            weather_condition,
            temperature,
            humidity,
            rainfall,
            passenger_count,
            is_morning_peak,
            is_evening_peak,
            is_peak_hour,
            lag_1_passenger_count,
            rolling_avg_4
        )
        VALUES %s
        """

        data = [
            tuple(row)
            for row in demand_df[
                [
                    "station_id",
                    "station_name",
                    "zone",
                    "latitude",
                    "longitude",
                    "year",
                    "month",
                    "day",
                    "hour",
                    "minutes",
                    "seconds",
                    "day_of_week",
                    "is_weekend",
                    "demand_profile",
                    "weather_condition",
                    "temperature",
                    "humidity",
                    "rainfall",
                    "passenger_count",
                    "is_morning_peak",
                    "is_evening_peak",
                    "is_peak_hour",
                    "lag_1_passenger_count",
                    "rolling_avg_4"
                ]
            ].itertuples(
                index=False,
                name=None
            )
        ]

        execute_values(
            cursor,
            insert_query,
            data,
            page_size=10000
        )

        raw_connection.commit()

        cursor.close()

        raw_connection.close()

        print(
            f"{len(demand_df)} rows loaded into final.demand_features"
        )




if __name__ == "__main__":
    feature_engineer = FeatureEngineering()

    demand_df = feature_engineer.load_data()

    demand_df = feature_engineer.create_time_features()

    demand_df = feature_engineer.create_lag_features(
        demand_df
    )

    demand_df = feature_engineer.handle_missing_lag_values(
        demand_df
    )


    demand_df = feature_engineer.create_rolling_features(
        demand_df
    )

    feature_engineer.load_to_final(
        demand_df
    )