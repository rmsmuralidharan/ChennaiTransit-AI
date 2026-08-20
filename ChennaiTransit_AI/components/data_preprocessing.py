import pandas as pd
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from database.connection import DatabaseConnection

class DataPreprocessing:
    def __init__(self):
        self.database = DatabaseConnection()

        self.feature_columns = [

            # Station
            "station_id",
            "zone",

            # Location
            "latitude",
            "longitude",

            # Time
            "month",
            "day",
            "hour",
            "minutes",
            "day_of_week",
            "is_weekend",

            # Demand pattern
            "demand_profile",
            "is_morning_peak",
            "is_evening_peak",
            "is_peak_hour",

            # Weather
            "weather_condition",
            "temperature",
            "humidity",
            "rainfall",

            # Historical demand
            "lag_1_passenger_count",
            "rolling_avg_4"
        ]

        self.target_column = "passenger_count"

        ## Next: categorize the 20 features

        self.categorical_columns = [
        "station_id",
        "zone",
        "demand_profile",
        "weather_condition"
        ]

        self.numerical_columns = [
        "latitude",
        "longitude",
        "month",
        "day",
        "hour",
        "minutes",
        "day_of_week",
        "temperature",
        "humidity",
        "rainfall",
        "lag_1_passenger_count",
        "rolling_avg_4"
        ]

        self.boolean_columns = [
        "is_weekend",
        "is_morning_peak",
        "is_evening_peak",
        "is_peak_hour"
    ]

        self.artifact_dir = Path('artifacts')

        self.artifact_dir.mkdir(
            parents = True,
            exist_ok = True
        )

    def load_data(self):

        engine = self.database.get_engine()

        query = """
        select * from final.demand_features
         """

        demand_df = pd.read_sql(
            query,
            con=engine
        )

        print(
            f"Rows loaded for preprocessing: "
            f"{len(demand_df)}"
        )

        return demand_df   

    def validate_features(self, demand_df):

        required_columns = (
            self.feature_columns + [self.target_column]
        )     

        missing_columns = (
            set(required_columns) - set(demand_df.columns)
        )

        if missing_columns:
            raise ValueError(
                f"missing required columns: "
                f"{missing_columns}"
            )

    def split_data(Self, demand_df):

        train_list = []
        validation_list = []
        test_list = []

        for station_id, station_df in demand_df.groupby(
            'station_id',
            sort = False
        ):
            

            total_rows = len(station_df)

            train_end = int(
                total_rows * 0.70
            )

            validation_end = int(
                total_rows * 0.85
            )

            train_list.append(station_df.iloc[
                :train_end
            ]
            )

            validation_list.append(station_df.iloc[
                train_end:validation_end
            ]
            )


            test_list.append(station_df.iloc[
                validation_end:
            ]
            )

            train_df = pd.concat(
                train_list,
                ignore_index=True
            )

            validation_df = pd.concat(
                validation_list,
                ignore_index=True
            )

            test_df = pd.concat(
                test_list,
                ignore_index=True
            )

        print("\nData split completed.")

        print(
            f"Training rows: {len(train_df)}"
        )

        print(
            f"Validation rows: {len(validation_df)}"
        )

        print(
            f"Test rows: {len(test_df)}"
        )

        print(
            f"Unique stations in training: "
            f"{train_df['station_id'].nunique()}"
        )

        print(
            f"Unique stations in validation: "
            f"{validation_df['station_id'].nunique()}"
        )

        print(
            f"Unique stations in test: "
            f"{test_df['station_id'].nunique()}"
        )

        return (
            train_df,
            validation_df,
            test_df
        )    

    def seperate_features_and_target(
            self,
            train_df,
            validation_df,
            test_df
    ):
        x_train = train_df[
            self.feature_columns
        ]

        y_train = train_df[
            self.target_column
        ]

        x_validation = validation_df[
            self.feature_columns
        ]

        y_validation = validation_df[
            self.target_column
        ]

        x_test = test_df[
            self.feature_columns
        ]

        y_test = test_df[
            self.target_column
        ]


        return (
            x_train,
            x_validation,
            x_test,
            y_train,
            y_validation,
            y_test
        ) 

    def create_preprocessor(self):

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    'categorical',
                    OneHotEncoder(
                        handle_unknown='ignore'
                    ),
                    self.categorical_columns
                ),
                (
                    'numerical',
                    StandardScaler(),
                    self.numerical_columns
                ),

                (
                    'boolean',
                    'passthrough',
                    self.boolean_columns
                )
            ]
        )

        return preprocessor

    def process_data(
            self,
            x_train,
            x_validation,
            x_test
    ):

        Preprocessor = self.create_preprocessor()

        x_train_processed = Preprocessor.fit_transform(
            x_train
        )

        x_validation_processed = Preprocessor.transform(
            x_validation
        )

        x_test_processed = Preprocessor.transform(
            x_test
        )

        return (
            x_train_processed,
            x_validation_processed,
            x_test_processed,
            Preprocessor
        )

    def save_preprocessor(
            self,
            preprocessor
    ):
        preprocessor_path = (
            self.artifact_dir / "preprocessor.joblib"
        )

        joblib.dump(
            preprocessor,
            preprocessor_path
        )

        print(
            f'\n preprocessor saved successfully at:'
        )

        print(
            preprocessor_path
        )

    def get_processed_feature_names(
            self,
            preprocessor
    ):
        feature_names = preprocessor.get_feature_names_out()

        return feature_names


    def create_processed_dataframes(
            self,
            x_train_processed,
            x_validation_processed,
            x_test_processed,
            y_train,
            y_validation,
            y_test,
            feature_names
    ):

        train_processed_df = pd.DataFrame(
            x_train_processed.toarray(),
            columns=feature_names
        )

        validation_processed_df = pd.DataFrame(
            x_validation_processed.toarray(),
            columns=feature_names
        )

        test_processed_df = pd.DataFrame(
            x_test_processed.toarray(),
            columns= feature_names
        )

        train_processed_df[
            self.target_column
        ] = y_train.reset_index(drop = True)

        validation_processed_df[
            self.target_column
        ] = y_validation.reset_index(drop = True)

        test_processed_df[
            self.target_column
        ] = y_test.reset_index(drop = True)

        print("\nProcessed DataFrames created successfully.")

        print(
            f"Train shape: "
            f"{train_processed_df.shape}"
        )

        print(
            f"Validation shape: "
            f"{validation_processed_df.shape}"
        )

        print(
            f"Test shape: "
            f"{test_processed_df.shape}"
        )

        return (
            train_processed_df,
            validation_processed_df,
            test_processed_df
        )

    def load_processed_data(
            self,
            train_processed_df,
            validation_processed_df,
            test_processed_df
    ):

        engine = self.database.get_engine()

        train_processed_df.to_sql(
            name = 'train_data',
            con = engine,
            schema = 'model',
            if_exists = 'replace',
            index = False,
            chunksize = 1000,
            method = 'multi'
        )

        print(
            f"\nTrain data loaded: "
            f"{len(train_processed_df)} rows"
        )

        validation_processed_df.to_sql(
            name = 'validation_data',
            con = engine,
            schema = 'model',
            if_exists = 'replace',
            index = False,
            chunksize = 1000,
            method = 'multi'
        )

        print(
            f"\nValidation data loaded: "
            f"{len(validation_processed_df)} rows"
        )

        test_processed_df.to_sql(
            name = 'test_data',
            con = engine,
            schema = 'model',
            if_exists = 'replace',
            index = False,
            chunksize = 1000,
            method = 'multi'
        )

        print(
            f"\nTest data loaded: "
            f"{len(test_processed_df)} rows"
        )

        print(
            "\nAll processed datasets "
            "loaded successfully."
        )



    





if __name__ == "__main__":
    data_preprocessor = DataPreprocessing()

    loaded_data = data_preprocessor.load_data()

    train_df, validation_df, test_df = data_preprocessor.split_data(loaded_data)

    x_train, x_validation, x_test, y_train, y_validation, y_test =data_preprocessor.seperate_features_and_target(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df
    )

    (
        x_train_processed,
        x_validation_processed,
        x_test_processed,
        preprocessor
    ) = data_preprocessor.process_data(

        x_train=x_train,
        x_validation=x_validation,
        x_test=x_test
    )

    feature_names = data_preprocessor.get_processed_feature_names(
        preprocessor=preprocessor
    )

    (
        train_processed_df,
        validation_processed_df,
        test_processed_df
    ) = data_preprocessor.create_processed_dataframes(
        x_train_processed=x_train_processed,
        x_validation_processed=x_validation_processed,
        x_test_processed=x_test_processed,

        y_train=y_train,
        y_validation = y_validation,
        y_test=y_test,

        feature_names=feature_names
    )

    data_preprocessor.load_processed_data(
        train_processed_df=train_processed_df,
        validation_processed_df= validation_processed_df,
        test_processed_df=test_processed_df
    )



    