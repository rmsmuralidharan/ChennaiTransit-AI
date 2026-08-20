import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D
)

from tensorflow.keras.callbacks import EarlyStopping

from database.connection import DatabaseConnection


class TransformerModel:
    """
    Transformer model for Chennai transit passenger demand prediction.
    """

    def __init__(self):

        self.db_connection = DatabaseConnection()

        self.target_column = "passenger_count"

        self.sequence_length = 24

        self.identifier_columns = [
            "station_id",
            "station_name",
            "passenger_count"
        ]


    def load_data(self):
        """
        Load the transformed final dataset.

        IMPORTANT:
        This table must contain the original station_id
        and time columns.
        """

        engine = self.db_connection.get_engine()

        demand_df = pd.read_sql(
            """
            SELECT *
            FROM final.demand_features
            """,
            con=engine
        )

        print(
            f"Rows loaded: {len(demand_df)}"
        )

        return demand_df


    def sort_data(
        self,
        demand_df
    ):
        """
        Sort each station chronologically.
        """

        required_columns = [
            "station_id",
            "year",
            "month",
            "day",
            "hour",
            "minutes"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in demand_df.columns
        ]

        if missing_columns:

            raise ValueError(
                f"Missing required columns: "
                f"{missing_columns}"
            )

        demand_df = demand_df.sort_values(
            by=[
                "station_id",
                "year",
                "month",
                "day",
                "hour",
                "minutes"
            ]
        ).reset_index(
            drop=True
        )

        print(
            "\nData sorted by station and time."
        )

        return demand_df


    def split_data(
        self,
        demand_df
    ):
        """
        Split every station chronologically:

        Train      = 70%
        Validation = 15%
        Test       = 15%
        """

        train_list = []

        validation_list = []

        test_list = []

        for station_id, station_df in demand_df.groupby(
            "station_id",
            sort=False
        ):

            total_rows = len(station_df)

            train_end = int(
                total_rows * 0.70
            )

            validation_end = int(
                total_rows * 0.85
            )

            train_list.append(
                station_df.iloc[
                    :train_end
                ]
            )

            validation_list.append(
                station_df.iloc[
                    train_end:validation_end
                ]
            )

            test_list.append(
                station_df.iloc[
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

        print(
            "\nData split completed."
        )

        print(
            f"Training rows: "
            f"{len(train_df)}"
        )

        print(
            f"Validation rows: "
            f"{len(validation_df)}"
        )

        print(
            f"Test rows: "
            f"{len(test_df)}"
        )

        return (
            train_df,
            validation_df,
            test_df
        )


    def create_preprocessor(
        self,
        train_df
    ):
        """
        Create and fit the preprocessing pipeline
        using ONLY training data.
        """

        categorical_columns = [
            "zone",
            "demand_profile",
            "weather_condition"
        ]

        numerical_columns = [
            "latitude",
            "longitude",
            "year",
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

        boolean_columns = [
            "is_weekend",
            "is_morning_peak",
            "is_evening_peak",
            "is_peak_hour"
        ]

        all_feature_columns = (
            categorical_columns
            + numerical_columns
            + boolean_columns
        )

        missing_columns = [
            column
            for column in all_feature_columns
            if column not in train_df.columns
        ]

        if missing_columns:

            raise ValueError(
                f"Missing feature columns: "
                f"{missing_columns}"
            )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    ),
                    categorical_columns
                ),

                (
                    "numerical",
                    StandardScaler(),
                    numerical_columns
                ),

                (
                    "boolean",
                    "passthrough",
                    boolean_columns
                )
            ]
        )

        X_train = train_df[
            all_feature_columns
        ]

        preprocessor.fit(
            X_train
        )

        print(
            "\nPreprocessor fitted successfully."
        )

        print(
            f"Number of input features: "
            f"{len(all_feature_columns)}"
        )

        return (
            preprocessor,
            all_feature_columns
        )


    def create_sequences(
        self,
        X,
        y
    ):
        """
        Create fixed-length sequences.

        Example:

        X[t-24:t] -> y[t]
        """

        X_sequences = []

        y_sequences = []

        for i in range(
            self.sequence_length,
            len(X)
        ):

            X_sequences.append(
                X[
                    i - self.sequence_length:i
                ]
            )

            y_sequences.append(
                y[i]
            )

        return (
            np.array(
                X_sequences,
                dtype=np.float32
            ),
            np.array(
                y_sequences,
                dtype=np.float32
            )
        )


    def create_station_sequences(
        self,
        train_df,
        validation_df,
        test_df,
        preprocessor,
        feature_columns
    ):
        """
        Transform and create sequences separately
        for every station.

        This prevents station histories from mixing.
        """

        X_train_sequences = []

        y_train_sequences = []

        X_validation_sequences = []

        y_validation_sequences = []

        X_test_sequences = []

        y_test_sequences = []

        station_ids = train_df[
            "station_id"
        ].unique()

        for station_id in station_ids:

            print(
                f"Processing station: "
                f"{station_id}"
            )

            train_station = train_df[
                train_df["station_id"] == station_id
            ]

            validation_station = validation_df[
                validation_df["station_id"] == station_id
            ]

            test_station = test_df[
                test_df["station_id"] == station_id
            ]

            # -------------------------
            # Transform features
            # -------------------------

            train_X = preprocessor.transform(
                train_station[
                    feature_columns
                ]
            )

            validation_X = preprocessor.transform(
                validation_station[
                    feature_columns
                ]
            )

            test_X = preprocessor.transform(
                test_station[
                    feature_columns
                ]
            )

            # Convert to float32
            # to reduce memory usage.

            train_X = np.asarray(
                train_X,
                dtype=np.float32
            )

            validation_X = np.asarray(
                validation_X,
                dtype=np.float32
            )

            test_X = np.asarray(
                test_X,
                dtype=np.float32
            )

            # -------------------------
            # Target values
            # -------------------------

            train_y = train_station[
                self.target_column
            ].values

            validation_y = validation_station[
                self.target_column
            ].values

            test_y = test_station[
                self.target_column
            ].values

            # -------------------------
            # Create sequences
            # -------------------------

            (
                train_sequence_X,
                train_sequence_y
            ) = self.create_sequences(
                train_X,
                train_y
            )

            (
                validation_sequence_X,
                validation_sequence_y
            ) = self.create_sequences(
                validation_X,
                validation_y
            )

            (
                test_sequence_X,
                test_sequence_y
            ) = self.create_sequences(
                test_X,
                test_y
            )

            X_train_sequences.append(
                train_sequence_X
            )

            y_train_sequences.append(
                train_sequence_y
            )

            X_validation_sequences.append(
                validation_sequence_X
            )

            y_validation_sequences.append(
                validation_sequence_y
            )

            X_test_sequences.append(
                test_sequence_X
            )

            y_test_sequences.append(
                test_sequence_y
            )

        # -------------------------
        # Combine all stations
        # -------------------------

        X_train = np.concatenate(
            X_train_sequences,
            axis=0
        )

        y_train = np.concatenate(
            y_train_sequences,
            axis=0
        )

        X_validation = np.concatenate(
            X_validation_sequences,
            axis=0
        )

        y_validation = np.concatenate(
            y_validation_sequences,
            axis=0
        )

        X_test = np.concatenate(
            X_test_sequences,
            axis=0
        )

        y_test = np.concatenate(
            y_test_sequences,
            axis=0
        )

        print(
            "\nSequences created successfully."
        )

        print(
            f"X_train shape: "
            f"{X_train.shape}"
        )

        print(
            f"y_train shape: "
            f"{y_train.shape}"
        )

        print(
            f"X_validation shape: "
            f"{X_validation.shape}"
        )

        print(
            f"y_validation shape: "
            f"{y_validation.shape}"
        )

        print(
            f"X_test shape: "
            f"{X_test.shape}"
        )

        print(
            f"y_test shape: "
            f"{y_test.shape}"
        )

        return (
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_test,
            y_test
        )


    def build_transformer(
        self,
        input_shape
    ):
        """
        Build Transformer regression model.
        """

        inputs = Input(
            shape=input_shape
        )

        # -------------------------
        # Multi-head attention
        # -------------------------

        attention_output = MultiHeadAttention(
            num_heads=4,
            key_dim=16
        )(
            inputs,
            inputs
        )

        attention_output = Dropout(
            0.1
        )(
            attention_output
        )

        # Residual connection

        x = LayerNormalization()(
            inputs + attention_output
        )

        # -------------------------
        # Feed-forward network
        # -------------------------

        feed_forward = Dense(
            64,
            activation="relu"
        )(
            x
        )

        feed_forward = Dense(
            input_shape[-1]
        )(
            feed_forward
        )

        feed_forward = Dropout(
            0.1
        )(
            feed_forward
        )

        # Residual connection

        x = LayerNormalization()(
            x + feed_forward
        )

        # -------------------------
        # Sequence aggregation
        # -------------------------

        x = GlobalAveragePooling1D()(
            x
        )

        x = Dense(
            64,
            activation="relu"
        )(
            x
        )

        x = Dropout(
            0.1
        )(
            x
        )

        outputs = Dense(
            1
        )(
            x
        )

        model = Model(
            inputs=inputs,
            outputs=outputs
        )

        model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mae"]
        )

        print(
            "\nTransformer model created successfully."
        )

        return model


    def train_model(
        self,
        model,
        X_train,
        y_train,
        X_validation,
        y_validation
    ):
        """
        Train Transformer model.
        """

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(
                X_validation,
                y_validation
            ),
            epochs=30,
            batch_size=256,
            callbacks=[
                early_stopping
            ],
            verbose=1
        )

        print(
            "\nTransformer training completed."
        )

        return (
            model,
            history
        )


    def evaluate_model(
        self,
        model,
        X_data,
        y_true,
        dataset_name
    ):
        """
        Evaluate model.
        """

        predictions = model.predict(
            X_data,
            batch_size=512
        )

        predictions = predictions.flatten()

        mae = mean_absolute_error(
            y_true,
            predictions
        )

        mse = mean_squared_error(
            y_true,
            predictions
        )

        rmse = np.sqrt(
            mse
        )

        r2 = r2_score(
            y_true,
            predictions
        )

        print(
            f"\n{dataset_name} Evaluation"
        )

        print(
            f"MAE: {mae:.4f}"
        )

        print(
            f"RMSE: {rmse:.4f}"
        )

        print(
            f"R² Score: {r2:.4f}"
        )

        return {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }


if __name__ == "__main__":

    transformer = TransformerModel()

    # =================================
    # 1. LOAD DATA
    # =================================

    demand_df = transformer.load_data()


    # =================================
    # 2. SORT DATA
    # =================================

    demand_df = transformer.sort_data(
        demand_df
    )


    # =================================
    # 3. SPLIT DATA
    # =================================

    (
        train_df,
        validation_df,
        test_df
    ) = transformer.split_data(
        demand_df
    )


    # =================================
    # 4. CREATE PREPROCESSOR
    # =================================

    (
        preprocessor,
        feature_columns
    ) = transformer.create_preprocessor(
        train_df
    )


    # =================================
    # 5. CREATE SEQUENCES
    # =================================

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test
    ) = transformer.create_station_sequences(
        train_df,
        validation_df,
        test_df,
        preprocessor,
        feature_columns
    )


    # =================================
    # 6. BUILD TRANSFORMER
    # =================================

    model = transformer.build_transformer(
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    )


    print("\nModel Summary:")

    model.summary()


    # =================================
    # 7. TRAIN
    # =================================

    (
        model,
        history
    ) = transformer.train_model(
        model,
        X_train,
        y_train,
        X_validation,
        y_validation
    )


    # =================================
    # 8. VALIDATION EVALUATION
    # =================================

    validation_metrics = transformer.evaluate_model(
        model,
        X_validation,
        y_validation,
        "Validation"
    )


    # =================================
    # 9. TEST EVALUATION
    # =================================

    test_metrics = transformer.evaluate_model(
        model,
        X_test,
        y_test,
        "Test"
    )


    print("\nFinal Results")

    print(
        "\nValidation Metrics:"
    )

    print(
        validation_metrics
    )

    print(
        "\nTest Metrics:"
    )

    print(
        test_metrics
    )