import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from database.connection import DatabaseConnection

class BaseLineRidge:
    def __init__(self):
        self.database = DatabaseConnection()
        self.target_column = 'passenger_count'

    def load_data(self):
        engine = self.database.get_engine()

        train_df = pd.read_sql(
            'select * from model.train_data',
            con=engine
        )

        validation_df = pd.read_sql(
            'select * from model.validation_data',
            con=engine
        )

        print(
            f"Training data loaded: {train_df.shape}"
        )

        print(
            f"Validation data loaded: {validation_df.shape}"
        )

        return train_df, validation_df

    def seperate_features_and_target(
            self,
            train_df,
            validation_df
    ):
        x_train = train_df.drop(
            columns = [self.target_column]
        )

        y_train = train_df[self.target_column]

        x_validation = validation_df.drop(
            columns = [self.target_column]
        )

        y_validation = validation_df[
            self.target_column
        ]

        print("\nFeatures and target separated.")

        print(
            f"X_train shape: {x_train.shape}"
        )

        print(
            f"y_train shape: {y_train.shape}"
        )

        print(
            f"X_validation shape: "
            f"{x_validation.shape}"
        )

        print(
            f"y_validation shape: "
            f"{y_validation.shape}"
        )

        return (
            x_train,
            y_train,
            x_validation,
            y_validation
        )

    def train_ridge_model(
            self, 
            x_train,
            y_train
    ):
        ridge_model = Ridge(
            alpha=1.0
        )

        ridge_model.fit(
            x_train,
            y_train
        )

        print(
            "\nRidge Regression model "
            "trained successfully."
        )

        return ridge_model

    def predict(
            self,
            model,
            x_validation
    ):
        predictions = model.predict(
            x_validation
        )        

        print(
            "\nValidation predictions "
            "generated successfully."
        )

        return predictions

    def evaluate_model(
            self,
            y_validation,
            predictions
    ):
        mae = mean_absolute_error(
            y_validation,
            predictions
        )       

        mse = mean_squared_error(
            y_validation,
            predictions
        )

        rmse = mse ** 0.5

        score = r2_score(
            y_validation,
            predictions
        )

        print("\nModel Evaluation:")

        print(
            f"MAE: {mae:.4f}"
        )

        print(
            f"RMSE: {rmse:.4f}"
        )

        print(
            f"R² Score: {score:.4f}"
        )

        return {
            'MAE': mae,
            'MSE': mse,
            'R2_score': score
        }

if __name__ == "__main__":
    ridge_model = BaseLineRidge()

    train_df, validation_df = ridge_model.load_data()

    (
        x_train,
        y_train,
        x_validation,
        y_validation
    ) = ridge_model.seperate_features_and_target(
        train_df=train_df,
        validation_df=validation_df
    )

    model = (
        ridge_model.train_ridge_model(
            x_train=x_train,
            y_train=y_train
        )
    )

    predictions = ridge_model.predict(
        model=model,
        x_validation=x_validation
    )

    metrics = ridge_model.evaluate_model(
        y_validation=y_validation,
        predictions=predictions
    )


