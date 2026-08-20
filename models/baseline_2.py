from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from models.baseline_1 import BaseLineRidge

baseline_1 = BaseLineRidge()

class ModelTrainer:

    def __init__(self):
        pass

    def train_xgboost_model( 
            x_train,
            y_train
    ):
        xgb_model = XGBRegressor(
            objective = 'reg:squarederror',
            n_estimators = 500,
            learning_rate = 0.05,
            max_depth = 8,
            subsample = 0.8,
            colsample_bytree = 0.8,
            random_state = 42,
            n_jobs = -1
        )

        xgb_model.fit(
            x_train,
            y_train
        )


        print(
            "\nXGBoost model trained successfully."
        )

        return xgb_model


if __name__ == "__main__":

    train_df, validation_df = baseline_1.load_data()

    x_train, y_train, x_validation, y_validation = (
        baseline_1.seperate_features_and_target(
            train_df=train_df,
            validation_df=validation_df
        )
    )


    xgb_model = (
        ModelTrainer.train_xgboost_model(
            x_train=x_train,
            y_train=y_train
        )
    )        

    predictions = baseline_1.predict(
        xgb_model,
        x_validation=x_validation
    )

    metrics = baseline_1.evaluate_model(
        y_validation=y_validation,
        predictions=predictions
    )



