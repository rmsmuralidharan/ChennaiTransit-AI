import joblib
import pandas as pd

class PredictionPipeline:

    def __init__(self):

        self.preprocessor = joblib.load(
            'artifacts/preprocessor.joblib'
        )

        self.model = joblib.load(
            'artifacts/xgboost_model.joblib'
        )

        print(
            "Prediction pipeline loaded successfully."
        )

    def predict(
            self,
            input_data
    ):

        # convert input to df

        input_df = pd.DataFrame(
            [input_data]
        )

        ## apply preprocessor

        processed_data = self.preprocessor.transform(
            input_df
        )

        ## predict passenger_count

        prediction = self.model.predict(
            processed_data
        )

        return prediction[0]

if __name__ == "__main__":

    pipeline = PredictionPipeline()

    sample_input = {

        "station_id": "ST001",
        "zone": "Central",

        "demand_profile": "commercial",

        "weather_condition": "clear",

        "latitude": 13.0827,

        "longitude": 80.2707,

        "month": 8,

        "day": 20,

        "hour": 9,

        "minutes": 0,

        "day_of_week": 3,

        "temperature": 30.0,

        "humidity": 70.0,

        "rainfall": 0.0,

        "lag_1_passenger_count": 120,

        "rolling_avg_4": 115,

        "is_weekend": False,

        "is_morning_peak": True,

        "is_evening_peak": False,

        "is_peak_hour": True
    }


    prediction = pipeline.predict(
        sample_input
    )

    print(
        f"\nPredicted passenger count: "
        f"{prediction:.2f}"
    )