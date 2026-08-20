"""
schemas.py

Pydantic models for ChennaiTransit AI — Passenger Demand Prediction API.

Field set here matches the original PredictionInput exactly (station_id,
zone, demand_profile, weather_condition, peak-hour flags, rolling_avg_4,
etc.) — nothing renamed, nothing dropped, nothing invented.
"""

from pydantic import BaseModel, Field, field_validator


class PredictionInput(BaseModel):
    """Input schema for a single passenger-demand prediction request."""

    # --- Identifiers ---
    station_id: str = Field(..., min_length=1, description="Unique identifier of the station.")
    zone: str = Field(..., min_length=1, description="Transit zone the station belongs to.")

    # --- Location ---
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees.")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees.")

    # --- Date/time features ---
    month: int = Field(..., ge=1, le=12, description="Calendar month (1-12).")
    day: int = Field(..., ge=1, le=31, description="Day of month (1-31).")
    hour: int = Field(..., ge=0, le=23, description="Hour of day, 24-hour clock (0-23).")
    minutes: int = Field(..., ge=0, le=59, description="Minute of the hour (0-59).")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week, Monday=0 ... Sunday=6.")
    is_weekend: bool = Field(..., description="Whether the prediction date falls on a weekend.")

    # --- Demand / peak-hour context ---
    demand_profile: str = Field(
        ..., min_length=1, description="Demand profile category for this station/time (e.g. 'high', 'low')."
    )
    is_morning_peak: bool = Field(..., description="Whether this falls in the morning peak window.")
    is_evening_peak: bool = Field(..., description="Whether this falls in the evening peak window.")
    is_peak_hour: bool = Field(..., description="Whether this is a peak hour overall.")

    # --- Weather features ---
    weather_condition: str = Field(
        ..., min_length=1, description="Weather condition category (e.g. 'clear', 'rain')."
    )
    temperature: float = Field(..., description="Ambient temperature in degrees Celsius.")
    humidity: float = Field(..., ge=0.0, le=100.0, description="Relative humidity as a percentage (0-100).")
    rainfall: float = Field(..., ge=0.0, description="Rainfall in millimeters. Cannot be negative.")

    # --- Passenger history features ---
    lag_1_passenger_count: float = Field(
        ..., ge=0.0, description="Passenger count from the previous time step. Cannot be negative."
    )
    rolling_avg_4: float = Field(
        ..., ge=0.0, description="4-period rolling average passenger count. Cannot be negative."
    )

    @field_validator("temperature")
    @classmethod
    def validate_temperature_range(cls, value: float) -> float:
        """
        Chennai's realistic ambient temperature range is roughly 15-48 C,
        with a safety margin for data noise. Values outside this band are
        almost always a bad sensor reading or a unit mix-up (e.g. Fahrenheit
        sent by mistake), so we reject them before they reach the model.
        """
        min_temp, max_temp = -10.0, 55.0
        if not (min_temp <= value <= max_temp):
            raise ValueError(
                f"temperature must be between {min_temp} and {max_temp} "
                f"degrees Celsius, got {value}"
            )
        return value

    model_config = {
        "json_schema_extra": {
            "example": {
                "station_id": "CHN-014",
                "zone": "Central",
                "latitude": 13.0827,
                "longitude": 80.2707,
                "month": 8,
                "day": 20,
                "hour": 9,
                "minutes": 15,
                "day_of_week": 3,
                "is_weekend": False,
                "demand_profile": "high",
                "is_morning_peak": True,
                "is_evening_peak": False,
                "is_peak_hour": True,
                "weather_condition": "clear",
                "temperature": 31.5,
                "humidity": 68.0,
                "rainfall": 0.0,
                "lag_1_passenger_count": 142.0,
                "rolling_avg_4": 140.5,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response schema returned by POST /predict."""

    predicted_passenger_count: float = Field(
        ..., description="Predicted passenger count, rounded to 2 decimal places."
    )
    model_name: str = Field(..., description="Name/identifier of the model used for prediction.")
    status: str = Field(..., description="Outcome status of the prediction, e.g. 'success'.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "predicted_passenger_count": 145.32,
                "model_name": "ChennaiTransit_PredictionPipeline_v1",
                "status": "success",
            }
        }
    }