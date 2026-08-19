import numpy as np
import pandas as pd

from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]

GENERATED_FILE_PATH = PROJECT_ROOT / 'data' / 'generated'

STATION_CSV_FILE = GENERATED_FILE_PATH / 'stations.csv'

station_df = pd.read_csv(STATION_CSV_FILE)

START_DATE = '2026-01-01'

END_DATE = '2026-12-31 23:59:59'

timestamps = pd.date_range(
    start=START_DATE,
    end=END_DATE,
    freq='15min',
    inclusive='left'
)



demand_df = station_df.merge(
    pd.DataFrame({'timestamp': timestamps}),
    how='cross'
)

station_counts = demand_df.groupby('station_id').size()

demand_df['hour'] = demand_df['timestamp'].dt.hour

demand_df['day_of_week'] = demand_df['timestamp'].dt.day_of_week

demand_df['is_weekend'] = demand_df['day_of_week'] >= 5


station_profiles = {
    "ST001": "transport_hub",      # Guindy
    "ST002": "it_hub",             # Taramani
    "ST003": "it_hub",             # Perungudi
    "ST004": "it_hub",             # Thoraipakkam
    "ST005": "it_hub",             # Sholinganallur
    "ST006": "it_hub",             # Navalur
    "ST007": "it_hub",             # Siruseri
    "ST008": "commercial",         # Kelambakkam
    "ST009": "residential",        # Velachery
    "ST010": "commercial",         # Adyar

    "ST011": "industrial",         # Ambattur
    "ST012": "industrial",         # Padi
    "ST013": "industrial",         # Avadi
    "ST014": "transport_hub",      # Poonamallee
    "ST015": "it_hub",             # Porur
    "ST016": "it_hub",             # Manapakkam
    "ST017": "industrial",         # Sriperumbudur
    "ST018": "industrial",         # Oragadam
    "ST019": "industrial",         # Irungattukottai
    "ST020": "industrial",         # Maraimalai Nagar

    "ST021": "transport_hub",      # Tambaram
    "ST022": "residential",        # Chromepet
    "ST023": "transport_hub",      # Pallavaram
    "ST024": "residential",        # Guduvanchery
    "ST025": "residential",        # Vandalur
    "ST026": "industrial",         # Singaperumal Koil
    "ST027": "transport_hub",      # Chengalpattu

    "ST028": "industrial",         # Manali
    "ST029": "industrial",         # Ennore
    "ST030": "transport_hub",      # Madhavaram
    "ST031": "industrial",         # Tiruvottiyur
    "ST032": "industrial",         # Tondiarpet
    "ST033": "transport_hub",      # Royapuram
    "ST034": "transport_hub",      # Perambur
    "ST035": "industrial",         # Vyasarpadi
    "ST036": "residential",        # Villivakkam
    "ST037": "industrial",         # Thirumudivakkam

    "ST038": "transport_hub",      # Koyambedu
    "ST039": "residential",        # Anna Nagar
    "ST040": "commercial",         # Nungambakkam
    "ST041": "commercial",         # T. Nagar
    "ST042": "transport_hub",      # Saidapet
    "ST043": "commercial",         # Nandanam
    "ST044": "transport_hub",      # Alandur
    "ST045": "transport_hub",      # St. Thomas Mount

    "ST046": "residential",        # Medavakkam
    "ST047": "residential",        # Pallikaranai
    "ST048": "residential",        # Madipakkam
    "ST049": "industrial",         # Ponneri
    "ST050": "industrial"          # Gummidipoondi
}

if len(station_profiles) != len(station_df):
    raise ValueError(
        'every section must have exactly one deamnd profile'
    )

demand_df['demand_profile'] = demand_df['station_id'].map(
    station_profiles
)

if demand_df['demand_profile'].isna().any():
    raise ValueError(
        'one or more stations are missing'
    )



base_demand_map = {
    "transport_hub": 220,
    "it_hub": 170,
    "commercial": 150,
    "residential": 90,
    "industrial": 60
}

demand_df['base_demand'] = demand_df['demand_profile'].map(
    base_demand_map
)

if demand_df['base_demand'].isna().any():
    raise ValueError(
        'no baseline model'
    )


def get_time_multiplier(hour):
    if 0 <= hour < 6:
        return 0.25
    elif 7 <= hour < 10:
        return 1.80

    elif 10 <= hour < 17:
        return 0.90

    elif 17 <= hour < 20:
        return 1.70

    elif 20 <= hour < 23:
        return 0.70
    else:
        return 0.50

demand_df['time_multiplier'] = demand_df['hour'].apply(
    get_time_multiplier
)

demand_df['time_adjusted_demand'] = (
    demand_df['base_demand']
    * demand_df['time_multiplier']
)


def get_day_multiplier(is_weekend):
    if is_weekend:
        return 0.80
    else:
        return 1.00

demand_df['day_multiplier'] = demand_df['is_weekend'].apply(
    get_day_multiplier
)

demand_df['day_adjusted_demand'] = (
    demand_df['time_adjusted_demand']
    * demand_df['day_multiplier']
)


profile_multiplier_map = {
    "transport_hub": 1.10,
    "it_hub": 1.15,
    "commercial": 1.05,
    "residential": 0.95,
    "industrial": 0.90
}

demand_df['profile_multiplier'] = demand_df['demand_profile'].map(
    profile_multiplier_map
)

if demand_df['profile_multiplier'].isna().any():
    raise ValueError(
        'no profile multiplier'
    )

demand_df['profile_adjusted_demand'] = (
    demand_df['day_adjusted_demand']
    * demand_df['profile_multiplier']
)

weather_conditions = [
    "clear",
    "cloudy",
    "light_rain",
    "heavy_rain"
]

weather_probabilities = [
    0.60,
    0.20,
    0.15,
    0.05
]

demand_df['weather_condition'] = np.random.choice(
    weather_conditions,
    size=len(demand_df),
    p=weather_probabilities
)


def generate_temperature(weather_condition):
    if weather_condition == 'clear':
        return np.random.uniform(27, 34)

    elif weather_condition == 'cloudy':
        return np.random.uniform(26,32)

    elif weather_condition == 'light_rain':
        return np.random.uniform(24,30)

    else:
        return np.random.uniform(22, 28)

demand_df['temperature'] = demand_df['weather_condition'].apply(
    generate_temperature
)


def generate_humidity(weather_condition):
    if weather_condition == 'clear':
        return np.random.uniform(45, 70)

    elif weather_condition == 'cloudy':
        return np.random.uniform(55, 80)

    elif weather_condition == 'light_rain':
        return np.random.uniform(70, 90)

    else:
        return np.random.uniform(80, 90)

demand_df['humidity'] = demand_df['weather_condition'].apply(
    generate_humidity
)


def generate_rainfall(weather_condition):
    if weather_condition == "clear":
        return 0.0

    elif weather_condition == "cloudy":
        return np.random.uniform(0.0, 0.5)

    elif weather_condition == "light_rain":
        return np.random.uniform(0.5, 5.0)

    else:
        return np.random.uniform(5.0, 30.0)


demand_df["rainfall"] = demand_df["weather_condition"].apply(
    generate_rainfall
)



weather_multiplier_map = {
    "clear": 1.00,
    "cloudy": 0.95,
    "light_rain": 1.05,
    "heavy_rain": 1.15
}

demand_df["weather_multiplier"] = demand_df["weather_condition"].map(
    weather_multiplier_map
)

if demand_df["weather_multiplier"].isna().any():
    raise ValueError(
        "One or more weather conditions have no weather multiplier."
    )


demand_df["weather_adjusted_demand"] = (
    demand_df["profile_adjusted_demand"]
    * demand_df["weather_multiplier"]
)


def generate_passenger_count(expected_demand):

    variation = np.random.normal(
        0,
        expected_demand * 0.05
    )

    passenger_count = expected_demand + variation

    passenger_count = max(0, passenger_count)

    return int(round(passenger_count))

demand_df["passenger_count"] = demand_df["weather_adjusted_demand"].apply(
    generate_passenger_count
)

print(len(demand_df))

print(
    "Unique stations:",
    demand_df["station_id"].nunique()
)

print(
    demand_df.groupby("station_id").size().describe()
)

print(
    demand_df.isna().sum()
)

print(
    demand_df["passenger_count"].describe()
)

OUTPUT_CSV_FILE = GENERATED_FILE_PATH / 'demand.csv'

demand_df.to_csv(
    OUTPUT_CSV_FILE,
    index=False
)