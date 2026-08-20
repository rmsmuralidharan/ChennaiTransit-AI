import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[2]

GENERATED_DATA_DIR = PROJECT_ROOT / 'data' /'generated'

def generate_station_master():

    GENERATED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    ## stations

    stations = {
        "ST001": {
            "place_name": "Guindy",
            "zone": "South",
            "latitude": 13.0067,
            "longitude": 80.2206
        },

        "ST002": {
            "place_name": "Taramani",
            "zone": "South",
            "latitude": 12.9895,
            "longitude": 80.2410
        },

        "ST003": {
            "place_name": "Perungudi",
            "zone": "South",
            "latitude": 12.9591,
            "longitude": 80.2406
        },

        "ST004": {
            "place_name": "Thoraipakkam",
            "zone": "South",
            "latitude": 12.9377,
            "longitude": 80.2364
        },

        "ST005": {
            "place_name": "Sholinganallur",
            "zone": "South",
            "latitude": 12.9010,
            "longitude": 80.2279
        },

        "ST006": {
            "place_name": "Navalur",
            "zone": "South",
            "latitude": 12.8454,
            "longitude": 80.2263
        },

        "ST007": {
            "place_name": "Siruseri",
            "zone": "South",
            "latitude": 12.8352,
            "longitude": 80.2300
        },

        "ST008": {
            "place_name": "Kelambakkam",
            "zone": "South",
            "latitude": 12.7886,
            "longitude": 80.2209
        },

        "ST009": {
            "place_name": "Velachery",
            "zone": "South",
            "latitude": 12.9815,
            "longitude": 80.2180
        },

        "ST010": {
            "place_name": "Adyar",
            "zone": "South",
            "latitude": 13.0012,
            "longitude": 80.2565
        },

        "ST011": {
            "place_name": "Ambattur",
            "zone": "West",
            "latitude": 13.1143,
            "longitude": 80.1548
        },

        "ST012": {
            "place_name": "Padi",
            "zone": "West",
            "latitude": 13.0955,
            "longitude": 80.1845
        },

        "ST013": {
            "place_name": "Avadi",
            "zone": "West",
            "latitude": 13.1147,
            "longitude": 80.1098
        },

        "ST014": {
            "place_name": "Poonamallee",
            "zone": "West",
            "latitude": 13.0481,
            "longitude": 80.1113
        },

        "ST015": {
            "place_name": "Porur",
            "zone": "West",
            "latitude": 13.0358,
            "longitude": 80.1562
        },

        "ST016": {
            "place_name": "Manapakkam",
            "zone": "West",
            "latitude": 13.0144,
            "longitude": 80.1714
        },

        "ST017": {
            "place_name": "Sriperumbudur",
            "zone": "West",
            "latitude": 12.9676,
            "longitude": 79.9418
        },

        "ST018": {
            "place_name": "Oragadam",
            "zone": "West",
            "latitude": 12.8496,
            "longitude": 80.0500
        },

        "ST019": {
            "place_name": "Irungattukottai",
            "zone": "West",
            "latitude": 13.0155,
            "longitude": 79.9560
        },

        "ST020": {
            "place_name": "Maraimalai Nagar",
            "zone": "South",
            "latitude": 12.7936,
            "longitude": 80.0255
        },

        "ST021": {
            "place_name": "Tambaram",
            "zone": "South",
            "latitude": 12.9249,
            "longitude": 80.1000
        },

        "ST022": {
            "place_name": "Chromepet",
            "zone": "South",
            "latitude": 12.9516,
            "longitude": 80.1462
        },

        "ST023": {
            "place_name": "Pallavaram",
            "zone": "South",
            "latitude": 12.9675,
            "longitude": 80.1491
        },

        "ST024": {
            "place_name": "Guduvanchery",
            "zone": "South",
            "latitude": 12.8452,
            "longitude": 80.0607
        },

        "ST025": {
            "place_name": "Vandalur",
            "zone": "South",
            "latitude": 12.8913,
            "longitude": 80.0809
        },

        "ST026": {
            "place_name": "Singaperumal Koil",
            "zone": "South",
            "latitude": 12.7590,
            "longitude": 80.0030
        },

        "ST027": {
            "place_name": "Chengalpattu",
            "zone": "South",
            "latitude": 12.6819,
            "longitude": 79.9888
        },

        "ST028": {
            "place_name": "Manali",
            "zone": "North",
            "latitude": 13.1667,
            "longitude": 80.2583
        },

        "ST029": {
            "place_name": "Ennore",
            "zone": "North",
            "latitude": 13.2146,
            "longitude": 80.3203
        },

        "ST030": {
            "place_name": "Madhavaram",
            "zone": "North",
            "latitude": 13.1482,
            "longitude": 80.2314
        },

        "ST031": {
            "place_name": "Tiruvottiyur",
            "zone": "North",
            "latitude": 13.1589,
            "longitude": 80.3019
        },

        "ST032": {
            "place_name": "Tondiarpet",
            "zone": "North",
            "latitude": 13.1274,
            "longitude": 80.2892
        },

        "ST033": {
            "place_name": "Royapuram",
            "zone": "North",
            "latitude": 13.1131,
            "longitude": 80.2946
        },

        "ST034": {
            "place_name": "Perambur",
            "zone": "North",
            "latitude": 13.1166,
            "longitude": 80.2326
        },

        "ST035": {
            "place_name": "Vyasarpadi",
            "zone": "North",
            "latitude": 13.1097,
            "longitude": 80.2565
        },

        "ST036": {
            "place_name": "Villivakkam",
            "zone": "North",
            "latitude": 13.1094,
            "longitude": 80.2057
        },

        "ST037": {
            "place_name": "Thirumudivakkam",
            "zone": "West",
            "latitude": 12.9638,
            "longitude": 80.0860
        },

        "ST038": {
            "place_name": "Koyambedu",
            "zone": "Central",
            "latitude": 13.0732,
            "longitude": 80.1949
        },

        "ST039": {
            "place_name": "Anna Nagar",
            "zone": "Central",
            "latitude": 13.0850,
            "longitude": 80.2101
        },

        "ST040": {
            "place_name": "Nungambakkam",
            "zone": "Central",
            "latitude": 13.0569,
            "longitude": 80.2425
        },

        "ST041": {
            "place_name": "T. Nagar",
            "zone": "Central",
            "latitude": 13.0418,
            "longitude": 80.2341
        },

        "ST042": {
            "place_name": "Saidapet",
            "zone": "South",
            "latitude": 13.0213,
            "longitude": 80.2231
        },

        "ST043": {
            "place_name": "Nandanam",
            "zone": "South",
            "latitude": 13.0324,
            "longitude": 80.2394
        },

        "ST044": {
            "place_name": "Alandur",
            "zone": "South",
            "latitude": 13.0048,
            "longitude": 80.2014
        },

        "ST045": {
            "place_name": "St. Thomas Mount",
            "zone": "South",
            "latitude": 12.9951,
            "longitude": 80.1987
        },

        "ST046": {
            "place_name": "Medavakkam",
            "zone": "South",
            "latitude": 12.9228,
            "longitude": 80.1927
        },

        "ST047": {
            "place_name": "Pallikaranai",
            "zone": "South",
            "latitude": 12.9387,
            "longitude": 80.2120
        },

        "ST048": {
            "place_name": "Madipakkam",
            "zone": "South",
            "latitude": 12.9622,
            "longitude": 80.1986
        },

        "ST049": {
            "place_name": "Ponneri",
            "zone": "North",
            "latitude": 13.3387,
            "longitude": 80.1948
        },

        "ST050": {
            "place_name": "Gummidipoondi",
            "zone": "North",
            "latitude": 13.4070,
            "longitude": 80.1084
        }
    }

    station_df = pd.DataFrame.from_dict(
        stations,
        orient='index'
    )

    station_df.reset_index(
        inplace=True
    )

    station_df.rename(
        columns={
            'index': 'station_id'
        },

        inplace=True
    )

    station_df.rename(
        columns={
            'place_name': 'station_name'
        },
        inplace=True
    )

    if len(station_df) != 50:
        raise ValueError(
            'Station master must contain exactly 50 stations'
        )

    if station_df['station_id'].nunique() != 50:
        raise ValueError(
            "Station ids must be unique"
        )

    output_file = GENERATED_DATA_DIR / 'stations.csv'

    station_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"Station master generated successfully: {output_file}"
    )


if __name__ == "__main__":
    generate_station_master()