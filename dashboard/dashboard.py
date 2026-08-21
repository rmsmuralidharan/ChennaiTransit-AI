"""
Chennai Transit AI — Passenger Demand Prediction Dashboard.

Streamlit frontend for the FastAPI /predict service.

Features:
- Station, date, and time selection
- Automatic feature derivation
- Advanced weather and demand options
- Passenger demand prediction
- Color-coded demand level
- Prediction history
- Demand analytics and visualization

The Streamlit dashboard communicates with FastAPI over HTTP
and never imports the ML model directly.

------------------------------------------------------------
CHANGELOG (patched version)
------------------------------------------------------------
1. Replaced deprecated `use_container_width` with `width=` on
   every widget/chart call (popover, buttons, dataframe,
   download button, line/bar charts). `use_container_width`
   is removed in current Streamlit releases.
2. `check_api_health()` is now wrapped in `st.cache_data(ttl=...)`
   so it no longer fires an HTTP request on every single rerun
   (i.e. every keystroke/widget interaction) — only once every
   10 seconds per api_url.
3. `history_df` is now built once, above both the "Operational
   Summary" and "Prediction History" sections, instead of being
   silently reused across two separate `if st.session_state.history:`
   blocks (previously fragile — a NameError waiting to happen if
   those blocks were ever refactored independently).
4. Removed the unused `render_html()` helper (dead code) — the
   `st.markdown` monkeypatch below already dedents/strips all
   `unsafe_allow_html=True` calls.
------------------------------------------------------------
"""

from datetime import date, datetime, time
from typing import Optional
from textwrap import dedent

import pandas as pd
import requests
import streamlit as st


# ============================================================
# SAFE HTML RENDERER (monkeypatch)
# ============================================================
# Streamlit can interpret indented multiline HTML as a Markdown
# code block. This wrapper removes common indentation before
# rendering any unsafe_allow_html=True markdown call.

# ============================================================
# SAFE HTML RENDERER
# ============================================================

_original_markdown = st.markdown


def _safe_markdown(body, *args, **kwargs):
    if kwargs.get("unsafe_allow_html") and isinstance(body, str):
        body = dedent(body).strip()

    return _original_markdown(body, *args, **kwargs)


st.markdown = _safe_markdown



# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Chennai Transit AI",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    /* ========================================================
       METRICS
       ======================================================== */

    div[data-testid="stMetric"] {
        background: linear-gradient(
            145deg,
            rgba(30, 41, 59, 0.55),
            rgba(15, 23, 42, 0.65)
        );

        border: 1px solid rgba(148, 163, 184, 0.16);

        border-radius: 14px;

        padding: 16px 18px;

        box-shadow:
            0 8px 24px rgba(0, 0, 0, 0.08);
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }

    /* ========================================================
       HEADER
       ======================================================== */

    .app-header {
        padding: 5px 0 22px 0;
    }

    .app-kicker {
    color: #64748b;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;

    line-height: 1.5;
    min-height: 20px;

    padding-top: 3px;
    padding-bottom: 3px;

    margin-bottom: 5px;

    overflow: visible;
}

    .app-title {
        color: #f8fafc;
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.15;
    }

    .app-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 7px;
    }

    /* ========================================================
       STATUS PILLS
       ======================================================== */

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 6px 13px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 650;
    }

    .status-online {
        background-color: rgba(22, 163, 74, 0.12);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.18);
    }

    .status-offline {
        background-color: rgba(220, 38, 38, 0.12);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.18);
    }

    /* ========================================================
       PROFESSIONAL CARDS
       ======================================================== */

    .professional-card {
        background: linear-gradient(
            145deg,
            rgba(30, 41, 59, 0.52),
            rgba(15, 23, 42, 0.72)
        );

        border: 1px solid rgba(148, 163, 184, 0.16);

        border-radius: 14px;

        padding: 18px;

        margin-bottom: 14px;

        min-height: 90px;

        box-shadow:
            0 8px 24px rgba(0, 0, 0, 0.08);
    }

    .card-kicker {
        color: #64748b;
        font-size: 0.67rem;
        font-weight: 750;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .card-main-value {
        color: #f8fafc;
        font-size: 1.15rem;
        font-weight: 750;
    }

    .card-muted {
        color: #94a3b8;
        font-size: 0.76rem;
        margin-top: 5px;
    }

    .card-divider {
        height: 1px;
        background: rgba(148, 163, 184, 0.12);
        margin: 13px 0;
    }

    .card-row {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        margin: 9px 0;
        color: #94a3b8;
        font-size: 0.78rem;
    }

    .card-row strong {
        color: #e2e8f0;
        font-weight: 600;
    }

    .status-large {
        color: #e2e8f0;
        font-size: 1rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .network-status {
        color: #4ade80;
        font-size: 0.95rem;
        font-weight: 700;
    }

    .status-dot-green {
        color: #22c55e;
    }

    .status-dot-orange {
        color: #fb923c;
    }

    .profile-value {
        color: #fb923c;
        font-size: 1.25rem;
        font-weight: 800;
    }

    .big-number {
        color: #f8fafc;
        font-size: 1.7rem;
        font-weight: 800;
    }

    /* ========================================================
       CENTER WORKSPACE
       ======================================================== */

    .workspace-header {
        padding: 2px 0 12px 0;
    }

    .workspace-kicker {
        color: #64748b;
        font-size: 0.7rem;
        font-weight: 750;
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }

    .workspace-title {
        color: #f8fafc;
        font-size: 1.65rem;
        font-weight: 800;
        margin-top: 4px;
    }

    .workspace-description {
        color: #94a3b8;
        font-size: 0.84rem;
        line-height: 1.5;
        margin-top: 5px;
    }

    /* ========================================================
       RESULT CARD
       ======================================================== */

    .result-card {
        text-align: center;

        padding: 34px 20px 30px 20px;

        border-radius: 18px;

        border: 1px solid rgba(148, 163, 184, 0.22);

        background:
            radial-gradient(
                circle at top center,
                rgba(59, 130, 246, 0.08),
                transparent 45%
            ),
            rgba(30, 41, 59, 0.42);

        margin-top: 8px;

        box-shadow:
            0 12px 35px rgba(0, 0, 0, 0.10);
    }

    .result-label {
        font-size: 0.72rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-weight: 700;
        margin-bottom: 9px;
    }

    .result-count {
        color: #f8fafc;
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 7px 0 18px 0;
    }

    .result-unit {
        font-size: 0.95rem;
        font-weight: 500;
        color: #94a3b8;
    }

    .result-demand {
        color: #e2e8f0;
        font-size: 0.9rem;
        font-weight: 600;
    }

    .result-level {
        display: inline-flex;
        align-items: center;
        padding: 7px 18px;
        margin-left: 8px;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 750;
    }

    .level-low {
        background-color: rgba(22, 163, 74, 0.13);
        color: #4ade80;
    }

    .level-medium {
        background-color: rgba(202, 138, 4, 0.13);
        color: #facc15;
    }

    .level-high {
        background-color: rgba(234, 88, 12, 0.13);
        color: #fb923c;
    }

    .level-critical {
        background-color: rgba(220, 38, 38, 0.13);
        color: #f87171;
    }

    /* ========================================================
       SECTION HEADERS
       ======================================================== */

    .section-title {
        color: #f8fafc;
        font-size: 1.35rem;
        font-weight: 750;
        margin-top: 10px;
        margin-bottom: 12px;
    }

    /* ========================================================
       FOOTER
       ======================================================== */

    .dashboard-footer {
        text-align: center;
        color: #64748b;
        font-size: 0.72rem;
        padding-top: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DEFAULT API URL
# ============================================================

DEFAULT_API_URL = "https://chennaitransit-ai.onrender.com"


# ============================================================
# STATION REFERENCE DATA
# ============================================================

STATIONS = {
    "Anna Nagar": {
        "id": "CHN-001",
        "zone": "West",
        "lat": 13.0850,
        "lon": 80.2101,
    },
    "Chennai Central": {
        "id": "CHN-002",
        "zone": "Central",
        "lat": 13.0827,
        "lon": 80.2757,
    },
    "Egmore": {
        "id": "CHN-003",
        "zone": "Central",
        "lat": 13.0732,
        "lon": 80.2609,
    },
    "Guindy": {
        "id": "CHN-004",
        "zone": "South",
        "lat": 13.0067,
        "lon": 80.2206,
    },
    "Tambaram": {
        "id": "CHN-005",
        "zone": "South",
        "lat": 12.9246,
        "lon": 80.1000,
    },
    "Velachery": {
        "id": "CHN-006",
        "zone": "South",
        "lat": 12.9791,
        "lon": 80.2183,
    },
    "T Nagar": {
        "id": "CHN-007",
        "zone": "Central",
        "lat": 13.0418,
        "lon": 80.2341,
    },
    "Adyar": {
        "id": "CHN-008",
        "zone": "East",
        "lat": 13.0012,
        "lon": 80.2565,
    },
    "Mylapore": {
        "id": "CHN-009",
        "zone": "East",
        "lat": 13.0339,
        "lon": 80.2698,
    },
    "Kodambakkam": {
        "id": "CHN-010",
        "zone": "West",
        "lat": 13.0512,
        "lon": 80.2231,
    },
    "Koyambedu": {
        "id": "CHN-011",
        "zone": "West",
        "lat": 13.0694,
        "lon": 80.1948,
    },
    "Vadapalani": {
        "id": "CHN-012",
        "zone": "West",
        "lat": 13.0503,
        "lon": 80.2126,
    },
    "Porur": {
        "id": "CHN-013",
        "zone": "West",
        "lat": 13.0374,
        "lon": 80.1575,
    },
    "Ambattur": {
        "id": "CHN-014",
        "zone": "West",
        "lat": 13.1143,
        "lon": 80.1548,
    },
    "Avadi": {
        "id": "CHN-015",
        "zone": "West",
        "lat": 13.1147,
        "lon": 80.0970,
    },
    "Perambur": {
        "id": "CHN-016",
        "zone": "North",
        "lat": 13.1140,
        "lon": 80.2340,
    },
    "Washermanpet": {
        "id": "CHN-017",
        "zone": "North",
        "lat": 13.1136,
        "lon": 80.2836,
    },
    "Royapettah": {
        "id": "CHN-018",
        "zone": "Central",
        "lat": 13.0553,
        "lon": 80.2646,
    },
    "Sholinganallur": {
        "id": "CHN-019",
        "zone": "East",
        "lat": 12.9010,
        "lon": 80.2279,
    },
    "Thiruvanmiyur": {
        "id": "CHN-020",
        "zone": "East",
        "lat": 12.9830,
        "lon": 80.2594,
    },
    "Pallavaram": {
        "id": "CHN-021",
        "zone": "South",
        "lat": 12.9675,
        "lon": 80.1491,
    },
    "Chromepet": {
        "id": "CHN-022",
        "zone": "South",
        "lat": 12.9516,
        "lon": 80.1462,
    },
    "Medavakkam": {
        "id": "CHN-023",
        "zone": "South",
        "lat": 12.9186,
        "lon": 80.1878,
    },
    "Tambaram Sanatorium": {
        "id": "CHN-024",
        "zone": "South",
        "lat": 12.9155,
        "lon": 80.1223,
    },
    "Poonamallee": {
        "id": "CHN-025",
        "zone": "West",
        "lat": 13.0475,
        "lon": 80.1090,
    },
    "Nungambakkam": {
        "id": "CHN-026",
        "zone": "Central",
        "lat": 13.0603,
        "lon": 80.2417,
    },
    "Chetpet": {
        "id": "CHN-027",
        "zone": "Central",
        "lat": 13.0725,
        "lon": 80.2432,
    },
    "Kilpauk": {
        "id": "CHN-028",
        "zone": "North",
        "lat": 13.0813,
        "lon": 80.2408,
    },
    "Aminjikarai": {
        "id": "CHN-029",
        "zone": "West",
        "lat": 13.0728,
        "lon": 80.2247,
    },
    "Arumbakkam": {
        "id": "CHN-030",
        "zone": "West",
        "lat": 13.0725,
        "lon": 80.2104,
    },
    "Ashok Nagar": {
        "id": "CHN-031",
        "zone": "West",
        "lat": 13.0357,
        "lon": 80.2101,
    },
    "Saidapet": {
        "id": "CHN-032",
        "zone": "South",
        "lat": 13.0212,
        "lon": 80.2226,
    },
    "Little Mount": {
        "id": "CHN-033",
        "zone": "South",
        "lat": 13.0106,
        "lon": 80.2276,
    },
    "Ekkatuthangal": {
        "id": "CHN-034",
        "zone": "South",
        "lat": 13.0107,
        "lon": 80.2027,
    },
    "Alandur": {
        "id": "CHN-035",
        "zone": "South",
        "lat": 12.9975,
        "lon": 80.2001,
    },
    "Meenambakkam": {
        "id": "CHN-036",
        "zone": "South",
        "lat": 12.9906,
        "lon": 80.1732,
    },
    "Airport": {
        "id": "CHN-037",
        "zone": "South",
        "lat": 12.9941,
        "lon": 80.1709,
    },
    "Pallikaranai": {
        "id": "CHN-038",
        "zone": "South",
        "lat": 12.9345,
        "lon": 80.2107,
    },
    "Madipakkam": {
        "id": "CHN-039",
        "zone": "South",
        "lat": 12.9631,
        "lon": 80.1988,
    },
    "Keelkattalai": {
        "id": "CHN-040",
        "zone": "South",
        "lat": 12.9484,
        "lon": 80.1854,
    },
    "Thoraipakkam": {
        "id": "CHN-041",
        "zone": "East",
        "lat": 12.9401,
        "lon": 80.2367,
    },
    "Karapakkam": {
        "id": "CHN-042",
        "zone": "East",
        "lat": 12.9209,
        "lon": 80.2331,
    },
    "Perungudi": {
        "id": "CHN-043",
        "zone": "East",
        "lat": 12.9635,
        "lon": 80.2422,
    },
    "Navalur": {
        "id": "CHN-044",
        "zone": "East",
        "lat": 12.8410,
        "lon": 80.2264,
    },
    "Siruseri": {
        "id": "CHN-045",
        "zone": "East",
        "lat": 12.8231,
        "lon": 80.2266,
    },
    "Taramani": {
        "id": "CHN-046",
        "zone": "East",
        "lat": 12.9829,
        "lon": 80.2380,
    },
    "Guindy Industrial Estate": {
        "id": "CHN-047",
        "zone": "South",
        "lat": 13.0107,
        "lon": 80.2071,
    },
    "Madhavaram": {
        "id": "CHN-048",
        "zone": "North",
        "lat": 13.1478,
        "lon": 80.2317,
    },
    "Manali": {
        "id": "CHN-049",
        "zone": "North",
        "lat": 13.1653,
        "lon": 80.2649,
    },
    "Ennore": {
        "id": "CHN-050",
        "zone": "North",
        "lat": 13.2146,
        "lon": 80.3221,
    },
}


# ============================================================
# SESSION STATE
# ============================================================

if "result" not in st.session_state:
    st.session_state.result = None

if "history" not in st.session_state:
    st.session_state.history = []

if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL


# ============================================================
# API HELPERS
# ============================================================

@st.cache_data(ttl=10, show_spinner=False)
def check_api_health(
    api_url: str,
) -> Optional[dict]:
    """
    Cached for 10s so a health check doesn't fire on every
    single Streamlit rerun (every widget interaction triggers
    a full script rerun, so without caching this would hit the
    API on every keystroke).
    """

    try:
        response = requests.get(
            f"{api_url}/health",
            timeout=3,
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException:
        return None


def request_prediction(
    api_url: str,
    payload: dict,
) -> dict:

    response = requests.post(
        f"{api_url}/predict",
        json=payload,
        timeout=10,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# RECENT DEMAND
# ============================================================

def fetch_recent_demand(
    station_id: str,
    pred_date: date,
    pred_time: time,
) -> dict:

    hour = pred_time.hour

    is_peak = (
        (7 <= hour < 10)
        or
        (17 <= hour < 20)
    )

    if is_peak:

        base = 190.0
        profile = "high"

    elif 10 <= hour < 17:

        base = 120.0
        profile = "medium"

    else:

        base = 60.0
        profile = "low"

    return {
        "lag_1_passenger_count": base,
        "rolling_avg_4": base * 0.97,
        "demand_profile": profile,
    }


# ============================================================
# DEMAND LEVEL
# ============================================================

def demand_level(
    predicted: float,
) -> tuple[str, str, str]:

    if predicted < 80:

        return (
            "LOW",
            "🟢",
            "level-low",
        )

    elif predicted < 150:

        return (
            "MEDIUM",
            "🟡",
            "level-medium",
        )

    elif predicted < 220:

        return (
            "HIGH",
            "🟠",
            "level-high",
        )

    else:

        return (
            "CRITICAL",
            "🔴",
            "level-critical",
        )


# ============================================================
# HEADER
# ============================================================

st.markdown(
"""<div class="app-header">

<div class="app-kicker">

CHENNAI TRANSIT INTELLIGENCE

</div>

<div class="app-title">
🚇 Chennai Transit AI
</div>

<div class="app-subtitle">
Passenger demand forecasting and operational intelligence
</div>
</div>""",
unsafe_allow_html=True
)


# ============================================================
# SETTINGS + API STATUS
# ============================================================

settings_col, status_col = st.columns(
    [1, 6]
)

with settings_col:

    with st.popover(
        "⚙️ Settings",
        width="stretch",
    ):

        api_url_input = st.text_input(
            "API base URL",
            value=st.session_state.api_url,
        )

        st.session_state.api_url = (
            api_url_input.rstrip("/")
        )


api_url = st.session_state.api_url

health = check_api_health(api_url)


with status_col:

    if health and health.get("model_loaded"):

        st.markdown(
            """
            <span class="status-pill status-online">
                ● API ONLINE · MODEL LOADED
            </span>
            """,
            unsafe_allow_html=True,
        )

    elif health:

        st.markdown(
            """
            <span class="status-pill status-offline">
                ● API ONLINE · MODEL NOT LOADED
            </span>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <span class="status-pill status-offline">
                ● API UNREACHABLE
            </span>
            """,
            unsafe_allow_html=True,
        )


st.write("")


# ============================================================
# CORE INPUTS
# ============================================================

station_name = st.selectbox(
    "🚉 Select Station",
    list(STATIONS.keys()),
)

pred_date = st.date_input(
    "📅 Select Date",
    value=date.today(),
)

pred_time = st.time_input(
    "🕒 Select Time",
    value=time(9, 15),
)


station = STATIONS[station_name]


# ============================================================
# DERIVED TIME FEATURES
# ============================================================

month = pred_date.month
day = pred_date.day
hour = pred_time.hour
minutes = pred_time.minute

day_of_week = pred_date.weekday()

is_weekend = day_of_week >= 5

is_morning_peak = (
    7 <= hour < 10
)

is_evening_peak = (
    17 <= hour < 20
)

is_peak_hour = (
    is_morning_peak
    or
    is_evening_peak
)


# ============================================================
# RECENT DEMAND
# ============================================================

recent_demand = fetch_recent_demand(
    station["id"],
    pred_date,
    pred_time,
)

demand_profile = (
    recent_demand[
        "demand_profile"
    ]
)

lag_1_passenger_count = (
    recent_demand[
        "lag_1_passenger_count"
    ]
)

rolling_avg_4 = (
    recent_demand[
        "rolling_avg_4"
    ]
)


# ============================================================
# WEATHER DEFAULTS
# ============================================================

weather_condition = "clear"
temperature = 31.5
humidity = 68.0
rainfall = 0.0


# ============================================================
# THREE-COLUMN PROFESSIONAL DASHBOARD
# ============================================================

left_col, center_col, right_col = st.columns(
    [1.05, 2.35, 1.05],
    gap="large",
)


# ============================================================
# LEFT COLUMN — NETWORK INTELLIGENCE
# ============================================================

with left_col:

    st.markdown("### Network Intelligence")

    st.html(
    f"""<div class="professional-card">

    <div class="card-kicker">
    SELECTED STATION
    </div>

    <div class="card-main-value">
    🚉 {station_name}
    </div>

    <div class="card-muted">
    {station["id"]}
    </div>

    <div class="card-divider"></div>

    <div class="card-row">
    <span>Zone</span>
    <strong>{station["zone"]}</strong>
    </div>

    <div class="card-row">
    <span>Latitude</span>
    <strong>{station["lat"]:.4f}</strong>
    </div>

    <div class="card-row">
    <span>Longitude</span>
    <strong>{station["lon"]:.4f}</strong>
    </div>

    </div>"""
    )

    peak_status = (
        "Peak Period"
        if is_peak_hour
        else "Normal Period"
    )

    peak_color_class = (
        "status-dot-orange"
        if is_peak_hour
        else "status-dot-green"
    )

    st.html(
        f"""
        <div class="professional-card">

            <div class="card-kicker">
                SERVICE PERIOD
            </div>

            <div class="status-large">

                <span class="{
                    peak_color_class
                }">
                    ●
                </span>

                {peak_status}

            </div>

            <div class="card-muted">
                {pred_time.strftime("%I:%M %p")}
            </div>

            <div class="card-divider"></div>

            <div class="card-row">
                <span>Morning peak</span>
                <strong>
                    {"Active" if is_morning_peak else "Inactive"}
                </strong>
            </div>

            <div class="card-row">
                <span>Evening peak</span>
                <strong>
                    {"Active" if is_evening_peak else "Inactive"}
                </strong>
            </div>

            <div class="card-row">
                <span>Weekend</span>
                <strong>
                    {"Yes" if is_weekend else "No"}
                </strong>
            </div>

        </div>
        """
    )

    api_state = (
        "Operational"
        if health
        else "Unavailable"
    )

    api_color = (
        "#4ade80"
        if health
        else "#f87171"
    )

    st.html(
        f"""
        <div class="professional-card">

            <div class="card-kicker">
                PREDICTION SERVICE
            </div>

            <div class="network-status"
                 style="color: {api_color};">

                ● {api_state}

            </div>

            <div class="card-muted">
                FastAPI prediction endpoint
            </div>

        </div>
        """
    )


# ============================================================
# CENTER COLUMN — FORECAST WORKSPACE
# ============================================================

with center_col:

    st.html(
        """
        <div class="workspace-header">

            <div class="workspace-kicker">
                DEMAND FORECAST
            </div>

            <div class="workspace-title">
                Passenger Demand Prediction
            </div>

            <div class="workspace-description">
                Estimate expected passenger volume using
                station, temporal, weather and recent-demand
                features.
            </div>

        </div>
        """
    )

    with st.expander(
        "⚙️ Advanced prediction parameters"
    ):

        st.markdown("#### Weather Conditions")

        weather_col_1, weather_col_2 = st.columns(2)

        with weather_col_1:

            weather_condition = st.selectbox(
                "Weather condition",
                [
                    "clear",
                    "cloudy",
                    "rain",
                    "fog",
                    "storm",
                ],
            )

        with weather_col_2:

            temperature = st.number_input(
                "Temperature (°C)",
                value=31.5,
                step=0.5,
            )

        weather_col_3, weather_col_4 = st.columns(2)

        with weather_col_3:

            humidity = st.number_input(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=68.0,
                step=1.0,
            )

        with weather_col_4:

            rainfall = st.number_input(
                "Rainfall (mm)",
                min_value=0.0,
                value=0.0,
                step=0.5,
            )

        st.markdown(
            "#### Recent Demand Features"
        )

        st.caption(
            f"Demand profile: "
            f"**{demand_profile.upper()}** · "
            f"Lag-1: **{lag_1_passenger_count:,.0f}** · "
            f"Rolling average: **{rolling_avg_4:,.0f}**"
        )

        override_demand = st.checkbox(
            "Override recent-demand values manually"
        )

        if override_demand:

            override_1, override_2, override_3 = (
                st.columns(3)
            )

            with override_1:

                demand_profile = st.selectbox(
                    "Demand profile",
                    [
                        "low",
                        "medium",
                        "high",
                    ],
                    index=[
                        "low",
                        "medium",
                        "high",
                    ].index(
                        demand_profile
                    ),
                )

            with override_2:

                lag_1_passenger_count = (
                    st.number_input(
                        "Lag-1 passenger count",
                        min_value=0.0,
                        value=float(
                            lag_1_passenger_count
                        ),
                        step=1.0,
                    )
                )

            with override_3:

                rolling_avg_4 = (
                    st.number_input(
                        "Rolling avg (last 4)",
                        min_value=0.0,
                        value=float(
                            rolling_avg_4
                        ),
                        step=1.0,
                    )
                )


# ============================================================
# RIGHT COLUMN — OPERATIONAL VIEW
# ============================================================

with right_col:

    st.markdown("### Operational View")

    st.html(
        f"""
        <div class="professional-card">

            <div class="card-kicker">
                DEMAND PROFILE
            </div>

            <div class="profile-value">
                {demand_profile.upper()}
            </div>

            <div class="card-muted">
                Current historical pattern
            </div>

        </div>
        """
    )

    lag_display = f"{lag_1_passenger_count:,.0f}"

    st.html(
        f"""
        <div class="professional-card">

            <div class="card-kicker">
                RECENT VOLUME
            </div>

            <div class="big-number">
                {lag_display}
            </div>

            <div class="card-muted">
                passengers · latest observation
            </div>

        </div>
        """
    )

    rolling_display = f"{rolling_avg_4:,.0f}"

    st.html(
        f"""
        <div class="professional-card">

            <div class="card-kicker">
                ROLLING BASELINE
            </div>

            <div class="big-number">
                {rolling_display}
            </div>

            <div class="card-muted">
                average · last 4 observations
            </div>

        </div>
        """
    )

    weather_icons = {
        "clear": "☀️",
        "cloudy": "☁️",
        "rain": "🌧️",
        "fog": "🌫️",
        "storm": "⛈️",
    }

    weather_icon = weather_icons.get(
        weather_condition,
        "🌤️",
    )

    st.html(
        f"""
        <div class="professional-card">

            <div class="card-kicker">
                WEATHER
            </div>

            <div class="status-large">
                {weather_icon}
                {weather_condition.title()}
            </div>

            <div class="card-muted">
                {temperature:.1f}°C ·
                {humidity:.0f}% humidity
            </div>

        </div>
        """
        )


# ============================================================
# PREDICTION PAYLOAD
# ============================================================

payload = {
    "station_id": station["id"],
    "zone": station["zone"],
    "latitude": station["lat"],
    "longitude": station["lon"],
    "month": month,
    "day": day,
    "hour": hour,
    "minutes": minutes,
    "day_of_week": day_of_week,
    "is_weekend": is_weekend,
    "demand_profile": demand_profile,
    "is_morning_peak": is_morning_peak,
    "is_evening_peak": is_evening_peak,
    "is_peak_hour": is_peak_hour,
    "weather_condition": weather_condition,
    "temperature": temperature,
    "humidity": humidity,
    "rainfall": rainfall,
    "lag_1_passenger_count": (
        lag_1_passenger_count
    ),
    "rolling_avg_4": rolling_avg_4,
}


# ============================================================
# PREDICT BUTTON
# ============================================================

st.markdown("")

predict_clicked = st.button(
    "🔍  Generate Passenger Demand Forecast",
    type="primary",
    width="stretch",
)


# ============================================================
# PREDICT
# ============================================================

if predict_clicked:

    if health is None:

        st.error(
            f"Can't reach the API at `{api_url}`. "
            "Is FastAPI running?"
        )

    else:

        with st.spinner(
            "Generating passenger demand forecast..."
        ):

            try:

                result = request_prediction(
                    api_url,
                    payload,
                )

            except requests.exceptions.HTTPError as exc:

                if exc.response is not None:

                    try:

                        detail = (
                            exc.response
                            .json()
                            .get(
                                "detail",
                                str(exc),
                            )
                        )

                    except ValueError:

                        detail = str(exc)

                    status_code = (
                        exc.response.status_code
                    )

                else:

                    detail = str(exc)
                    status_code = "?"

                st.error(
                    f"Prediction failed "
                    f"({status_code}): {detail}"
                )

                st.session_state.result = None

            except requests.exceptions.RequestException as exc:

                st.error(
                    f"Request to API failed: {exc}"
                )

                st.session_state.result = None

            except (KeyError, TypeError, ValueError) as exc:

                st.error(
                    "The API returned an unexpected "
                    f"prediction response: {exc}"
                )

                st.session_state.result = None

            else:

                try:

                    predicted = float(
                        result[
                            "predicted_passenger_count"
                        ]
                    )

                except (
                    KeyError,
                    TypeError,
                    ValueError,
                ):

                    st.error(
                        "The API response does not contain "
                        "a valid predicted_passenger_count."
                    )

                    st.session_state.result = None

                else:

                    label, emoji, css_class = (
                        demand_level(predicted)
                    )

                    st.session_state.result = predicted

                    st.session_state.history.append(
                        {
                            "timestamp": datetime.now(),
                            "station": station_name,
                            "date": pred_date.isoformat(),
                            "time": pred_time.strftime(
                                "%I:%M %p"
                            ),
                            "predicted_passenger_count": round(
                                predicted,
                                2,
                            ),
                            "demand_level": label,
                            "hour": hour,
                            "is_peak_hour": is_peak_hour,
                        }
                    )


# ============================================================
# PREDICTION RESULT
# ============================================================

if st.session_state.result is not None:

    predicted = float(
        st.session_state.result
    )

    label, emoji, css_class = demand_level(
        predicted
    )

    # IMPORTANT:
    # Pre-format before multiline HTML f-string.
    # This prevents the invalid format-specifier error.

    predicted_display = f"{predicted:,.0f}"

    st.markdown("---")

    st.html(
        f"""
        <div class="result-card">

            <div class="result-label">
                EXPECTED PASSENGER DEMAND
            </div>

            <div class="result-count">
                👥 {predicted_display}
                <span class="result-unit">
                    passengers
                </span>
            </div>

            <div class="result-demand">

                Demand Level:

                <span class="result-level {css_class}">
                    {emoji} {label}
                </span>

            </div>

        </div>
        """
    )


# ============================================================
# SHARED HISTORY DATAFRAME
# ============================================================
# Built once here so it's not silently reused across two
# separate `if st.session_state.history:` blocks below.

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
else:
    history_df = None


# ============================================================
# OPERATIONAL SUMMARY
# ============================================================

if history_df is not None:

    total_predictions = len(
        history_df
    )

    avg_demand = float(
        history_df[
            "predicted_passenger_count"
        ].mean()
    )

    max_demand = float(
        history_df[
            "predicted_passenger_count"
        ].max()
    )

    peak_predictions = int(
        history_df[
            "is_peak_hour"
        ].sum()
    )

    avg_display = f"{avg_demand:,.0f}"
    max_display = f"{max_demand:,.0f}"

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        'Operational Summary'
        '</div>',
        unsafe_allow_html=True,
    )

    metric_1, metric_2, metric_3, metric_4 = (
        st.columns(4)
    )

    with metric_1:

        st.metric(
            "Forecasts Generated",
            total_predictions,
        )

    with metric_2:

        st.metric(
            "Average Demand",
            avg_display,
        )

    with metric_3:

        st.metric(
            "Maximum Demand",
            max_display,
        )

    with metric_4:

        st.metric(
            "Peak Forecasts",
            peak_predictions,
        )


# ============================================================
# PREDICTION HISTORY
# ============================================================

if history_df is not None:

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '📊 Prediction History'
        '</div>',
        unsafe_allow_html=True,
    )

    display_history = history_df[
        [
            "station",
            "date",
            "time",
            "predicted_passenger_count",
            "demand_level",
        ]
    ].copy()

    display_history = display_history.rename(
        columns={
            "station": "Station",
            "date": "Date",
            "time": "Time",
            "predicted_passenger_count":
                "Predicted Passengers",
            "demand_level":
                "Demand Level",
        }
    )

    st.dataframe(
        display_history,
        width="stretch",
        hide_index=True,
    )

    csv_bytes = (
        display_history
        .to_csv(index=False)
        .encode("utf-8")
    )

    download_col, clear_col = st.columns(2)

    with download_col:

        st.download_button(
            "⬇️ Download History CSV",
            data=csv_bytes,
            file_name=(
                "chennai_transit_predictions.csv"
            ),
            mime="text/csv",
            width="stretch",
        )

    with clear_col:

        if st.button(
            "🗑️ Clear History",
            width="stretch",
        ):

            st.session_state.history = []
            st.session_state.result = None

            st.rerun()


# ============================================================
# ANALYTICS
# ============================================================

if len(st.session_state.history) >= 2:

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '📈 Prediction Analytics'
        '</div>',
        unsafe_allow_html=True,
    )

    analytics_df = history_df.copy()

    # --------------------------------------------------------
    # Demand trend
    # --------------------------------------------------------

    st.markdown("#### 📈 Demand Trend")

    analytics_df["prediction_time"] = (
        pd.to_datetime(
            analytics_df["timestamp"]
        )
    )

    # Extract hour from prediction timestamp
    trend_df = (
        analytics_df
        .groupby(
            "hour",
            as_index=False,
        )[
            "predicted_passenger_count"
        ]
        .mean()
        .sort_values("hour")
        .set_index("hour")
    )

    st.line_chart(
        trend_df,
        width="stretch",
)

    # --------------------------------------------------------
    # Station comparison
    # --------------------------------------------------------

    st.markdown("#### 🚉 Station Comparison")

    station_comparison = (
        analytics_df
        .groupby(
            "station",
            as_index=False,
        )[
            "predicted_passenger_count"
        ]
        .mean()
        .sort_values(
            "predicted_passenger_count",
            ascending=False,
        )
    )

    station_chart = (
        station_comparison
        .set_index("station")
    )

    st.bar_chart(
        station_chart,
        width="stretch",
    )

    # --------------------------------------------------------
    # Hourly demand
    # --------------------------------------------------------

    st.markdown("#### ⏰ Hourly Demand Analysis")

    hourly_demand = (
        analytics_df
        .groupby(
            "hour",
            as_index=False,
        )[
            "predicted_passenger_count"
        ]
        .mean()
        .sort_values("hour")
    )

    hourly_chart = (
        hourly_demand
        .set_index("hour")
    )

    st.bar_chart(
        hourly_chart,
        width="stretch",
    )

    # --------------------------------------------------------
    # Peak vs non-peak
    # --------------------------------------------------------

    st.markdown(
        "#### 🚦 Peak vs Non-Peak Demand"
    )

    peak_analysis = (
        analytics_df
        .groupby(
            "is_peak_hour",
            as_index=False,
        )[
            "predicted_passenger_count"
        ]
        .mean()
    )

    peak_analysis["period"] = (
        peak_analysis[
            "is_peak_hour"
        ].map(
            {
                True: "Peak Hour",
                False: "Non-Peak Hour",
            }
        )
    )

    peak_chart = (
        peak_analysis
        .set_index("period")
        [
            [
                "predicted_passenger_count"
            ]
        ]
    )

    st.bar_chart(
        peak_chart,
        width="stretch",
    )


elif len(st.session_state.history) == 1:

    st.info(
        "📈 Make at least one more prediction "
        "to unlock demand analytics and comparisons."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="dashboard-footer">
        Chennai Transit AI · Passenger Demand Intelligence
        · FastAPI + Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)