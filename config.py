from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

for _d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# MENA country display names → ISO2 codes
MENA_COUNTRIES = {
    "Yemen": "YE",
    "Syria": "SY",
    "Iraq": "IQ",
    "Lebanon": "LB",
    "Libya": "LY",
    "Sudan": "SD",
    "Egypt": "EG",
    "Morocco": "MA",
    "Algeria": "DZ",
    "Tunisia": "TN",
    "Jordan": "JO",
    "Saudi Arabia": "SA",
    "Iran": "IR",
    "Palestine": "PS",
}

ISO2_TO_NAME = {v: k for k, v in MENA_COUNTRIES.items()}

# ISO alpha-3 codes for Plotly choropleth
MENA_ISO3 = {
    "Yemen": "YEM",
    "Syria": "SYR",
    "Iraq": "IRQ",
    "Lebanon": "LBN",
    "Libya": "LBY",
    "Sudan": "SDN",
    "Egypt": "EGY",
    "Morocco": "MAR",
    "Algeria": "DZA",
    "Tunisia": "TUN",
    "Jordan": "JOR",
    "Saudi Arabia": "SAU",
    "Iran": "IRN",
    "Palestine": "PSE",
}

NAME_TO_ISO3 = MENA_ISO3

# HMM
N_STATES = 3
# These labels are assigned after fitting, based on mean intensity ordering
STATE_LABELS = {0: "Stable", 1: "Rising Tension", 2: "Active Conflict"}
STATE_COLORS = {"Stable": "#2ecc71", "Rising Tension": "#f39c12", "Active Conflict": "#e74c3c"}
STATE_INT_COLORS = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}

# XGBoost
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "mlogloss",
    "random_state": 42,
}

# GDELT DOC API
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_TIMESPAN = "6m"
GDELT_CONFLICT_KEYWORDS = "conflict war attack violence battle killed"

# RSS feeds (MENA-focused)
NEWS_FEEDS = [
    "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/MiddleEast.xml",
]

# Feature engineering
LAG_WEEKS = [1, 2, 4]
ROLLING_WINDOWS = [4, 8]

# Intensity score weights (features must be normalized before applying)
INTENSITY_WEIGHTS = {
    "conflict_tone_neg": 0.40,   # -conflict_tone (negative tone → higher intensity)
    "conflict_volume_norm": 0.30,
    "conflict_ratio_norm": 0.20,
    "news_negativity_norm": 0.10,
}

# Forecast horizon (weeks)
FORECAST_WEEKS = 8
