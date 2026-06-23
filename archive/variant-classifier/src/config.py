"""Central configuration for the variant pathogenicity classifier.

All paths are derived from the repository root so scripts run from anywhere.
Tune the knobs in the CONFIG section to trade off dataset size vs. runtime.
"""
import os

# --- Paths -------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")

RAW_CLINVAR_GZ = os.path.join(DATA_DIR, "variant_summary.txt.gz")
LABELED_CSV = os.path.join(DATA_DIR, "clinvar_labeled.csv")
ANNOTATIONS_CSV = os.path.join(DATA_DIR, "annotations.csv")
FEATURES_CSV = os.path.join(DATA_DIR, "features.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.json")
FEATURE_COLS_PATH = os.path.join(MODELS_DIR, "feature_columns.json")
METRICS_PATH = os.path.join(REPORTS_DIR, "metrics.json")

# --- Data sources ------------------------------------------------------------
CLINVAR_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)

# --- CONFIG: dataset + model knobs ------------------------------------------
RANDOM_SEED = 42
# Cap on the number of labeled variants kept (stratified). Lower = faster.
MAX_VARIANTS = 40_000
# Minimum ClinVar review-confidence stars to keep (0-4). 1 = "criteria provided".
MIN_REVIEW_STARS = 1
# Batch size for the myvariant.info annotation API.
ANNOTATION_BATCH = 1000
# Fraction of GENES held out for the test set (grouped split prevents leakage).
TEST_SIZE = 0.2
VALID_SIZE = 0.1  # carved out of the training genes for early stopping
