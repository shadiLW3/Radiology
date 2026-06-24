"""SQLite persistence (stdlib sqlite3 — no ORM)."""
import os
import sqlite3

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "..", "data")
CASES_DIR = os.path.join(DATA_DIR, "cases")
DB_PATH = os.path.join(DATA_DIR, "app.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    image_path TEXT, gt_mask_path TEXT, model_mask_path TEXT,
    gt_diagnosis TEXT, model_diagnosis TEXT,
    width INTEGER, height INTEGER,
    modality TEXT DEFAULT 'dermoscopy'
);
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT, case_id TEXT, badge TEXT, created_at TEXT,
    diagnosis TEXT, confidence INTEGER, draw_ms INTEGER,
    dice REAL, iou REAL, threshold_jaccard REAL, hausdorff95 REAL,
    diagnosis_correct INTEGER, beat_model_on_dice INTEGER, model_dice REAL,
    verified INTEGER DEFAULT 0, modality TEXT DEFAULT 'dermoscopy'
);
CREATE TABLE IF NOT EXISTS seen (
    session_id TEXT, case_id TEXT,
    PRIMARY KEY (session_id, case_id)
);
CREATE TABLE IF NOT EXISTS credentials (
    session_id TEXT PRIMARY KEY,
    badge TEXT, npi_last4 TEXT, specialty TEXT,
    name_match INTEGER, verified_at TEXT
);
"""


def get_conn():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.executescript(SCHEMA)
    # migrations for pre-existing DBs
    acols = [r[1] for r in conn.execute("PRAGMA table_info(attempts)")]
    if "verified" not in acols:
        conn.execute("ALTER TABLE attempts ADD COLUMN verified INTEGER DEFAULT 0")
    if "modality" not in acols:
        conn.execute("ALTER TABLE attempts ADD COLUMN modality TEXT DEFAULT 'dermoscopy'")
    ccols = [r[1] for r in conn.execute("PRAGMA table_info(cases)")]
    if "modality" not in ccols:
        conn.execute("ALTER TABLE cases ADD COLUMN modality TEXT DEFAULT 'dermoscopy'")
    conn.commit()
    conn.close()


def reset_modality(modality):
    """Drop existing cases for ONE modality (so modalities coexist on reseed)."""
    conn = get_conn()
    conn.execute("DELETE FROM cases WHERE modality = ?", (modality,))
    conn.commit()
    conn.close()
