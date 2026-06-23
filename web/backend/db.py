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
    width INTEGER, height INTEGER
);
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT, case_id TEXT, badge TEXT, created_at TEXT,
    diagnosis TEXT, confidence INTEGER, draw_ms INTEGER,
    dice REAL, iou REAL, threshold_jaccard REAL, hausdorff95 REAL,
    diagnosis_correct INTEGER, beat_model_on_dice INTEGER, model_dice REAL,
    verified INTEGER DEFAULT 0
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
    # migration: add attempts.verified to pre-existing DBs
    cols = [r[1] for r in conn.execute("PRAGMA table_info(attempts)")]
    if "verified" not in cols:
        conn.execute("ALTER TABLE attempts ADD COLUMN verified INTEGER DEFAULT 0")
    conn.commit()
    conn.close()


def reset_cases():
    """Drop existing cases (used by seed_cases for a clean reseed)."""
    conn = get_conn()
    conn.execute("DELETE FROM cases")
    conn.commit()
    conn.close()
