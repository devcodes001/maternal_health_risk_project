"""
database.py — SQLite helpers for MaternalCare AI (Extended Edition)
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / 'maternalcare.db'

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    age         INTEGER,
    lmp_date    TEXT,
    blood_group TEXT,
    phone       TEXT,
    doctor_name TEXT,
    hospital    TEXT,
    height_cm   REAL,
    weight_kg   REAL,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS assessments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id   INTEGER NOT NULL,
    age          REAL,
    systolic_bp  REAL,
    diastolic_bp REAL,
    bs           REAL,
    body_temp    REAL,
    heart_rate   REAL,
    weight_kg    REAL,
    hemoglobin   REAL,
    urine_protein TEXT,
    swelling     INTEGER DEFAULT 0,
    headache     INTEGER DEFAULT 0,
    vision_change INTEGER DEFAULT 0,
    risk_level   TEXT,
    risk_score   REAL,
    notes        TEXT,
    assessed_at  TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id)
);

CREATE TABLE IF NOT EXISTS appointments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id   INTEGER NOT NULL,
    title        TEXT NOT NULL,
    date         TEXT NOT NULL,
    time         TEXT,
    type         TEXT DEFAULT 'routine',
    doctor       TEXT,
    notes        TEXT,
    is_completed INTEGER DEFAULT 0,
    created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id)
);

CREATE TABLE IF NOT EXISTS journal_entries (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id       INTEGER NOT NULL,
    pregnancy_week   INTEGER,
    pregnancy_day    INTEGER,
    title            TEXT,
    content          TEXT NOT NULL,
    mood             TEXT DEFAULT 'neutral',
    symptoms         TEXT DEFAULT '[]',
    baby_movements   INTEGER DEFAULT 0,
    entry_date       TEXT NOT NULL,
    created_at       TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id)
);

CREATE TABLE IF NOT EXISTS water_intake (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id    INTEGER NOT NULL,
    date          TEXT NOT NULL,
    cups_consumed INTEGER DEFAULT 0,
    goal_cups     INTEGER DEFAULT 8,
    FOREIGN KEY (patient_id) REFERENCES patients(id),
    UNIQUE(patient_id, date)
);

CREATE TABLE IF NOT EXISTS medications (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id  INTEGER NOT NULL,
    name        TEXT NOT NULL,
    dosage      TEXT,
    frequency   TEXT DEFAULT 'once_daily',
    times       TEXT DEFAULT '["08:00"]',
    start_date  TEXT,
    end_date    TEXT,
    color       TEXT DEFAULT '#667eea',
    purpose     TEXT,
    notes       TEXT,
    is_active   INTEGER DEFAULT 1,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id)
);

CREATE TABLE IF NOT EXISTS medication_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    medication_id   INTEGER NOT NULL,
    patient_id      INTEGER NOT NULL,
    date            TEXT NOT NULL,
    scheduled_time  TEXT,
    taken           INTEGER DEFAULT 0,
    taken_at        TEXT,
    FOREIGN KEY (medication_id) REFERENCES medications(id)
);

CREATE TABLE IF NOT EXISTS clinic_visits (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id        INTEGER NOT NULL,
    date              TEXT NOT NULL,
    doctor_name       TEXT,
    hospital          TEXT,
    pregnancy_week    INTEGER,
    weight_kg         REAL,
    bp_systolic       INTEGER,
    bp_diastolic      INTEGER,
    hemoglobin        REAL,
    urine_protein     TEXT,
    fetal_heart_rate  INTEGER,
    fundal_height     REAL,
    baby_position     TEXT,
    doctor_notes      TEXT,
    patient_concerns  TEXT,
    prescriptions     TEXT,
    next_visit_date   TEXT,
    created_at        TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id)
);

CREATE TABLE IF NOT EXISTS diet_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id  INTEGER NOT NULL,
    date        TEXT NOT NULL,
    meal_type   TEXT,
    food_items  TEXT NOT NULL,
    portion     TEXT DEFAULT 'medium',
    nutrients   TEXT DEFAULT '{}',
    notes       TEXT,
    logged_at   TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id)
);

CREATE TABLE IF NOT EXISTS activity_logs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id       INTEGER NOT NULL,
    date             TEXT NOT NULL,
    activity_type    TEXT,
    duration_minutes INTEGER,
    intensity        TEXT DEFAULT 'light',
    steps            INTEGER DEFAULT 0,
    notes            TEXT,
    logged_at        TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id)
);
"""

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    print(f"[OK] Database ready at {DB_PATH}")

def get_patient(patient_id):
    conn = get_db()
    p = conn.execute('SELECT * FROM patients WHERE id=?', (patient_id,)).fetchone()
    conn.close()
    return dict(p) if p else None
