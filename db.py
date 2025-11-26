import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path("data/app.db")
DB_PATH.parent.mkdir(exist_ok=True)

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        text TEXT,
        true_label INTEGER,
        pred_label INTEGER,
        confidence REAL
    );
    """)
    conn.commit()
    conn.close()

def log_prediction(text, true_label, pred_label, confidence):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO predictions (timestamp, text, true_label, pred_label, confidence)
    VALUES (?, ?, ?, ?, ?)
    """, (datetime.utcnow().isoformat(), text, true_label, pred_label, confidence))
    conn.commit()
    conn.close()

def fetch_all_predictions():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, timestamp, text, true_label, pred_label, confidence
        FROM predictions
        ORDER BY timestamp DESC
    """)
    rows = cur.fetchall()
    conn.close()
    # On transforme chaque row en dict pour que pandas récupère bien les noms de colonnes
    return [dict(row) for row in rows]
