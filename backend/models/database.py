import sqlite3
from datetime import datetime

DB_PATH = "plates.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS plates(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_text TEXT,
                confidence REAL,
                detected_at TEXT
        )
    """
    )
    conn.commit()
    conn.close()


def save_plate_result(result):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO plates (plate_text, confidence, detected_at)
        VALUES (?, ?, ?)
    """,
        (result["plate_text"], result["confidence"], datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
