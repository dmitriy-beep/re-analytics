"""
db.py — Database initialization and connection management.

Run directly to initialize the database:
    python src/db.py
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "re_analytics.db"


def get_connection() -> sqlite3.Connection:
    """Return a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create all tables if they don't already exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            address TEXT NOT NULL,
            city TEXT,
            zip TEXT,
            sale_price INTEGER NOT NULL,
            sale_date TEXT NOT NULL,
            beds INTEGER,
            baths REAL,
            sqft INTEGER NOT NULL,
            lot_sqft INTEGER,
            year_built INTEGER,
            hoa INTEGER DEFAULT 0,
            days_on_market INTEGER,
            property_type TEXT,
            latitude REAL,
            longitude REAL,
            data_source TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS parcels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parcel_id TEXT,
            address TEXT NOT NULL,
            city TEXT,
            zip TEXT,
            owner_name TEXT,
            purchase_date TEXT,
            purchase_price INTEGER,
            assessed_value INTEGER,
            sqft INTEGER,
            lot_sqft INTEGER,
            year_built INTEGER,
            beds INTEGER,
            baths REAL,
            property_type TEXT,
            latitude REAL,
            longitude REAL,
            data_source TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS model_accuracy (
            zip TEXT PRIMARY KEY,
            n_transactions INTEGER,
            mape REAL,
            rmse REAL,
            confidence_tier TEXT,
            last_updated TEXT DEFAULT CURRENT_TIMESTAMP
        );

        -- Reserved for future use
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parcel_id TEXT,
            address TEXT,
            contact_type TEXT,       -- 'door_knock', 'call', 'email'
            outcome TEXT,            -- 'no_answer', 'not_interested', 'lead', 'callback'
            notes TEXT,
            contacted_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS lead_scores (
            parcel_id TEXT PRIMARY KEY,
            address TEXT,
            score REAL,
            score_components TEXT,   -- JSON blob
            last_scored TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS follow_ups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parcel_id TEXT,
            address TEXT,
            trigger_reason TEXT,
            follow_up_date TEXT,
            completed INTEGER DEFAULT 0,
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    conn.close()
    print(f"Database initialized at: {DB_PATH}")


if __name__ == "__main__":
    init_db()
