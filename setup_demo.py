"""
Generate a demo SQLite database with realistic sample transaction data
and model accuracy metrics. Run this once to bootstrap the dashboard.

Usage: python setup_demo.py
"""

import sqlite3
import json
import random
import os
from pathlib import Path
from datetime import datetime, timedelta

DB_PATH = Path(__file__).parent / "re_analytics.db"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ZIP profiles: (zip, city, avg_price, price_std, avg_sqft, sqft_std, avg_lot, avg_dom, count)
ZIP_PROFILES = [
    ("95765", "Roseville", 625000, 120000, 2100, 500, 7200, 22, 1102),
    ("95677", "Rocklin", 680000, 150000, 2250, 550, 8000, 19, 735),
    ("95678", "Roseville", 475000, 90000, 1750, 400, 5500, 25, 731),
    ("95661", "Roseville", 530000, 100000, 1900, 450, 6200, 21, 701),
    ("95746", "Granite Bay", 1050000, 350000, 3200, 800, 25000, 35, 669),
    ("95650", "Loomis", 875000, 250000, 2800, 700, 18000, 30, 357),
]

MODEL_ACCURACY = [
    ("95650", 69, 14.6, 302592, "low"),
    ("95661", 138, 10.8, 126281, "low"),
    ("95677", 144, 11.4, 169287, "low"),
    ("95678", 144, 16.9, 135774, "low"),
    ("95746", 127, 11.4, 291519, "low"),
    ("95765", 201, 12.4, 137713, "low"),
]

# Realistic OLS coefficients
COEFFICIENTS = {
    "coefficients": {
        "const": -157832.45,
        "sqft": 198.73,
        "lot_sqft_log": 42156.88,
        "beds": -18432.61,
        "baths": 27841.33,
        "age": -2145.67,
        "hoa": -89.23,
        "zip_95661": 12453.82,
        "zip_95677": 48721.55,
        "zip_95678": -62134.21,
        "zip_95746": 215643.90,
        "zip_95765": 18234.67
    },
    "base_zip": "95650",
    "r_squared": 0.83,
    "rmse": 190620,
    "mape": 12.5,
    "n_train": 3456,
    "n_test": 864,
    "trained_at": "2026-03-14T14:30:00"
}

STREETS = {
    "95765": ["Westpark Dr", "Diamond Creek Blvd", "Fiddyment Rd", "Blue Oaks Blvd", "Pleasant Grove Blvd", "Hayden Pkwy", "Woodcreek Oaks Blvd"],
    "95677": ["Granite Dr", "Pacific St", "Sunset Blvd", "Stanford Ranch Rd", "Clover Valley Rd", "Whitney Ranch Pkwy", "Aguilar Rd"],
    "95678": ["Cirby Way", "Douglas Blvd", "Riverside Ave", "Main St", "Oak St", "Vernon St", "Judah St"],
    "95661": ["Foothills Blvd", "Junction Blvd", "Atkinson St", "Industrial Ave", "Washington Blvd", "Roseville Pkwy", "Lead Hill Blvd"],
    "95746": ["Auburn Folsom Rd", "Douglas Blvd", "Barton Rd", "Eureka Rd", "Olympus Dr", "Ridge View Dr", "Granite Bay Ct"],
    "95650": ["Taylor Rd", "Horseshoe Bar Rd", "King Rd", "Sierra College Blvd", "Brace Rd", "Barton Rd", "Bankhead Rd"],
}


def create_tables(conn):
    conn.executescript("""
        DROP TABLE IF EXISTS transactions;
        DROP TABLE IF EXISTS parcels;
        DROP TABLE IF EXISTS model_accuracy;
        DROP TABLE IF EXISTS interactions;
        DROP TABLE IF EXISTS lead_scores;
        DROP TABLE IF EXISTS follow_ups;

        CREATE TABLE transactions (
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

        CREATE TABLE parcels (
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

        CREATE TABLE model_accuracy (
            zip TEXT PRIMARY KEY,
            n_transactions INTEGER,
            mape REAL,
            rmse REAL,
            confidence_tier TEXT,
            last_updated TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            address TEXT,
            contact_name TEXT,
            interaction_type TEXT,
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE lead_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parcel_id TEXT,
            address TEXT,
            score REAL,
            factors TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE follow_ups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            address TEXT,
            contact_name TEXT,
            due_date TEXT,
            priority TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)


def generate_transactions(conn):
    base_date = datetime(2023, 3, 14)
    rows = []

    for zip_code, city, avg_price, price_std, avg_sqft, sqft_std, avg_lot, avg_dom, count in ZIP_PROFILES:
        streets = STREETS[zip_code]
        for i in range(count):
            sqft = max(800, int(random.gauss(avg_sqft, sqft_std)))
            beds = max(1, min(6, round(sqft / 550)))
            baths = max(1, min(5, round(sqft / 700, 1)))
            baths = round(baths * 2) / 2  # snap to .5
            year_built = random.choice(range(1960, 2025))
            lot_sqft = max(2000, int(random.gauss(avg_lot, avg_lot * 0.3)))
            hoa = random.choice([0, 0, 0, 0, 0, 50, 75, 100, 125, 150, 200])
            dom = max(1, int(random.gauss(avg_dom, avg_dom * 0.5)))
            days_offset = random.randint(0, 1095)
            sale_date = (base_date + timedelta(days=days_offset)).strftime("%Y-%m-%d")

            # Price correlates with features
            price = (
                avg_price
                + (sqft - avg_sqft) * 200
                + (beds - 3) * (-15000)
                + (baths - 2) * 25000
                + (2000 - (2026 - year_built)) * 50
                + random.gauss(0, price_std * 0.5)
            )
            price = max(100000, int(price / 1000) * 1000)

            street = random.choice(streets)
            num = random.randint(100, 9999)
            address = f"{num} {street}"

            rows.append((
                address, city, zip_code, price, sale_date, beds, baths,
                sqft, lot_sqft, year_built, hoa, dom,
                "Single Family Residential", None, None, "redfin"
            ))

    conn.executemany("""
        INSERT INTO transactions
        (address, city, zip, sale_price, sale_date, beds, baths, sqft,
         lot_sqft, year_built, hoa, days_on_market, property_type,
         latitude, longitude, data_source)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    print(f"  Inserted {len(rows)} transactions")


def insert_model_accuracy(conn):
    for zip_code, n, mape, rmse, tier in MODEL_ACCURACY:
        conn.execute(
            "INSERT INTO model_accuracy (zip, n_transactions, mape, rmse, confidence_tier) VALUES (?,?,?,?,?)",
            (zip_code, n, mape, rmse, tier)
        )
    conn.commit()
    print(f"  Inserted {len(MODEL_ACCURACY)} model accuracy records")


def save_coefficients():
    path = MODELS_DIR / "coefficients.json"
    with open(path, "w") as f:
        json.dump(COEFFICIENTS, f, indent=2)
    print(f"  Saved coefficients to {path}")


if __name__ == "__main__":
    if DB_PATH.exists():
        os.remove(DB_PATH)
        print(f"Removed existing {DB_PATH}")

    conn = sqlite3.connect(str(DB_PATH))
    print("Creating tables...")
    create_tables(conn)
    print("Generating transactions...")
    generate_transactions(conn)
    print("Inserting model accuracy...")
    insert_model_accuracy(conn)
    print("Saving coefficients...")
    save_coefficients()
    conn.close()
    print(f"\nDone. Database at {DB_PATH}")
    print("Run: python app.py")
