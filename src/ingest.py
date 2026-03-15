"""
ingest.py — CSV cleaning and database loading.

Usage:
    python src/ingest.py data/raw/your_redfin_file.csv
"""

import sys
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# Add src/ to path so we can import db
sys.path.insert(0, str(Path(__file__).parent))
from db import get_connection, init_db

# ── Column mapping: Redfin header → our schema ──────────────────────────────

REDFIN_COLUMN_MAP = {
    "SOLD DATE": "sale_date",
    "PROPERTY TYPE": "property_type",
    "ADDRESS": "address",
    "CITY": "city",
    "ZIP OR POSTAL CODE": "zip",
    "PRICE": "sale_price",
    "BEDS": "beds",
    "BATHS": "baths",
    "SQUARE FEET": "sqft",
    "LOT SIZE": "lot_sqft",
    "YEAR BUILT": "year_built",
    "DAYS ON MARKET": "days_on_market",
    "HOA/MONTH": "hoa",
    "LATITUDE": "latitude",
    "LONGITUDE": "longitude",
}

KEEP_PROPERTY_TYPES = {
    "Single Family Residential",
    "Townhouse",
    "Condo/Co-op",
    "Condo",
}

# ── Cleaning thresholds ──────────────────────────────────────────────────────

MIN_SALE_PRICE = 50_000
MIN_SQFT = 400
MAX_SQFT = 10_000
MIN_PRICE_PER_SQFT = 50
MAX_PRICE_PER_SQFT = 1_500


def load_redfin_csv(file_path: str | Path) -> pd.DataFrame:
    """
    Load a Redfin sold homes CSV, skipping the first metadata row if present.

    Redfin CSVs sometimes have a 'PAST SALE' or similar label in row 0
    before the actual header. This function handles both formats.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV not found: {file_path}")

    # Peek at first line to check for Redfin metadata row
    with open(file_path, "r", encoding="utf-8-sig") as f:
        first_line = f.readline().strip()

    # If the first cell isn't a known column name, skip it
    skip_rows = 0 if first_line.startswith("SALE TYPE") else 1

    df = pd.read_csv(file_path, skiprows=skip_rows, dtype=str)
    df.columns = df.columns.str.strip()
    return df


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename Redfin columns to schema names and drop columns we don't need.
    Raises ValueError if any required columns are missing.
    """
    missing = [col for col in REDFIN_COLUMN_MAP if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns in CSV: {missing}\n"
            f"Actual columns: {list(df.columns)}"
        )

    df = df.rename(columns=REDFIN_COLUMN_MAP)
    keep = list(REDFIN_COLUMN_MAP.values())
    return df[keep].copy()


def clean_redfin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to a mapped Redfin DataFrame.
    Returns a cleaned DataFrame ready for database insertion.
    """
    original_count = len(df)
    log = []

    # ── Cast numeric columns ─────────────────────────────────────────────────
    numeric_cols = {
        "sale_price": int,
        "beds": "Int64",       # nullable integer
        "sqft": int,
        "lot_sqft": "Int64",
        "year_built": "Int64",
        "days_on_market": "Int64",
        "hoa": float,
        "latitude": float,
        "longitude": float,
        "baths": float,
    }
    for col, dtype in numeric_cols.items():
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if dtype not in ("Int64", float):
            df[col] = df[col].astype(dtype, errors="ignore")

    # ── Parse dates ──────────────────────────────────────────────────────────
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # ── ZIP: keep as string, strip spaces ────────────────────────────────────
    df["zip"] = df["zip"].astype(str).str.strip().str.zfill(5)

    # ── HOA: fill missing with 0 ─────────────────────────────────────────────
    df["hoa"] = df["hoa"].fillna(0).astype(int)

    # ── Drop rows missing required fields ────────────────────────────────────
    required = ["sale_price", "sqft", "beds", "sale_date", "address"]
    before = len(df)
    df = df.dropna(subset=required)
    dropped = before - len(df)
    if dropped:
        log.append(f"  Dropped {dropped} rows missing required fields ({required})")

    # ── Filter to residential property types ─────────────────────────────────
    before = len(df)
    df = df[df["property_type"].isin(KEEP_PROPERTY_TYPES)]
    dropped = before - len(df)
    if dropped:
        log.append(f"  Dropped {dropped} non-residential rows")

    # ── Price outliers ───────────────────────────────────────────────────────
    before = len(df)
    df = df[df["sale_price"] >= MIN_SALE_PRICE]
    dropped = before - len(df)
    if dropped:
        log.append(f"  Dropped {dropped} rows with sale_price < ${MIN_SALE_PRICE:,}")

    # ── Sqft outliers ────────────────────────────────────────────────────────
    before = len(df)
    df = df[(df["sqft"] >= MIN_SQFT) & (df["sqft"] <= MAX_SQFT)]
    dropped = before - len(df)
    if dropped:
        log.append(f"  Dropped {dropped} rows with sqft outside [{MIN_SQFT}, {MAX_SQFT}]")

    # ── Price per sqft outliers ──────────────────────────────────────────────
    ppsf = df["sale_price"] / df["sqft"]
    before = len(df)
    df = df[(ppsf >= MIN_PRICE_PER_SQFT) & (ppsf <= MAX_PRICE_PER_SQFT)]
    dropped = before - len(df)
    if dropped:
        log.append(f"  Dropped {dropped} rows with $/sqft outside [${MIN_PRICE_PER_SQFT}, ${MAX_PRICE_PER_SQFT}]")

    # ── Deduplicate on address + sale_date ───────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["address", "sale_date"], keep="first")
    dropped = before - len(df)
    if dropped:
        log.append(f"  Dropped {dropped} duplicate rows (address + sale_date)")

    # ── Tag data source ──────────────────────────────────────────────────────
    df["data_source"] = "redfin"

    # ── Summary ──────────────────────────────────────────────────────────────
    final_count = len(df)
    print(f"\nCleaning summary:")
    print(f"  Raw rows:    {original_count:,}")
    for entry in log:
        print(entry)
    print(f"  Final rows:  {final_count:,}")

    return df.reset_index(drop=True)


def print_ingest_summary(df: pd.DataFrame) -> None:
    """Print a data quality summary after cleaning."""
    print(f"\nData summary:")
    print(f"  Date range:     {df['sale_date'].min()} → {df['sale_date'].max()}")
    print(f"  Median price:   ${df['sale_price'].median():,.0f}")
    print(f"  Price range:    ${df['sale_price'].min():,} – ${df['sale_price'].max():,}")
    print(f"  Median sqft:    {df['sqft'].median():,.0f}")

    print(f"\nRows per ZIP (post-cleaning):")
    zip_counts = df["zip"].value_counts().sort_index()
    for zip_code, count in zip_counts.items():
        flag = "  ⚠️  LOW SAMPLE — will default to medium/low confidence tier" if count < 50 else ""
        print(f"  {zip_code}: {count} transactions{flag}")

    print(f"\nProperty type breakdown:")
    for pt, count in df["property_type"].value_counts().items():
        print(f"  {pt}: {count}")


def load_into_db(df: pd.DataFrame, conn: sqlite3.Connection) -> int:
    """
    Insert cleaned transactions into the database.
    Skips rows that already exist (address + sale_date match).
    Returns count of rows inserted.
    """
    schema_cols = [
        "address", "city", "zip", "sale_price", "sale_date",
        "beds", "baths", "sqft", "lot_sqft", "year_built", "hoa",
        "days_on_market", "property_type", "latitude", "longitude", "data_source",
    ]
    df_to_insert = df[schema_cols].copy()

    # Check for existing records to avoid duplicates on re-run
    existing = pd.read_sql(
        "SELECT address, sale_date FROM transactions WHERE data_source = 'redfin'",
        conn
    )
    if not existing.empty:
        existing_keys = set(zip(existing["address"], existing["sale_date"]))
        mask = ~df_to_insert.apply(
            lambda row: (row["address"], row["sale_date"]) in existing_keys, axis=1
        )
        df_to_insert = df_to_insert[mask]
        print(f"\n  Skipped {(~mask).sum()} already-loaded rows")

    if df_to_insert.empty:
        print("  No new rows to insert.")
        return 0

    df_to_insert.to_sql("transactions", conn, if_exists="append", index=False)
    conn.commit()
    return len(df_to_insert)


def ingest_redfin(file_path: str | Path) -> None:
    """
    Full pipeline: load CSV → clean → insert into database.
    This is the main entry point for Redfin data ingestion.
    """
    file_path = Path(file_path)
    print(f"Ingesting: {file_path.name}")

    # Ensure DB and tables exist
    init_db()

    # Load and clean
    raw = load_redfin_csv(file_path)
    print(f"Loaded {len(raw):,} raw rows from CSV")

    mapped = map_columns(raw)
    cleaned = clean_redfin(mapped)

    print_ingest_summary(cleaned)

    # Insert
    conn = get_connection()
    inserted = load_into_db(cleaned, conn)
    conn.close()

    print(f"\n✅ Inserted {inserted:,} new rows into transactions table")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/ingest.py data/raw/your_redfin_file.csv")
        sys.exit(1)

    ingest_redfin(sys.argv[1])
