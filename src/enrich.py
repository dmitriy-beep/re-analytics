"""
enrich.py — Enrich the transactions table with external market data.

Data Source 1: FRED 30-year fixed mortgage rate (weekly, Thursdays)
Data Source 2: Redfin market conditions for Placer County (4-week rolling)

Usage:
    python src/enrich.py              # enrich all transactions
    python src/enrich.py --status     # show enrichment coverage without writing

Idempotent — safe to re-run. Overwrites existing enrichment columns.
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from urllib.request import urlretrieve

sys.path.insert(0, str(Path(__file__).parent))
from db import get_connection, init_db

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FRED_CSV = DATA_DIR / "MORTGAGE30US.csv"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"

REDFIN_MARKET_TSV = DATA_DIR / "redfin_market_conditions.tsv"

# New columns added to the transactions table
ENRICH_COLUMNS = {
    "mortgage_rate":           "REAL",
    "weeks_of_supply":         "REAL",
    "median_days_on_market":   "REAL",
    "sale_to_list_ratio":      "REAL",
    "pct_listings_price_drop": "REAL",
    "off_market_in_two_weeks": "REAL",
}


# ── Schema migration ─────────────────────────────────────────────────────────

def ensure_columns(conn) -> None:
    """Add enrichment columns to the transactions table if they don't exist."""
    cursor = conn.cursor()
    existing = {row[1] for row in cursor.execute("PRAGMA table_info(transactions)").fetchall()}

    for col, dtype in ENRICH_COLUMNS.items():
        if col not in existing:
            cursor.execute(f"ALTER TABLE transactions ADD COLUMN {col} {dtype}")
            print(f"  Added column: transactions.{col} ({dtype})")

    conn.commit()


# ── FRED Mortgage Rates ──────────────────────────────────────────────────────

def download_fred_if_missing() -> Path:
    """Download the FRED MORTGAGE30US CSV if not already present."""
    if FRED_CSV.exists():
        print(f"  FRED CSV already exists: {FRED_CSV.name}")
        return FRED_CSV

    print(f"  Downloading FRED mortgage rates → {FRED_CSV.name} ...")
    urlretrieve(FRED_URL, FRED_CSV)
    print(f"  Downloaded {FRED_CSV.stat().st_size:,} bytes")
    return FRED_CSV


def load_fred_rates() -> pd.DataFrame:
    """
    Load FRED weekly mortgage rates.
    Returns DataFrame with columns: [date, mortgage_rate]
    where date is the Thursday publication date.
    """
    download_fred_if_missing()
    df = pd.read_csv(FRED_CSV)
    df.columns = ["date", "mortgage_rate"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # FRED uses '.' for missing/unavailable values
    df["mortgage_rate"] = pd.to_numeric(df["mortgage_rate"], errors="coerce")
    df = df.dropna(subset=["date", "mortgage_rate"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  FRED rates: {len(df)} weekly observations, "
          f"{df['date'].min().date()} → {df['date'].max().date()}")
    return df


def join_mortgage_rate(transactions: pd.DataFrame, fred: pd.DataFrame) -> pd.Series:
    """
    For each transaction, find the nearest Thursday on or before sale_date.
    Uses pandas merge_asof for efficient sorted join.
    """
    txn = transactions[["id", "sale_date"]].copy()
    txn["sale_dt"] = pd.to_datetime(txn["sale_date"], errors="coerce")
    txn = txn.sort_values("sale_dt")

    fred_sorted = fred.rename(columns={"date": "fred_date"}).sort_values("fred_date")

    merged = pd.merge_asof(
        txn, fred_sorted,
        left_on="sale_dt", right_on="fred_date",
        direction="backward",  # nearest Thursday on or before
    )
    # Re-index to match original transaction order
    merged = merged.set_index("id")
    return merged["mortgage_rate"]


# ── Redfin Market Conditions ─────────────────────────────────────────────────

def load_redfin_market() -> pd.DataFrame:
    """
    Load Redfin market conditions TSV, filtered to:
      - DURATION == '4 weeks'
      - REGION_NAME == 'Placer County, CA'
    Returns DataFrame with date range and market metrics.
    """
    if not REDFIN_MARKET_TSV.exists():
        raise FileNotFoundError(
            f"Redfin market conditions file not found: {REDFIN_MARKET_TSV}\n"
            "Download from Redfin Data Center and place at that path."
        )

    df = pd.read_csv(REDFIN_MARKET_TSV, sep="\t", dtype=str)
    df.columns = df.columns.str.strip()

    # Normalize column names to uppercase for matching
    col_map = {c: c.upper() for c in df.columns}
    df = df.rename(columns=col_map)

    # Filter
    if "DURATION" in df.columns:
        df = df[df["DURATION"].str.strip() == "4 weeks"]
    if "REGION_NAME" in df.columns:
        df = df[df["REGION_NAME"].str.strip() == "Placer County, CA"]

    if df.empty:
        raise ValueError(
            "No rows match DURATION='4 weeks' AND REGION_NAME='Placer County, CA'. "
            "Check the TSV file contents."
        )

    # Parse dates
    df["PERIOD_BEGIN"] = pd.to_datetime(df["PERIOD_BEGIN"], errors="coerce")
    df["PERIOD_END"] = pd.to_datetime(df["PERIOD_END"], errors="coerce")

    # Parse numeric columns
    metric_cols = [
        "WEEKS_OF_SUPPLY", "MEDIAN_DAYS_ON_MARKET",
        "AVERAGE_SALE_TO_LIST_RATIO", "PERCENT_ACTIVE_LISTINGS_WITH_PRICE_DROPS",
        "OFF_MARKET_IN_TWO_WEEKS",
    ]
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["PERIOD_BEGIN", "PERIOD_END"])
    df = df.sort_values("PERIOD_BEGIN").reset_index(drop=True)

    print(f"  Redfin market: {len(df)} periods, "
          f"{df['PERIOD_BEGIN'].min().date()} → {df['PERIOD_END'].max().date()}")
    return df


def join_market_conditions(transactions: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """
    For each transaction, find the market period where
    PERIOD_BEGIN <= sale_date <= PERIOD_END.

    Returns a DataFrame indexed by transaction id with market columns.
    """
    txn = transactions[["id", "sale_date"]].copy()
    txn["sale_dt"] = pd.to_datetime(txn["sale_date"], errors="coerce")

    # Column mapping: Redfin name → our schema name
    col_remap = {
        "WEEKS_OF_SUPPLY": "weeks_of_supply",
        "MEDIAN_DAYS_ON_MARKET": "median_days_on_market",
        "AVERAGE_SALE_TO_LIST_RATIO": "sale_to_list_ratio",
        "PERCENT_ACTIVE_LISTINGS_WITH_PRICE_DROPS": "pct_listings_price_drop",
        "OFF_MARKET_IN_TWO_WEEKS": "off_market_in_two_weeks",
    }

    results = {col: pd.Series(dtype=float, index=txn.index) for col in col_remap.values()}

    # Build an interval lookup — for each transaction, find matching period
    for idx, row in txn.iterrows():
        if pd.isna(row["sale_dt"]):
            continue
        mask = (market["PERIOD_BEGIN"] <= row["sale_dt"]) & (row["sale_dt"] <= market["PERIOD_END"])
        matches = market[mask]
        if not matches.empty:
            # Take the most recent matching period
            match = matches.iloc[-1]
            for src, dst in col_remap.items():
                if src in match.index:
                    results[dst].iloc[idx] = match[src]

    result_df = pd.DataFrame(results)
    result_df["id"] = txn["id"].values
    result_df = result_df.set_index("id")
    return result_df


# ── Write enrichment to DB ───────────────────────────────────────────────────

def write_enrichment(conn, transactions: pd.DataFrame,
                     mortgage_rates: pd.Series,
                     market_data: pd.DataFrame) -> None:
    """
    Batch-update the transactions table with enrichment columns.
    Uses a single UPDATE per row for atomicity.
    """
    cursor = conn.cursor()
    updated = 0

    for _, row in transactions.iterrows():
        txn_id = row["id"]

        rate = mortgage_rates.get(txn_id)
        rate = float(rate) if pd.notna(rate) else None

        market_vals = {}
        for col in ["weeks_of_supply", "median_days_on_market", "sale_to_list_ratio",
                     "pct_listings_price_drop", "off_market_in_two_weeks"]:
            val = market_data[col].get(txn_id) if txn_id in market_data.index else None
            market_vals[col] = float(val) if pd.notna(val) else None

        cursor.execute("""
            UPDATE transactions SET
                mortgage_rate           = ?,
                weeks_of_supply         = ?,
                median_days_on_market   = ?,
                sale_to_list_ratio      = ?,
                pct_listings_price_drop = ?,
                off_market_in_two_weeks = ?
            WHERE id = ?
        """, (
            rate,
            market_vals["weeks_of_supply"],
            market_vals["median_days_on_market"],
            market_vals["sale_to_list_ratio"],
            market_vals["pct_listings_price_drop"],
            market_vals["off_market_in_two_weeks"],
            txn_id,
        ))
        updated += 1

    conn.commit()
    return updated


# ── Status report ────────────────────────────────────────────────────────────

def print_status(conn) -> None:
    """Print enrichment coverage stats."""
    total = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]

    stats = {
        "mortgage_rate":           conn.execute("SELECT COUNT(*) FROM transactions WHERE mortgage_rate IS NOT NULL").fetchone()[0],
        "weeks_of_supply":         conn.execute("SELECT COUNT(*) FROM transactions WHERE weeks_of_supply IS NOT NULL").fetchone()[0],
        "median_days_on_market":   conn.execute("SELECT COUNT(*) FROM transactions WHERE median_days_on_market IS NOT NULL").fetchone()[0],
        "sale_to_list_ratio":      conn.execute("SELECT COUNT(*) FROM transactions WHERE sale_to_list_ratio IS NOT NULL").fetchone()[0],
        "pct_listings_price_drop": conn.execute("SELECT COUNT(*) FROM transactions WHERE pct_listings_price_drop IS NOT NULL").fetchone()[0],
        "off_market_in_two_weeks": conn.execute("SELECT COUNT(*) FROM transactions WHERE off_market_in_two_weeks IS NOT NULL").fetchone()[0],
    }

    print(f"\n── Enrichment coverage ({total:,} total transactions) ──")
    for col, filled in stats.items():
        pct = filled / total * 100 if total > 0 else 0
        print(f"  {col:<30} {filled:>5,} / {total:,}  ({pct:.1f}%)")


# ── Main ─────────────────────────────────────────────────────────────────────

def enrich() -> None:
    """Full enrichment pipeline: download → join → write to DB."""
    init_db()
    conn = get_connection()
    ensure_columns(conn)

    # Load all transactions
    txn = pd.read_sql("SELECT id, sale_date FROM transactions", conn)
    print(f"\nLoaded {len(txn):,} transactions for enrichment")

    # ── Source 1: FRED Mortgage Rates ─────────────────────────────────────
    print("\n── FRED 30-Year Mortgage Rate ──────────────────")
    fred = load_fred_rates()
    mortgage_rates = join_mortgage_rate(txn, fred)
    matched = mortgage_rates.notna().sum()
    print(f"  Matched {matched:,} / {len(txn):,} transactions ({matched/len(txn)*100:.1f}%)")

    # ── Source 2: Redfin Market Conditions ────────────────────────────────
    print("\n── Redfin Market Conditions (Placer County) ───")
    try:
        market = load_redfin_market()
        market_data = join_market_conditions(txn, market)
        matched = market_data["weeks_of_supply"].notna().sum()
        print(f"  Matched {matched:,} / {len(txn):,} transactions ({matched/len(txn)*100:.1f}%)")
    except FileNotFoundError as e:
        print(f"  ⚠️  {e}")
        print("  Skipping market conditions — mortgage rate only.")
        market_data = pd.DataFrame(
            index=txn["id"],
            columns=["weeks_of_supply", "median_days_on_market",
                     "sale_to_list_ratio", "pct_listings_price_drop",
                     "off_market_in_two_weeks"],
        )

    # ── Write to DB ──────────────────────────────────────────────────────
    print("\nWriting enrichment to database...")
    n_updated = write_enrichment(conn, txn, mortgage_rates, market_data)
    print(f"  Updated {n_updated:,} rows")

    # ── Status ───────────────────────────────────────────────────────────
    print_status(conn)
    conn.close()

    print(f"\n✅ Enrichment complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich transactions with market data")
    parser.add_argument("--status", action="store_true",
                        help="Show enrichment coverage without writing")
    args = parser.parse_args()

    if args.status:
        init_db()
        conn = get_connection()
        ensure_columns(conn)
        print_status(conn)
        conn.close()
    else:
        enrich()
