"""
predict.py — Single-property price prediction with confidence interval.

Usage:
    python src/predict.py                         # runs built-in test cases
    python src/predict.py --interactive           # prompts for property details

Loads coefficients from models/coefficients.json and confidence tier
from model_accuracy table, then outputs a point estimate + interval
and a client-ready sentence.
"""

import sys
import json
import sqlite3
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from db import get_connection

COEFFICIENTS_PATH = Path(__file__).parent.parent / "models" / "coefficients.json"

# Prediction interval coverage — 90% by default
PI_Z = 1.645


@dataclass
class PropertyFeatures:
    """Input features for a single property prediction."""
    sqft: int
    lot_sqft: Optional[int]
    beds: int
    baths: float
    year_built: int
    hoa: int
    zip_code: str
    sale_year: int = 2025   # year to price as-of


@dataclass
class PredictionResult:
    """Output from a single property prediction."""
    address: str
    zip_code: str
    point_estimate: int
    interval_low: int
    interval_high: int
    confidence_tier: str
    n_transactions: int
    mape: float
    client_sentence: str


# ── Load model artifacts ──────────────────────────────────────────────────────

def load_coefficients() -> dict:
    """Load saved model coefficients from JSON."""
    if not COEFFICIENTS_PATH.exists():
        raise FileNotFoundError(
            f"No coefficients file found at {COEFFICIENTS_PATH}. "
            "Run model.py first."
        )
    with open(COEFFICIENTS_PATH) as f:
        return json.load(f)


def load_zip_accuracy(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load per-ZIP accuracy metrics from the database."""
    return pd.read_sql("SELECT * FROM model_accuracy", conn)


# ── Feature construction ──────────────────────────────────────────────────────

def build_feature_vector(
    prop: PropertyFeatures,
    coef_data: dict,
) -> dict:
    """
    Build a feature dict matching the model's training features.
    ZIP dummies are constructed from the feature names in coefficients.json.
    """
    age = prop.sale_year - prop.year_built
    median_lot = 7200   # fallback if lot_sqft missing — roughly median for area
    lot = prop.lot_sqft if prop.lot_sqft else median_lot
    lot_sqft_log = np.log1p(lot)

    features = {
        "const": 1.0,
        "sqft": float(prop.sqft),
        "lot_sqft_log": lot_sqft_log,
        "beds": float(prop.beds),
        "baths": float(prop.baths),
        "age": float(age),
        "hoa": float(prop.hoa),
    }

    # Add ZIP dummies — set all to 0, then flip the matching one to 1
    zip_feature_names = [k for k in coef_data["coefficients"] if k.startswith("zip_")]
    for zf in zip_feature_names:
        features[zf] = 0.0

    zip_key = f"zip_{prop.zip_code}"
    if zip_key in features:
        features[zip_key] = 1.0
    # If zip_key not in dummies, it's the base/reference ZIP — all zeros is correct

    return features


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(
    prop: PropertyFeatures,
    address: str = "",
    conn: Optional[sqlite3.Connection] = None,
) -> PredictionResult:
    """
    Predict sale price for a single property.

    Returns a PredictionResult with point estimate, prediction interval,
    confidence tier, and a client-ready sentence.
    """
    coef_data = load_coefficients()

    if conn is None:
        conn = get_connection()
    zip_accuracy = load_zip_accuracy(conn)

    # Build feature vector
    features = build_feature_vector(prop, coef_data)

    # Compute point estimate
    point = sum(
        coef_data["coefficients"][k]["coef"] * v
        for k, v in features.items()
        if k in coef_data["coefficients"]
    )
    point = max(50_000, int(round(point, -3)))   # floor at $50K, round to $1K

    # ZIP accuracy — look up before interval so we can use per-ZIP RMSE
    zip_row = zip_accuracy[zip_accuracy["zip"] == prop.zip_code]
    if not zip_row.empty:
        tier  = zip_row.iloc[0]["confidence_tier"]
        n     = int(zip_row.iloc[0]["n_transactions"])
        mape  = float(zip_row.iloc[0]["mape"])
        rmse  = float(zip_row.iloc[0]["rmse"])
    else:
        tier, n, mape = "low", 0, 99.0
        rmse = coef_data["overall_metrics"]["rmse"]  # fallback to global

    # Prediction interval using per-ZIP RMSE (tighter and more honest than global)
    margin = PI_Z * rmse
    interval_low  = max(50_000, int(round(point - margin, -3)))
    interval_high = int(round(point + margin, -3))

    # Client-ready sentence
    sentence = _build_client_sentence(prop.zip_code, point, interval_low, interval_high, tier, n)

    return PredictionResult(
        address=address,
        zip_code=prop.zip_code,
        point_estimate=point,
        interval_low=interval_low,
        interval_high=interval_high,
        confidence_tier=tier,
        n_transactions=n,
        mape=mape,
        client_sentence=sentence,
    )


def _build_client_sentence(
    zip_code: str,
    point: int,
    low: int,
    high: int,
    tier: str,
    n: int,
) -> str:
    """Generate a confidence-tier-appropriate client-facing sentence."""
    if tier == "high":
        return (
            f"Based on {n} recent sales in {zip_code}, our model estimates "
            f"your home's market value at ${point:,}, with most similar homes "
            f"selling between ${low:,} and ${high:,}."
        )
    elif tier == "medium":
        return (
            f"Based on {n} recent sales in {zip_code}, homes with similar "
            f"characteristics have been selling between ${low:,} and ${high:,}."
        )
    else:
        return (
            f"Based on recent sales activity in {zip_code}, I'd want to see "
            f"your home in person to give you an accurate number — "
            f"there's meaningful variation in this market that a drive-by estimate won't capture."
        )


# ── Pretty print ──────────────────────────────────────────────────────────────

def print_result(result: PredictionResult) -> None:
    """Print a formatted prediction result to stdout."""
    tier_label = {"high": "✅ High", "medium": "⚠️  Medium", "low": "🔴 Low"}
    print(f"\n{'─'*50}")
    if result.address:
        print(f"  {result.address}")
    print(f"  ZIP:              {result.zip_code}")
    print(f"  Estimate:         ${result.point_estimate:,}")
    print(f"  90% interval:     ${result.interval_low:,} – ${result.interval_high:,}")
    print(f"  Confidence tier:  {tier_label.get(result.confidence_tier, result.confidence_tier)}")
    print(f"  Model MAPE:       {result.mape:.1f}%  (based on {result.n_transactions} test transactions)")
    print(f"\n  Client sentence:")
    print(f"  \"{result.client_sentence}\"")
    print(f"{'─'*50}")


# ── Interactive mode ──────────────────────────────────────────────────────────

def interactive_predict() -> None:
    """Prompt user for property details and print a prediction."""
    print("\nProperty details (press Enter to skip optional fields)")
    address   = input("  Address: ").strip()
    zip_code  = input("  ZIP code: ").strip()
    sqft      = int(input("  Square feet: ").strip())
    beds      = int(input("  Bedrooms: ").strip())
    baths     = float(input("  Bathrooms: ").strip())
    year_built = int(input("  Year built: ").strip())

    lot_raw = input("  Lot size (sqft, optional): ").strip()
    lot_sqft = int(lot_raw) if lot_raw else None

    hoa_raw = input("  HOA/month (0 if none): ").strip()
    hoa = int(hoa_raw) if hoa_raw else 0

    prop = PropertyFeatures(
        sqft=sqft, lot_sqft=lot_sqft, beds=beds, baths=baths,
        year_built=year_built, hoa=hoa, zip_code=zip_code,
    )
    result = predict(prop, address=address)
    print_result(result)


# ── Test cases ────────────────────────────────────────────────────────────────

def run_test_cases() -> None:
    """Run a set of representative test properties across the 6 ZIPs."""
    test_properties = [
        ("1234 Oak Creek Dr, Granite Bay",   PropertyFeatures(2400, 8500,  4, 3.0, 2005, 0,   "95746")),
        ("567 Willowbrook Ln, Roseville",    PropertyFeatures(1850, 5200,  3, 2.0, 1998, 120, "95678")),
        ("890 Sierra Vista Way, Rocklin",    PropertyFeatures(2100, 7000,  4, 2.5, 2010, 0,   "95677")),
        ("321 Vintage Oaks Ct, Roseville",   PropertyFeatures(3200, 9800,  5, 3.5, 2015, 85,  "95765")),
        ("45 Gold Hill Rd, Loomis",          PropertyFeatures(3800, 43560, 4, 3.0, 1990, 0,   "95650")),
        ("789 Parkside Ave, Roseville",      PropertyFeatures(1600, 4800,  3, 2.0, 1988, 0,   "95661")),
    ]

    conn = get_connection()
    print(f"\nRunning {len(test_properties)} test predictions...\n")

    for address, prop in test_properties:
        result = predict(prop, address=address, conn=conn)
        print_result(result)

    conn.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict property value")
    parser.add_argument("--interactive", action="store_true",
                        help="Prompt for property details interactively")
    args = parser.parse_args()

    if args.interactive:
        interactive_predict()
    else:
        run_test_cases()