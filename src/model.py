"""
model.py — Hedonic pricing model training and evaluation.

Usage:
    python src/model.py

Trains an OLS regression on the transactions table, evaluates accuracy
per ZIP, assigns confidence tiers, and saves coefficients to models/.
"""


import sys
import json
import sqlite3
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import statsmodels.api as sm

from pathlib import Path
from datetime import datetime
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from db import get_connection, init_db

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

COEFFICIENTS_PATH = MODELS_DIR / "coefficients.json"
RESIDUAL_PLOT_PATH = MODELS_DIR / "residuals.png"

# ZIPs with enough data to model reliably
MIN_ZIP_SAMPLE = 30

# Confidence tier thresholds
TIER_HIGH_MAPE    = 8.0
TIER_MEDIUM_MAPE  = 10.0
TIER_HIGH_N       = 50
TIER_MEDIUM_N     = 30


# ── Data loading ─────────────────────────────────────────────────────────────

def load_transactions(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load all transactions from the database."""
    query = """
        SELECT
            sale_price, sqft, lot_sqft, beds, baths,
            year_built, hoa, zip, sale_date, address
        FROM transactions
        WHERE sale_price IS NOT NULL
          AND sqft IS NOT NULL
          AND beds IS NOT NULL
    """
    df = pd.read_sql(query, conn)
    print(f"Loaded {len(df):,} transactions from database")
    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build model features from raw transaction columns.

    Features:
        sqft            — living area (continuous)
        lot_sqft_log    — log of lot size (log-transforms right skew)
        beds            — bedroom count
        baths           — bathroom count (includes 0.5 increments)
        age             — years since built at time of sale
        hoa             — monthly HOA (0 if none)
        zip_XXXXX       — one-hot encoded ZIP dummies (drop_first=True)
    """
    df = df.copy()

    # Parse sale year from sale_date
    df["sale_year"] = pd.to_datetime(df["sale_date"], errors="coerce").dt.year

    # Age at time of sale
    df["age"] = df["sale_year"] - df["year_built"]
    # Cap extreme ages and drop implausible values
    df = df[(df["age"] >= 0) & (df["age"] <= 120)]

    # Log lot size — fill missing with median before log
    median_lot = df["lot_sqft"].median()
    df["lot_sqft_filled"] = df["lot_sqft"].fillna(median_lot)
    df["lot_sqft_log"] = np.log1p(df["lot_sqft_filled"])

    # HOA — already filled with 0 at ingest
    df["hoa"] = df["hoa"].fillna(0)

    # Drop ZIPs with too few transactions
    zip_counts = df["zip"].value_counts()
    valid_zips = zip_counts[zip_counts >= MIN_ZIP_SAMPLE].index
    dropped_zips = zip_counts[zip_counts < MIN_ZIP_SAMPLE].index.tolist()
    if dropped_zips:
        print(f"  Excluding low-sample ZIPs from model training: {dropped_zips}")
    df = df[df["zip"].isin(valid_zips)]

    # Reset index after ZIP filtering so df and features stay aligned
    df = df.reset_index(drop=True)

    # One-hot encode ZIP (drop_first avoids perfect multicollinearity)
    zip_dummies = pd.get_dummies(df["zip"], prefix="zip", drop_first=True).astype(int)

    feature_cols = ["sqft", "lot_sqft_log", "beds", "baths", "age", "hoa"]
    df_features = pd.concat([df[feature_cols], zip_dummies], axis=1)

    # Drop any remaining NaNs in features
    valid_idx = df_features.dropna().index
    df_features = df_features.loc[valid_idx].reset_index(drop=True)
    df = df.loc[valid_idx].reset_index(drop=True)

    return df, df_features


# ── Train / test split ────────────────────────────────────────────────────────

def train_test_split_by_zip(
    df: pd.DataFrame,
    features: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Stratified 80/20 split by ZIP so every ZIP is represented in both sets.
    Returns (X_train, X_test, y_train, y_test, test_df).
    """
    np.random.seed(random_state)
    test_idx = []

    for zip_code in df["zip"].unique():
        zip_mask = df["zip"] == zip_code
        zip_indices = df[zip_mask].index.tolist()
        n_test = max(1, int(len(zip_indices) * test_size))
        sampled = np.random.choice(zip_indices, size=n_test, replace=False)
        test_idx.extend(sampled)

    test_idx = sorted(set(test_idx))
    train_idx = [i for i in df.index if i not in set(test_idx)]

    X_train = features.loc[train_idx]
    X_test  = features.loc[test_idx]
    y_train = df.loc[train_idx, "sale_price"]
    y_test  = df.loc[test_idx, "sale_price"]
    test_df = df.loc[test_idx]

    return X_train, X_test, y_train, y_test, test_df


# ── Model training ────────────────────────────────────────────────────────────

def train_ols(X_train: pd.DataFrame, y_train: pd.Series) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fit OLS regression with a constant term.
    Uses statsmodels for p-values and confidence intervals.
    """
    X_with_const = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_with_const).fit()
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: pd.Series, y_pred: np.ndarray
) -> dict:
    """Compute R², RMSE, MAE, and MAPE."""
    residuals = y_true - y_pred
    ss_res = (residuals ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2   = 1 - ss_res / ss_tot
    rmse = np.sqrt((residuals ** 2).mean())
    mae  = np.abs(residuals).mean()
    mape = (np.abs(residuals) / y_true * 100).median()  # median APE
    return {"r2": round(r2, 4), "rmse": round(rmse, 2),
            "mae": round(mae, 2), "mape": round(mape, 2)}


def evaluate_by_zip(
    test_df: pd.DataFrame, y_pred: np.ndarray
) -> pd.DataFrame:
    """
    Compute MAPE and RMSE per ZIP on the test set.
    Assign confidence tier based on MAPE and sample size.
    """
    results = []
    pred_series = pd.Series(y_pred, index=test_df.index)

    for zip_code in sorted(test_df["zip"].unique()):
        mask = test_df["zip"] == zip_code
        y_true_zip = test_df.loc[mask, "sale_price"]
        y_pred_zip = pred_series.loc[mask]
        n = len(y_true_zip)

        mape = (np.abs(y_true_zip - y_pred_zip) / y_true_zip * 100).median()
        rmse = np.sqrt(((y_true_zip - y_pred_zip) ** 2).mean())

        # Confidence tier logic
        if mape < TIER_HIGH_MAPE and n >= TIER_HIGH_N:
            tier = "high"
        elif mape <= TIER_MEDIUM_MAPE and n >= TIER_MEDIUM_N:
            tier = "medium"
        else:
            tier = "low"

        results.append({
            "zip": zip_code,
            "n_test": n,
            "mape": round(mape, 2),
            "rmse": round(rmse, 2),
            "confidence_tier": tier,
        })

    return pd.DataFrame(results)


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, path: Path) -> None:
    """Save a predicted vs actual scatter plot."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Predicted vs Actual
    axes[0].scatter(y_pred / 1e6, y_true / 1e6, alpha=0.3, s=10, color="#2563eb")
    max_val = max(y_true.max(), y_pred.max()) / 1e6
    axes[0].plot([0, max_val], [0, max_val], "r--", linewidth=1)
    axes[0].set_xlabel("Predicted ($M)")
    axes[0].set_ylabel("Actual ($M)")
    axes[0].set_title("Predicted vs Actual")

    # Residuals
    residuals_pct = (y_true.values - y_pred) / y_true.values * 100
    axes[1].hist(residuals_pct, bins=50, color="#2563eb", alpha=0.7, edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Residual (%)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Residual plot saved: {path}")


# ── Persistence ───────────────────────────────────────────────────────────────

def save_coefficients(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    feature_names: list[str],
    metrics: dict,
    zip_metrics: pd.DataFrame,
    n_train: int,
) -> None:
    """Save model coefficients and metadata to JSON."""
    params = model.params.to_dict()
    conf_int = model.conf_int().to_dict()

    payload = {
        "trained_at": datetime.now().isoformat(),
        "n_train": n_train,
        "features": feature_names,
        "overall_metrics": metrics,
        "coefficients": {
            name: {
                "coef": round(params[name], 4),
                "ci_lower": round(conf_int[0][name], 4),
                "ci_upper": round(conf_int[1][name], 4),
                "pvalue": round(float(model.pvalues[name]), 6),
            }
            for name in params
        },
        "zip_metrics": zip_metrics.to_dict(orient="records"),
    }

    with open(COEFFICIENTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Coefficients saved: {COEFFICIENTS_PATH}")


def save_accuracy_to_db(zip_metrics: pd.DataFrame, conn: sqlite3.Connection) -> None:
    """Upsert per-ZIP accuracy metrics into the model_accuracy table."""
    cursor = conn.cursor()
    for _, row in zip_metrics.iterrows():
        cursor.execute("""
            INSERT INTO model_accuracy (zip, n_transactions, mape, rmse, confidence_tier, last_updated)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(zip) DO UPDATE SET
                n_transactions = excluded.n_transactions,
                mape           = excluded.mape,
                rmse           = excluded.rmse,
                confidence_tier = excluded.confidence_tier,
                last_updated   = excluded.last_updated
        """, (row["zip"], row["n_test"], row["mape"], row["rmse"], row["confidence_tier"]))
    conn.commit()


# ── Main ──────────────────────────────────────────────────────────────────────

def train_model() -> None:
    """Full training pipeline: load → features → train → evaluate → save."""
    init_db()
    conn = get_connection()

    # Load
    df = load_transactions(conn)

    # Features
    print("\nEngineering features...")
    df, features = engineer_features(df)
    print(f"  Training set: {len(df):,} transactions across {df['zip'].nunique()} ZIPs")

    # Split
    X_train, X_test, y_train, y_test, test_df = train_test_split_by_zip(df, features)
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Train
    print("\nFitting OLS model...")
    model = train_ols(X_train, y_train)

    # Overall metrics
    X_test_const = sm.add_constant(X_test, has_constant="add")
    y_pred = model.predict(X_test_const)
    metrics = compute_metrics(y_test, y_pred)

    print(f"\n── Overall test-set metrics ────────────────")
    print(f"  R²:    {metrics['r2']:.4f}")
    print(f"  RMSE:  ${metrics['rmse']:,.0f}")
    print(f"  MAE:   ${metrics['mae']:,.0f}")
    print(f"  MAPE:  {metrics['mape']:.1f}%")

    # Per-ZIP metrics
    zip_metrics = evaluate_by_zip(test_df, y_pred.values)
    print(f"\n── Per-ZIP accuracy ────────────────────────")
    print(f"  {'ZIP':<8} {'N':>5} {'MAPE':>7} {'RMSE':>10} {'Tier'}")
    print(f"  {'─'*8} {'─'*5} {'─'*7} {'─'*10} {'─'*8}")
    for _, row in zip_metrics.iterrows():
        print(f"  {row['zip']:<8} {row['n_test']:>5} {row['mape']:>6.1f}% "
              f"  ${row['rmse']:>8,.0f}   {row['confidence_tier']}")

    # Coefficient table
    print(f"\n── Coefficients ────────────────────────────")
    print(f"  {'Feature':<20} {'Coef':>12} {'p-value':>10}")
    print(f"  {'─'*20} {'─'*12} {'─'*10}")
    for name, coef in model.params.items():
        pval = model.pvalues[name]
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {name:<20} {coef:>12,.2f} {pval:>10.4f} {sig}")

    # Save
    print("\nSaving outputs...")
    plot_residuals(y_test, y_pred.values, RESIDUAL_PLOT_PATH)
    save_coefficients(model, list(features.columns), metrics, zip_metrics, len(X_train))
    save_accuracy_to_db(zip_metrics, conn)
    conn.close()

    print(f"\n✅ Model training complete")
    print(f"   Coefficients: {COEFFICIENTS_PATH}")
    print(f"   Residual plot: {RESIDUAL_PLOT_PATH}")


if __name__ == "__main__":
    train_model()