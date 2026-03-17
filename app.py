"""
Real Estate Analytics — Local Control Panel
Flask app wrapping the existing pipeline: model, predictions, one-pager generation.
Run: python app.py
Open: http://localhost:5050
"""

import sys
import json
import os
import sqlite3
import math
import subprocess
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file

# ---------------------------------------------------------------------------
# Config — update these paths to match your local project
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
DB_PATH = PROJECT_ROOT / "re_analytics.db"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
COEFFICIENTS_FILE = MODELS_DIR / "coefficients.json"
CONFIG_FILE = PROJECT_ROOT / "config.json"

OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Default configuration (persisted to config.json)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "agent": {
        "name": "Dmitriy",
        "phone": "(916) 555-0100",
        "email": "dmitriy@example.com",
        "tagline": "Data-Driven Real Estate",
        "license_number": "",
        "brokerage": ""
    },
    "onepager": {
        "primary_color": "#1a2744",
        "accent_color": "#c9953c",
        "background_color": "#fafaf8",
        "text_color": "#2c2c2c",
        "heading_font": "Playfair Display",
        "body_font": "DM Sans",
        "show_qr_code": True,
        "show_trend_chart": True,
        "show_neighborhood_stats": True,
        "show_confidence_interval": True,
        "show_comp_table": True,
        "base_url": "http://localhost:5050/value",
        "footer_text": "Estimates based on local transaction data. Not an appraisal.",
        "cta_text": "Schedule a walkthrough for a refined estimate"
    },
    "model": {
        "confidence_tiers": {
            "high": {"max_mape": 7.0, "label": "High Confidence"},
            "medium": {"max_mape": 12.0, "label": "Medium Confidence"},
            "low": {"max_mape": 100.0, "label": "Low Confidence"}
        },
        "features_enabled": {
            "sqft": True,
            "lot_sqft_log": True,
            "beds": True,
            "baths": True,
            "age": True,
            "hoa": True,
            "zip_dummies": True,
            "mortgage_rate": True
        },
        "min_zip_samples": 30,
        "test_split": 0.2,
        "prediction_interval_pct": 90
    },
    "lead_scoring": {
        "weights": {
            "tenure": 0.25,
            "rate_lock_in": 0.20,
            "value_gap": 0.25,
            "market_temperature": 0.15,
            "property_age": 0.15
        },
        "tiers": {
            "A": {"min_score": 75, "label": "High Propensity"},
            "B": {"min_score": 50, "label": "Medium Propensity"},
            "C": {"min_score": 25, "label": "Low Propensity"},
            "D": {"min_score": 0, "label": "Not a Lead"}
        },
        "zip_filter": ["95765", "95677", "95678", "95661", "95746", "95650"],
        "tenure_sweet_spot_years": 7,
        "rate_lock_threshold": 4.5,
        "value_gap_significant_pct": 15,
        "property_age_replacement_cycle": 18
    }
}


def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            saved = json.load(f)
        # Merge with defaults so new keys are always present
        merged = _deep_merge(DEFAULT_CONFIG, saved)
        return merged
    return DEFAULT_CONFIG.copy()


def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def _deep_merge(defaults, overrides):
    """Recursively merge overrides into defaults."""
    result = defaults.copy()
    for k, v in overrides.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def query_db(sql, args=(), one=False):
    conn = get_db()
    cur = conn.execute(sql, args)
    rv = [dict(row) for row in cur.fetchall()]
    conn.close()
    return (rv[0] if rv else None) if one else rv


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def load_coefficients():
    if COEFFICIENTS_FILE.exists():
        with open(COEFFICIENTS_FILE) as f:
            return json.load(f)
    return None


def get_model_accuracy():
    """Read per-ZIP accuracy from the model_accuracy table."""
    try:
        return query_db("SELECT * FROM model_accuracy ORDER BY mape ASC")
    except Exception:
        return []


def get_latest_mortgage_rate():
    """Fetch the most recent mortgage rate from enriched transactions."""
    try:
        row = query_db(
            "SELECT mortgage_rate FROM transactions "
            "WHERE mortgage_rate IS NOT NULL "
            "ORDER BY sale_date DESC LIMIT 1",
            one=True
        )
        return float(row["mortgage_rate"]) if row else None
    except Exception:
        return None


def _coef_val(coefs, key, default=0):
    """Extract numeric coefficient from nested coefficients.json structure.

    Each entry in the coefficients dict may be a plain number or a dict like
    {"coef": 259.66, "ci_lower": ..., "ci_upper": ..., "pvalue": ...}.
    This helper handles both forms transparently.
    """
    v = coefs.get(key, default)
    if isinstance(v, dict):
        return v.get("coef", default)
    return v


def compute_prediction(features, coefficients, config):
    """
    Replicate the prediction logic from predict.py.
    features: dict with sqft, lot_sqft, beds, baths, year_built, hoa, zip
    coefficients: dict from coefficients.json
    Returns: dict with estimate, interval, tier, explanation
    """
    if not coefficients:
        return {"error": "No model coefficients loaded. Train the model first."}

    coefs = coefficients.get("coefficients", coefficients)
    intercept = _coef_val(coefs, "const", _coef_val(coefs, "intercept", 0))

    estimate = intercept
    breakdown = [{"feature": "Base (intercept)", "value": "", "contribution": intercept}]

    # sqft
    if config["model"]["features_enabled"].get("sqft", True):
        sqft_coef = _coef_val(coefs, "sqft")
        contrib = sqft_coef * features["sqft"]
        estimate += contrib
        breakdown.append({"feature": "Square Feet", "value": features["sqft"], "contribution": round(contrib, 2)})

    # lot_sqft_log
    if config["model"]["features_enabled"].get("lot_sqft_log", True) and features.get("lot_sqft", 0) > 0:
        lot_log_coef = _coef_val(coefs, "lot_sqft_log")
        lot_log_val = math.log(features.get("lot_sqft", 1))
        contrib = lot_log_coef * lot_log_val
        estimate += contrib
        breakdown.append({"feature": "Lot Size (log)", "value": f"{features.get('lot_sqft', 0):,} sqft", "contribution": round(contrib, 2)})

    # beds
    if config["model"]["features_enabled"].get("beds", True):
        beds_coef = _coef_val(coefs, "beds")
        contrib = beds_coef * features["beds"]
        estimate += contrib
        breakdown.append({"feature": "Bedrooms", "value": features["beds"], "contribution": round(contrib, 2)})

    # baths
    if config["model"]["features_enabled"].get("baths", True):
        baths_coef = _coef_val(coefs, "baths")
        contrib = baths_coef * features["baths"]
        estimate += contrib
        breakdown.append({"feature": "Bathrooms", "value": features["baths"], "contribution": round(contrib, 2)})

    # age
    if config["model"]["features_enabled"].get("age", True):
        age_coef = _coef_val(coefs, "age")
        age_val = 2026 - features.get("year_built", 2000)
        contrib = age_coef * age_val
        estimate += contrib
        breakdown.append({"feature": "Age", "value": f"{age_val} years", "contribution": round(contrib, 2)})

    # hoa
    if config["model"]["features_enabled"].get("hoa", True):
        hoa_coef = _coef_val(coefs, "hoa")
        contrib = hoa_coef * features.get("hoa", 0)
        estimate += contrib
        breakdown.append({"feature": "HOA/month", "value": f"${features.get('hoa', 0)}", "contribution": round(contrib, 2)})

    # mortgage_rate
    if config["model"]["features_enabled"].get("mortgage_rate", True) and "mortgage_rate" in coefs:
        rate_coef = _coef_val(coefs, "mortgage_rate")
        rate_val = features.get("mortgage_rate")
        if rate_val is None:
            rate_val = get_latest_mortgage_rate() or 6.5
        contrib = rate_coef * rate_val
        estimate += contrib
        breakdown.append({"feature": "Mortgage Rate", "value": f"{rate_val}%", "contribution": round(contrib, 2)})

    # zip dummies
    if config["model"]["features_enabled"].get("zip_dummies", True):
        zip_key = f"zip_{features['zip']}"
        zip_coef = _coef_val(coefs, zip_key)
        if zip_coef != 0:
            estimate += zip_coef
            breakdown.append({"feature": f"ZIP {features['zip']}", "value": "", "contribution": round(zip_coef, 2)})

    # Get per-ZIP RMSE for confidence interval
    accuracy = query_db("SELECT * FROM model_accuracy WHERE zip = ?", (features["zip"],), one=True)
    rmse = accuracy["rmse"] if accuracy else 190620  # fallback to global
    mape = accuracy["mape"] if accuracy else 12.5

    # Determine tier from config thresholds
    tiers = config["model"]["confidence_tiers"]
    if mape <= tiers["high"]["max_mape"]:
        tier = "high"
    elif mape <= tiers["medium"]["max_mape"]:
        tier = "medium"
    else:
        tier = "low"

    # Prediction interval (default 90%)
    z_score = 1.645  # 90% interval
    interval_pct = config["model"].get("prediction_interval_pct", 90)
    if interval_pct == 95:
        z_score = 1.96
    elif interval_pct == 80:
        z_score = 1.28

    low = estimate - z_score * rmse
    high = estimate + z_score * rmse

    return {
        "estimate": round(estimate, 0),
        "interval_low": round(max(low, 0), 0),
        "interval_high": round(high, 0),
        "rmse": round(rmse, 0),
        "mape": round(mape, 1),
        "tier": tier,
        "tier_label": tiers[tier]["label"],
        "breakdown": breakdown,
        "interval_pct": interval_pct
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    config = load_config()
    coefficients = load_coefficients()
    accuracy = get_model_accuracy()

    # Get ZIP summary stats
    zip_stats = query_db("""
        SELECT zip, COUNT(*) as n,
               ROUND(AVG(sale_price)) as avg_price,
               ROUND(AVG(sqft)) as avg_sqft,
               ROUND(AVG(days_on_market), 1) as avg_dom
        FROM transactions
        GROUP BY zip
        HAVING COUNT(*) >= 30
        ORDER BY COUNT(*) DESC
    """)

    return render_template("dashboard.html",
                           config=config,
                           coefficients=coefficients,
                           accuracy=accuracy,
                           zip_stats=zip_stats)


@app.route("/api/config", methods=["GET"])
def api_get_config():
    return jsonify(load_config())


@app.route("/api/config", methods=["POST"])
def api_save_config():
    cfg = request.json
    save_config(cfg)
    return jsonify({"status": "ok"})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json
    config = load_config()
    coefficients = load_coefficients()

    features = {
        "sqft": int(data.get("sqft", 1800)),
        "lot_sqft": int(data.get("lot_sqft", 7000)),
        "beds": int(data.get("beds", 3)),
        "baths": float(data.get("baths", 2)),
        "year_built": int(data.get("year_built", 2000)),
        "hoa": int(data.get("hoa", 0)),
        "zip": str(data.get("zip", "95677")),
        "mortgage_rate": float(data["mortgage_rate"]) if data.get("mortgage_rate") else None
    }

    result = compute_prediction(features, coefficients, config)
    return jsonify(result)


@app.route("/api/model/accuracy", methods=["GET"])
def api_model_accuracy():
    return jsonify(get_model_accuracy())


@app.route("/api/model/coefficients", methods=["GET"])
def api_model_coefficients():
    coefs = load_coefficients()
    return jsonify(coefs or {})


@app.route("/api/stats/zip/<zip_code>", methods=["GET"])
def api_zip_stats(zip_code):
    stats = query_db("""
        SELECT
            COUNT(*) as total_transactions,
            ROUND(AVG(sale_price)) as avg_price,
            ROUND(MIN(sale_price)) as min_price,
            ROUND(MAX(sale_price)) as max_price,
            ROUND(AVG(sqft)) as avg_sqft,
            ROUND(AVG(days_on_market), 1) as avg_dom,
            ROUND(AVG(beds), 1) as avg_beds,
            ROUND(AVG(baths), 1) as avg_baths
        FROM transactions WHERE zip = ?
    """, (zip_code,), one=True)

    # Quarterly trend
    trend = query_db("""
        SELECT
            substr(sale_date, 1, 7) as month,
            COUNT(*) as n,
            ROUND(AVG(sale_price)) as avg_price,
            ROUND(AVG(sale_price / sqft), 2) as avg_ppsf
        FROM transactions
        WHERE zip = ?
        GROUP BY substr(sale_date, 1, 7)
        ORDER BY month
    """, (zip_code,))

    return jsonify({"stats": stats, "trend": trend})


@app.route("/api/transactions/recent", methods=["GET"])
def api_recent_transactions():
    zip_code = request.args.get("zip", None)
    limit = int(request.args.get("limit", 20))

    if zip_code:
        rows = query_db("""
            SELECT address, city, zip, sale_price, sale_date, beds, baths, sqft, lot_sqft, year_built, days_on_market
            FROM transactions WHERE zip = ?
            ORDER BY sale_date DESC LIMIT ?
        """, (zip_code, limit))
    else:
        rows = query_db("""
            SELECT address, city, zip, sale_price, sale_date, beds, baths, sqft, lot_sqft, year_built, days_on_market
            FROM transactions
            ORDER BY sale_date DESC LIMIT ?
        """, (limit,))

    return jsonify(rows)


@app.route("/api/onepager/preview", methods=["POST"])
def api_onepager_preview():
    """Return rendered HTML preview of a one-pager (not PDF)."""
    data = request.json
    config = load_config()
    coefficients = load_coefficients()

    features = {
        "sqft": int(data.get("sqft", 1800)),
        "lot_sqft": int(data.get("lot_sqft", 7000)),
        "beds": int(data.get("beds", 3)),
        "baths": float(data.get("baths", 2)),
        "year_built": int(data.get("year_built", 2000)),
        "hoa": int(data.get("hoa", 0)),
        "zip": str(data.get("zip", "95677")),
        "mortgage_rate": float(data["mortgage_rate"]) if data.get("mortgage_rate") else None
    }
    address = data.get("address", "123 Sample Street")
    city = data.get("city", "Rocklin")

    prediction = compute_prediction(features, coefficients, config)

    # Get neighborhood stats
    zip_stats = query_db("""
        SELECT COUNT(*) as n,
               ROUND(AVG(sale_price)) as median_price,
               ROUND(AVG(days_on_market), 1) as avg_dom,
               ROUND(AVG(sale_price / sqft), 2) as avg_ppsf
        FROM transactions WHERE zip = ?
    """, (features["zip"],), one=True)

    html = render_template("onepager_preview.html",
                           config=config,
                           prediction=prediction,
                           features=features,
                           address=address,
                           city=city,
                           zip_stats=zip_stats)
    return html


# ---------------------------------------------------------------------------
# Lead Scoring Routes
# ---------------------------------------------------------------------------
from datetime import datetime as _dt, timedelta as _td


@app.route("/api/lead-scores", methods=["GET"])
def api_lead_scores():
    zip_code = request.args.get("zip")
    tier = request.args.get("tier")
    min_score = request.args.get("min_score", 0, type=float)
    limit = request.args.get("limit", 50, type=int)
    offset = request.args.get("offset", 0, type=int)

    conditions = ["ls.score >= ?"]
    params = [min_score]

    if zip_code:
        conditions.append("p.zip = ?")
        params.append(zip_code)

    if tier:
        conditions.append("json_extract(ls.score_components, '$.tier') = ?")
        params.append(tier.upper())

    where_clause = " AND ".join(conditions)

    sql = f"""
        SELECT
            ls.parcel_id, ls.address, ls.score, ls.score_components, ls.last_scored,
            p.zip, p.owner_name, p.purchase_date, p.assessed_value,
            p.year_built, p.sqft, p.beds, p.baths,
            json_extract(ls.score_components, '$.tier') as tier,
            (SELECT MAX(contacted_at) FROM interactions i WHERE i.address = ls.address) as last_interaction
        FROM lead_scores ls
        LEFT JOIN parcels p ON ls.parcel_id = p.parcel_id OR ls.address = p.address
        WHERE {where_clause}
        ORDER BY ls.score DESC
        LIMIT ? OFFSET ?
    """
    rows = query_db(sql, params + [limit, offset])

    for row in rows:
        if row.get("score_components"):
            try:
                row["score_components"] = json.loads(row["score_components"])
            except Exception:
                pass

    count_row = query_db(
        f"SELECT COUNT(*) as n FROM lead_scores ls LEFT JOIN parcels p ON ls.parcel_id = p.parcel_id OR ls.address = p.address WHERE {where_clause}",
        params, one=True
    )
    total = count_row["n"] if count_row else 0

    return jsonify({"leads": rows, "total": total, "limit": limit, "offset": offset})


@app.route("/api/lead-scores/stats", methods=["GET"])
def api_lead_score_stats():
    tier_counts = query_db("""
        SELECT json_extract(score_components, '$.tier') as tier,
               COUNT(*) as n,
               ROUND(AVG(score), 1) as avg_score
        FROM lead_scores GROUP BY tier ORDER BY avg_score DESC
    """)
    total = sum(r["n"] for r in tier_counts)
    histogram = query_db("""
        SELECT CAST(score / 10 AS INTEGER) * 10 as bucket, COUNT(*) as n
        FROM lead_scores GROUP BY bucket ORDER BY bucket
    """)
    mean_row = query_db("SELECT ROUND(AVG(score), 1) as m FROM lead_scores", one=True)
    return jsonify({
        "tier_counts": tier_counts,
        "total": total,
        "mean_score": mean_row["m"] if mean_row else 0,
        "histogram": histogram,
    })


@app.route("/api/lead-scores/rescore", methods=["POST"])
def api_rescore_leads():
    data = request.json or {}
    zip_code = data.get("zip")

    script = PROJECT_ROOT / "src" / "score_leads.py"
    if not script.exists():
        script = PROJECT_ROOT / "score_leads.py"

    cmd = [sys.executable, str(script)]
    if zip_code:
        cmd += ["--zip", zip_code]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return jsonify({
            "status": "ok" if result.returncode == 0 else "error",
            "stdout": result.stdout[-3000:],
            "stderr": result.stderr[-1000:] if result.stderr else "",
        })
    except subprocess.TimeoutExpired:
        return jsonify({"status": "error", "message": "Timed out after 5 minutes"}), 408
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/interactions", methods=["POST"])
def api_log_interaction():
    data = request.json
    if not data or not data.get("address"):
        return jsonify({"error": "address is required"}), 400

    contacted_at = data.get("contacted_at") or _dt.utcnow().isoformat()
    conn = get_db()
    conn.execute("""
        INSERT INTO interactions (parcel_id, address, contact_type, outcome, notes, contacted_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (data.get("parcel_id"), data["address"], data.get("contact_type"),
          data.get("outcome"), data.get("notes"), contacted_at))

    outcome = data.get("outcome", "")
    if outcome in ("callback", "no_answer"):
        days = 3 if outcome == "callback" else 14
        follow_up_date = (_dt.utcnow() + _td(days=days)).strftime("%Y-%m-%d")
        conn.execute("""
            INSERT INTO follow_ups (parcel_id, address, trigger_reason, follow_up_date, notes)
            VALUES (?, ?, ?, ?, ?)
        """, (data.get("parcel_id"), data["address"],
              "callback_scheduled" if outcome == "callback" else "no_answer_retry",
              follow_up_date, f"Auto-created from {outcome} interaction"))

    conn.commit()
    conn.close()
    return jsonify({"status": "ok"})


@app.route("/api/interactions", methods=["GET"])
def api_get_interactions():
    address = request.args.get("address")
    limit = request.args.get("limit", 20, type=int)
    if address:
        rows = query_db("SELECT * FROM interactions WHERE address LIKE ? ORDER BY contacted_at DESC LIMIT ?",
                        (f"%{address}%", limit))
    else:
        rows = query_db("SELECT * FROM interactions ORDER BY contacted_at DESC LIMIT ?", (limit,))
    return jsonify(rows)


@app.route("/api/config/lead-scoring", methods=["POST"])
def api_save_lead_scoring_config():
    data = request.json
    cfg = load_config()
    ls = cfg.get("lead_scoring", {})

    if "weights" in data:
        w = data["weights"]
        total_w = sum(w.values())
        if total_w > 0:
            data["weights"] = {k: round(v / total_w, 4) for k, v in w.items()}
        ls["weights"] = data["weights"]

    for key in ("tiers", "zip_filter"):
        if key in data:
            ls[key] = data[key]
    for key in ("tenure_sweet_spot_years", "rate_lock_threshold",
                "value_gap_significant_pct", "property_age_replacement_cycle"):
        if key in data:
            ls[key] = data[key]

    cfg["lead_scoring"] = ls
    save_config(cfg)
    return jsonify({"status": "ok", "lead_scoring": ls})


@app.route("/api/lead-scores/addresses", methods=["GET"])
def api_lead_addresses():
    q = request.args.get("q", "")
    rows = query_db("""
        SELECT DISTINCT address FROM lead_scores
        WHERE address LIKE ? ORDER BY score DESC LIMIT 20
    """, (f"%{q}%",))
    return jsonify([r["address"] for r in rows])


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n  Real Estate Analytics Control Panel")
    print(f"  Database: {DB_PATH}")
    print(f"  Open: http://localhost:5050\n")
    app.run(debug=True, port=5050, host="0.0.0.0")