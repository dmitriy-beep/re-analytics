"""
score_leads.py — Compute lead scores for all parcels and write to lead_scores table.

5 scoring signals (each 0–100), weighted composite:
  1. Tenure         (0.25) — years owned → equity + life-stage proxy
  2. Rate lock-in   (0.20) — mortgage rate at purchase → friction to sell
  3. Value gap      (0.25) — model estimate vs assessed value → equity gap CTA
  4. Market temp    (0.15) — local supply/demand from Redfin market columns
  5. Property age   (0.15) — age of home → system replacement motivation

Usage:
    python src/score_leads.py                    # score all model-ZIP parcels
    python src/score_leads.py --zip 95746        # score one ZIP
    python src/score_leads.py --stats            # print tier distribution
    python src/score_leads.py --top 20           # print top N leads
"""

import sys
import json
import math
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime, date

sys.path.insert(0, str(Path(__file__).parent))
try:
    from db import get_connection
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from db import get_connection

PROJECT_ROOT = Path(__file__).parent.parent
COEFFICIENTS_PATH = PROJECT_ROOT / "models" / "coefficients.json"
CONFIG_PATH = PROJECT_ROOT / "config.json"
MORTGAGE_CSV = PROJECT_ROOT / "data" / "raw" / "MORTGAGE30US.csv"

MODEL_ZIPS = ["95765", "95677", "95678", "95661", "95746", "95650"]
CURRENT_YEAR = datetime.now().year

# ---------------------------------------------------------------------------
# Default config — matches data-dictionary.md schema
# ---------------------------------------------------------------------------
DEFAULT_LEAD_CONFIG = {
    "weights": {
        "tenure": 0.25,
        "rate_lock_in": 0.20,
        "value_gap": 0.25,
        "market_temperature": 0.15,
        "property_age": 0.15,
    },
    "tiers": {
        "A": {"min_score": 75, "label": "High Propensity"},
        "B": {"min_score": 50, "label": "Medium Propensity"},
        "C": {"min_score": 25, "label": "Low Propensity"},
        "D": {"min_score": 0, "label": "Not a Lead"},
    },
    "zip_filter": MODEL_ZIPS,
    "tenure_sweet_spot_years": 7,
    "rate_lock_threshold": 4.5,
    "value_gap_significant_pct": 15,
    "property_age_replacement_cycle": 18,
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_lead_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
            return cfg.get("lead_scoring", DEFAULT_LEAD_CONFIG)
        except Exception:
            pass
    return DEFAULT_LEAD_CONFIG


# ---------------------------------------------------------------------------
# Mortgage rate history (FRED CSV → dict of date → rate)
# ---------------------------------------------------------------------------
def load_mortgage_history() -> list[tuple[date, float]]:
    """Load FRED 30yr mortgage rate CSV as sorted [(date, rate)] list."""
    if not MORTGAGE_CSV.exists():
        return []
    history = []
    try:
        import csv
        with open(MORTGAGE_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                date_col = row.get("DATE") or row.get("date") or ""
                rate_col = row.get("MORTGAGE30US") or row.get("Value") or row.get("value") or ""
                if not date_col or not rate_col or rate_col.strip() == ".":
                    continue
                try:
                    d = datetime.strptime(date_col.strip(), "%Y-%m-%d").date()
                    r = float(rate_col.strip())
                    history.append((d, r))
                except (ValueError, TypeError):
                    continue
    except Exception:
        pass
    return sorted(history)


def lookup_rate_at_date(history: list, purchase_date_str: str) -> float | None:
    """Return the most recent mortgage rate on or before the given date."""
    if not history or not purchase_date_str:
        return None
    try:
        target = datetime.strptime(purchase_date_str[:10], "%Y-%m-%d").date()
    except ValueError:
        return None

    # Binary search / linear scan for merge_asof behavior
    best_rate = None
    for d, r in history:
        if d <= target:
            best_rate = r
        else:
            break
    return best_rate


# ---------------------------------------------------------------------------
# Model coefficients
# ---------------------------------------------------------------------------
def load_coefficients() -> dict | None:
    if not COEFFICIENTS_PATH.exists():
        return None
    with open(COEFFICIENTS_PATH) as f:
        return json.load(f)


def coef_val(coefs: dict, key: str, default: float = 0) -> float:
    v = coefs.get(key, default)
    if isinstance(v, dict):
        return v.get("coef", default)
    return float(v) if v is not None else default


def predict_price_from_coefs(coef_data: dict, sqft, lot_sqft, year_built, zip_code,
                              hoa=0, beds=None, baths=None, mortgage_rate=6.5) -> float | None:
    """Linear prediction directly from coefficients.json. Returns None if insufficient data."""
    if not coef_data or not sqft or not year_built:
        return None

    coefs = coef_data.get("coefficients", coef_data)
    intercept = coef_val(coefs, "const", coef_val(coefs, "intercept", 0))
    estimate = intercept

    estimate += coef_val(coefs, "sqft") * float(sqft)

    lot = lot_sqft if lot_sqft else 7200
    estimate += coef_val(coefs, "lot_sqft_log") * math.log1p(float(lot))

    if beds is not None:
        estimate += coef_val(coefs, "beds") * float(beds)

    if baths is not None:
        estimate += coef_val(coefs, "baths") * float(baths)

    age = CURRENT_YEAR - int(year_built)
    estimate += coef_val(coefs, "age") * float(age)

    estimate += coef_val(coefs, "hoa") * float(hoa or 0)

    if "mortgage_rate" in coefs:
        estimate += coef_val(coefs, "mortgage_rate") * float(mortgage_rate)

    zip_key = f"zip_{zip_code}"
    estimate += coef_val(coefs, zip_key, 0)

    return max(50_000, estimate)


# ---------------------------------------------------------------------------
# Scoring signal functions
# ---------------------------------------------------------------------------

def score_tenure(purchase_date_str: str | None, sweet_spot_years: int) -> tuple[float, str]:
    """Years of ownership → 0-100."""
    if not purchase_date_str:
        return 50.0, "No purchase date — neutral score"

    try:
        purchase = datetime.strptime(purchase_date_str[:10], "%Y-%m-%d").date()
    except ValueError:
        return 50.0, "Could not parse purchase date — neutral score"

    years = (date.today() - purchase).days / 365.25

    if years < 0:
        return 0.0, f"Future date ({purchase_date_str}) — scored 0"

    if years <= 2:
        # Recent purchase: 0-25 range
        raw = (years / 2) * 25
    elif years <= sweet_spot_years:
        # Ramp from 25 to 100 over years 2 → sweet_spot
        raw = 25 + ((years - 2) / (sweet_spot_years - 2)) * 75
    elif years <= 15:
        # Sweet spot: hold at 100
        raw = 100
    else:
        # Long-term owners: slight taper (100 → 70 over years 15-30)
        taper = min((years - 15) / 15, 1.0) * 30
        raw = 100 - taper

    score = round(min(100, max(0, raw)), 1)
    return score, f"{years:.1f} yrs owned (sweet spot: {sweet_spot_years} yrs)"


def score_rate_lock_in(purchase_date_str: str | None, mortgage_history: list,
                        threshold: float) -> tuple[float, str]:
    """Rate at purchase vs threshold → 0-100. Low rate = locked in = low score."""
    if not purchase_date_str:
        return 50.0, "No purchase date — neutral score"

    original_rate = lookup_rate_at_date(mortgage_history, purchase_date_str)

    if original_rate is None:
        return 50.0, "Rate history unavailable — neutral score"

    if original_rate < threshold:
        # Locked in: score 0-30 inversely proportional to gap from threshold
        gap = threshold - original_rate  # e.g. 4.5 - 3.0 = 1.5
        # Full lock-in (gap >= 2.5) → score ~0; gap=0 → score~30
        raw = 30 * (1 - min(gap / 2.5, 1.0))
    else:
        # No lock-in: score 60-100 proportional to how far above threshold
        excess = original_rate - threshold  # e.g. 7.0 - 4.5 = 2.5
        raw = 60 + min(excess / 3.0, 1.0) * 40

    score = round(min(100, max(0, raw)), 1)
    return score, f"Bought at {original_rate:.2f}% (threshold: {threshold}%)"


def score_value_gap(assessed_value: int | None, model_estimate: float | None,
                     gap_pct_threshold: float) -> tuple[float, str]:
    """Model estimate vs assessed value → 0-100. Large gap = high score."""
    if not assessed_value or not model_estimate or assessed_value <= 0:
        return 50.0, "Missing assessed or model value — neutral score"

    gap = model_estimate - assessed_value
    gap_pct = (gap / assessed_value) * 100

    if gap < 0:
        # Model estimate < assessed — property may be distressed or assessed high
        score = 0.0
        detail = f"Model ${model_estimate:,.0f} < assessed ${assessed_value:,.0f} ({gap_pct:.1f}%)"
    elif gap_pct < gap_pct_threshold:
        # Gap below threshold: score 0-40
        raw = (gap_pct / gap_pct_threshold) * 40
        score = round(raw, 1)
        detail = f"Gap {gap_pct:.1f}% (below {gap_pct_threshold:.0f}% threshold)"
    else:
        # Meaningful gap: ramp to 100 beyond threshold
        excess_pct = gap_pct - gap_pct_threshold
        raw = 40 + min(excess_pct / 30, 1.0) * 60  # 30pp above threshold → 100
        score = round(min(100, raw), 1)
        detail = f"Model ${model_estimate:,.0f} vs assessed ${assessed_value:,.0f} (+{gap_pct:.1f}% gap)"

    return score, detail


def score_market_temperature(market_data: dict | None) -> tuple[float, str]:
    """
    Composite of Redfin market condition columns from transactions table.
    Lower weeks_of_supply + higher sale_to_list + higher off_market = hotter.
    """
    if not market_data:
        return 50.0, "No market data available — neutral score"

    scores = []
    details = []

    wos = market_data.get("weeks_of_supply")
    if wos is not None and wos > 0:
        # < 4 weeks = very hot (score 100), > 16 weeks = cold (score 0)
        s = max(0, min(100, (16 - float(wos)) / 12 * 100))
        scores.append(s)
        details.append(f"{float(wos):.1f}wks supply")

    stl = market_data.get("sale_to_list_ratio")
    if stl is not None:
        stl = float(stl)
        # 0.96 = neutral, 1.03+ = very hot, < 0.93 = cold
        s = max(0, min(100, (stl - 0.93) / (1.04 - 0.93) * 100))
        scores.append(s)
        details.append(f"{stl:.3f} sale/list")

    omtw = market_data.get("off_market_in_two_weeks")
    if omtw is not None:
        omtw = float(omtw)
        # 0.20 = cold, 0.60 = very hot
        s = max(0, min(100, (omtw - 0.20) / 0.40 * 100))
        scores.append(s)
        details.append(f"{omtw:.0%} off-market <2wks")

    if not scores:
        return 50.0, "No market sub-signals available — neutral score"

    score = round(sum(scores) / len(scores), 1)
    return score, " | ".join(details)


def score_property_age(year_built: int | None, replacement_cycle: int) -> tuple[float, str]:
    """Home age relative to major system replacement cycle → 0-100."""
    if not year_built:
        return 50.0, "No year built — neutral score"

    age = CURRENT_YEAR - int(year_built)

    if age < 5:
        score = 0.0
        detail = f"{age} yrs old — too new"
    elif age <= replacement_cycle:
        # Ramp 0→100 from age 5 to replacement_cycle
        raw = (age - 5) / (replacement_cycle - 5) * 100
        score = round(min(100, raw), 1)
        detail = f"{age} yrs old (approaching {replacement_cycle}-yr cycle)"
    elif age <= 25:
        # Taper from 100 toward 60 between replacement_cycle and 25
        taper = (age - replacement_cycle) / (25 - replacement_cycle) * 40
        score = round(max(60, 100 - taper), 1)
        detail = f"{age} yrs old (past replacement cycle)"
    elif age <= 40:
        # Moderate — often renovated or owners have adapted
        raw = 80 - (age - 25) / 15 * 30
        score = round(max(50, raw), 1)
        detail = f"{age} yrs old — established home"
    else:
        score = 50.0
        detail = f"{age} yrs old — long-established"

    return score, detail


# ---------------------------------------------------------------------------
# Tier assignment
# ---------------------------------------------------------------------------
def assign_tier(score: float, tiers: dict) -> str:
    for tier_label in ("A", "B", "C", "D"):
        if score >= tiers.get(tier_label, {}).get("min_score", 0):
            return tier_label
    return "D"


# ---------------------------------------------------------------------------
# Market temperature data (one query per ZIP, reused)
# ---------------------------------------------------------------------------
def load_market_data_by_zip(conn: sqlite3.Connection) -> dict:
    """
    Pull latest Redfin market condition values per ZIP from transactions table.
    Returns dict of zip → {weeks_of_supply, sale_to_list_ratio, off_market_in_two_weeks}
    """
    try:
        rows = conn.execute("""
            SELECT zip,
                   weeks_of_supply,
                   sale_to_list_ratio,
                   off_market_in_two_weeks
            FROM transactions
            WHERE weeks_of_supply IS NOT NULL
              AND sale_to_list_ratio IS NOT NULL
            ORDER BY sale_date DESC
        """).fetchall()

        # Take the most recent non-null row per ZIP
        market = {}
        for row in rows:
            z = row[0]
            if z not in market:
                market[z] = {
                    "weeks_of_supply": row[1],
                    "sale_to_list_ratio": row[2],
                    "off_market_in_two_weeks": row[3],
                }
        return market
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------
def score_parcels(zip_filter: list[str] | None = None) -> dict:
    cfg = load_lead_config()
    weights = cfg["weights"]
    zip_filter = zip_filter or cfg.get("zip_filter", MODEL_ZIPS)

    sweet_spot = cfg.get("tenure_sweet_spot_years", 7)
    rate_threshold = cfg.get("rate_lock_threshold", 4.5)
    gap_pct = cfg.get("value_gap_significant_pct", 15)
    age_cycle = cfg.get("property_age_replacement_cycle", 18)

    coef_data = load_coefficients()
    mortgage_history = load_mortgage_history()
    conn = get_connection()

    market_by_zip = load_market_data_by_zip(conn)

    # Get current mortgage rate for value gap calculation
    try:
        rate_row = conn.execute("""
            SELECT mortgage_rate FROM transactions
            WHERE mortgage_rate IS NOT NULL
            ORDER BY sale_date DESC LIMIT 1
        """).fetchone()
        current_rate = float(rate_row[0]) if rate_row else 6.5
    except Exception:
        current_rate = 6.5

    # Load parcels
    placeholders = ",".join("?" * len(zip_filter))
    parcels = conn.execute(f"""
        SELECT id, parcel_id, address, city, zip, owner_name,
               purchase_date, assessed_value, sqft, lot_sqft,
               year_built, beds, baths
        FROM parcels
        WHERE zip IN ({placeholders})
    """, zip_filter).fetchall()

    print(f"  Scoring {len(parcels):,} parcels in ZIPs: {', '.join(zip_filter)}")

    stats = {"total": 0, "tiers": {"A": 0, "B": 0, "C": 0, "D": 0}}
    now_str = datetime.utcnow().isoformat()

    BATCH = 200
    batch_rows = []

    def flush():
        if batch_rows:
            conn.executemany("""
                INSERT OR REPLACE INTO lead_scores
                  (parcel_id, address, score, score_components, last_scored, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, batch_rows)
            conn.commit()
            batch_rows.clear()

    for row in parcels:
        (pid_int, parcel_id, address, city, zip_code, owner_name,
         purchase_date, assessed_value, sqft, lot_sqft, year_built, beds, baths) = row

        # Use row id as parcel_id if none
        effective_parcel_id = parcel_id or str(pid_int)

        # --- Signal 1: Tenure ---
        s_tenure, d_tenure = score_tenure(purchase_date, sweet_spot)

        # --- Signal 2: Rate lock-in ---
        s_rate, d_rate = score_rate_lock_in(purchase_date, mortgage_history, rate_threshold)

        # --- Signal 3: Value gap ---
        model_est = predict_price_from_coefs(
            coef_data, sqft, lot_sqft, year_built, zip_code,
            hoa=0, beds=beds, baths=baths, mortgage_rate=current_rate
        ) if coef_data else None
        s_gap, d_gap = score_value_gap(assessed_value, model_est, gap_pct)

        # --- Signal 4: Market temperature ---
        market_data = market_by_zip.get(zip_code)
        s_market, d_market = score_market_temperature(market_data)

        # --- Signal 5: Property age ---
        s_age, d_age = score_property_age(year_built, age_cycle)

        # --- Composite ---
        composite = (
            weights["tenure"] * s_tenure +
            weights["rate_lock_in"] * s_rate +
            weights["value_gap"] * s_gap +
            weights["market_temperature"] * s_market +
            weights["property_age"] * s_age
        )
        composite = round(composite, 1)

        tier = assign_tier(composite, cfg["tiers"])

        components = {
            "tenure":             {"value": round(s_tenure, 1), "detail": d_tenure, "weight": weights["tenure"]},
            "rate_lock_in":       {"value": round(s_rate, 1),   "detail": d_rate,   "weight": weights["rate_lock_in"]},
            "value_gap":          {"value": round(s_gap, 1),    "detail": d_gap,    "weight": weights["value_gap"]},
            "market_temperature": {"value": round(s_market, 1), "detail": d_market, "weight": weights["market_temperature"]},
            "property_age":       {"value": round(s_age, 1),    "detail": d_age,    "weight": weights["property_age"]},
            "tier": tier,
        }

        batch_rows.append((
            effective_parcel_id,
            address,
            composite,
            json.dumps(components),
            now_str,
            now_str,
        ))

        stats["total"] += 1
        stats["tiers"][tier] = stats["tiers"].get(tier, 0) + 1

        if len(batch_rows) >= BATCH:
            flush()

    flush()
    conn.close()

    return stats


# ---------------------------------------------------------------------------
# Stats / reporting
# ---------------------------------------------------------------------------
def print_stats():
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT json_extract(score_components, '$.tier') as tier,
                   COUNT(*) as n,
                   ROUND(AVG(score), 1) as avg_score,
                   ROUND(MIN(score), 1) as min_score,
                   ROUND(MAX(score), 1) as max_score
            FROM lead_scores
            GROUP BY tier ORDER BY avg_score DESC
        """).fetchall()

        total = conn.execute("SELECT COUNT(*) FROM lead_scores").fetchone()[0]

        print(f"\n  Lead Score Distribution  ({total:,} parcels)")
        print(f"  {'─'*50}")
        print(f"  {'Tier':<6} {'Count':>7} {'Avg':>6} {'Min':>6} {'Max':>6}")
        print(f"  {'─'*50}")
        for r in rows:
            tier, n, avg, mn, mx = r
            bar_len = int(n / max(1, total) * 25)
            bar = "█" * bar_len
            print(f"  {tier or '?':<6} {n:>7,} {avg:>6.1f} {mn:>6.1f} {mx:>6.1f}  {bar}")
        print(f"  {'─'*50}")
        print()
    except Exception as e:
        print(f"  Error: {e}")
    finally:
        conn.close()


def print_top_leads(n: int = 10):
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT ls.address, ls.score,
                   json_extract(ls.score_components, '$.tier') as tier,
                   p.zip, p.owner_name, p.purchase_date, p.assessed_value,
                   p.year_built
            FROM lead_scores ls
            LEFT JOIN parcels p ON ls.parcel_id = p.parcel_id
            ORDER BY ls.score DESC LIMIT ?
        """, (n,)).fetchall()

        print(f"\n  Top {n} Leads")
        print(f"  {'─'*80}")
        for i, r in enumerate(rows, 1):
            addr, score, tier, zip_code, owner, pdate, aval, yr = r
            print(f"  {i:2}. [{tier}] {score:5.1f}  {addr or '—'}")
            if owner:
                print(f"      Owner: {owner}  ZIP: {zip_code}  Built: {yr}")
            if pdate:
                print(f"      Purchased: {pdate}  Assessed: ${(aval or 0):,}")
        print()
    except Exception as e:
        print(f"  Error: {e}")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score leads from parcels table")
    parser.add_argument("--zip", type=str, help="Score a single ZIP code")
    parser.add_argument("--stats", action="store_true", help="Print tier distribution only")
    parser.add_argument("--top", type=int, default=0, help="Print top N leads after scoring")
    args = parser.parse_args()

    if args.stats:
        print_stats()
        sys.exit(0)

    zip_filter = [args.zip] if args.zip else None

    print(f"\n  Lead Scoring Engine")
    print(f"  {'─'*45}")

    start = datetime.now()
    stats = score_parcels(zip_filter)
    elapsed = (datetime.now() - start).total_seconds()

    print(f"\n  ✅  Scoring complete  ({elapsed:.1f}s)")
    print(f"  {'─'*45}")
    print(f"  Total scored: {stats['total']:,}")
    for tier in ("A", "B", "C", "D"):
        count = stats["tiers"].get(tier, 0)
        pct = count / max(1, stats["total"]) * 100
        print(f"    Tier {tier}: {count:>6,}  ({pct:.1f}%)")

    print_stats()

    if args.top:
        print_top_leads(args.top)
    elif stats["total"] > 0:
        print_top_leads(10)
