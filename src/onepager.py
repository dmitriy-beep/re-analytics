"""
onepager.py — HTML template rendering and PDF generation for door-knocking.

Usage:
    python src/onepager.py --zip 95746
    python src/onepager.py --address "1234 Oak St" --zip 95678 --sqft 1850 --beds 3 --baths 2 --year-built 1998 --hoa 0
    python src/onepager.py --test          # generates one sample per ZIP

Generates one print-ready PDF per property in output/.
Content adapts to confidence tier (high / medium / low).
"""

import sys
import re
import argparse
import base64
import sqlite3
import pandas as pd

from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Optional

import qrcode
from jinja2 import Template
from weasyprint import HTML, CSS

sys.path.insert(0, str(Path(__file__).parent))
from db import get_connection
from predict import PropertyFeatures, predict, PredictionResult

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

BASE_URL = "http://localhost:8000/value"

AGENT_NAME  = "Dmitriy"
AGENT_PHONE = "(916) 555-0100"
AGENT_EMAIL = "dmitriy@yourdomain.com"


# ── QR code ───────────────────────────────────────────────────────────────────

def make_qr_base64(url: str) -> str:
    """Generate a QR code and return it as a base64-encoded PNG string."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=6,
        border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#1a1a2e", back_color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def address_slug(address: str) -> str:
    """Convert an address to a URL-safe slug."""
    slug = address.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


# ── Neighborhood trend ────────────────────────────────────────────────────────

def get_neighborhood_trend(zip_code: str, conn: sqlite3.Connection) -> dict:
    """
    Pull last 12 months of median sale price by quarter for the ZIP.
    Returns a dict with trend data for the one-pager.
    """
    df = pd.read_sql("""
        SELECT
            sale_price,
            sale_date,
            strftime('%Y-Q', sale_date) ||
                CASE
                    WHEN CAST(strftime('%m', sale_date) AS INTEGER) <= 3  THEN '1'
                    WHEN CAST(strftime('%m', sale_date) AS INTEGER) <= 6  THEN '2'
                    WHEN CAST(strftime('%m', sale_date) AS INTEGER) <= 9  THEN '3'
                    ELSE '4'
                END as quarter
        FROM transactions
        WHERE zip = ?
          AND sale_date >= date('now', '-15 months')
        ORDER BY sale_date
    """, conn, params=(zip_code,))

    if df.empty:
        return {"quarters": [], "medians": [], "trend_pct": None}

    quarterly = df.groupby("quarter")["sale_price"].median().reset_index()
    quarterly = quarterly.tail(5)  # last 5 quarters max

    quarters = quarterly["quarter"].tolist()
    medians  = [int(m) for m in quarterly["sale_price"].tolist()]

    trend_pct = None
    if len(medians) >= 2:
        trend_pct = round((medians[-1] - medians[0]) / medians[0] * 100, 1)

    # Recent stats
    recent = df[df["sale_date"] >= df["sale_date"].max()[:7]]  # last month approx
    n_recent = pd.read_sql("""
        SELECT COUNT(*) as n FROM transactions
        WHERE zip = ? AND sale_date >= date('now', '-90 days')
    """, conn, params=(zip_code,)).iloc[0]["n"]

    median_dom = pd.read_sql("""
        SELECT CAST(AVG(days_on_market) AS INTEGER) as avg_dom
        FROM transactions
        WHERE zip = ? AND sale_date >= date('now', '-90 days')
          AND days_on_market IS NOT NULL
    """, conn, params=(zip_code,)).iloc[0]["avg_dom"]

    return {
        "quarters": quarters,
        "medians": medians,
        "trend_pct": trend_pct,
        "n_recent": int(n_recent),
        "avg_dom": int(median_dom) if median_dom else None,
        "current_median": medians[-1] if medians else None,
    }


# ── HTML templates ────────────────────────────────────────────────────────────

# Shared CSS across all tiers
BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

* { margin: 0; padding: 0; box-sizing: border-box; }

@page {
    size: letter;
    margin: 0;
}

body {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    color: #1a1a2e;
    background: white;
    width: 8.5in;
    height: 11in;
    overflow: hidden;
}

.page {
    width: 8.5in;
    height: 11in;
    display: flex;
    flex-direction: column;
    position: relative;
}

.accent-bar {
    height: 6px;
    background: linear-gradient(90deg, #1a1a2e 0%, #c9a84c 60%, #1a1a2e 100%);
}

.header {
    padding: 36px 52px 28px;
    border-bottom: 1px solid #e8e4dc;
}

.eyebrow {
    font-size: 9px;
    font-weight: 500;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #c9a84c;
    margin-bottom: 10px;
}

.headline {
    font-family: 'Playfair Display', serif;
    font-size: 28px;
    font-weight: 700;
    line-height: 1.2;
    color: #1a1a2e;
    max-width: 5.5in;
}

.subhead {
    font-size: 13px;
    color: #666;
    margin-top: 8px;
    font-weight: 400;
}

.body {
    flex: 1;
    padding: 32px 52px;
    display: flex;
    flex-direction: column;
    gap: 28px;
}

.value-block {
    background: #f9f6f0;
    border-left: 4px solid #c9a84c;
    padding: 22px 28px;
    border-radius: 0 6px 6px 0;
}

.value-label {
    font-size: 9px;
    font-weight: 500;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #999;
    margin-bottom: 6px;
}

.value-number {
    font-family: 'Playfair Display', serif;
    font-size: 42px;
    font-weight: 700;
    color: #1a1a2e;
    line-height: 1;
}

.value-range {
    font-size: 13px;
    color: #666;
    margin-top: 6px;
}

.stats-row {
    display: flex;
    gap: 0;
}

.stat {
    flex: 1;
    padding: 16px 20px;
    border: 1px solid #e8e4dc;
    border-radius: 6px;
    margin-right: 12px;
}
.stat:last-child { margin-right: 0; }

.stat-value {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 700;
    color: #1a1a2e;
}

.stat-label {
    font-size: 10px;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 3px;
}

.trend-block {
    padding: 0;
}

.trend-label {
    font-size: 9px;
    font-weight: 500;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #999;
    margin-bottom: 12px;
}

.bar-chart {
    display: flex;
    align-items: flex-end;
    gap: 8px;
    height: 80px;
}

.bar-wrap {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    height: 100%;
    justify-content: flex-end;
    gap: 4px;
}

.bar {
    width: 100%;
    background: #1a1a2e;
    border-radius: 2px 2px 0 0;
    min-height: 4px;
}

.bar.latest { background: #c9a84c; }

.bar-label {
    font-size: 8px;
    color: #999;
    text-align: center;
    white-space: nowrap;
}

.bar-val {
    font-size: 8px;
    color: #666;
    text-align: center;
}

.cta-block {
    background: #1a1a2e;
    color: white;
    padding: 22px 28px;
    border-radius: 6px;
}

.cta-question {
    font-family: 'Playfair Display', serif;
    font-size: 17px;
    font-weight: 400;
    line-height: 1.4;
    margin-bottom: 10px;
}

.cta-sub {
    font-size: 12px;
    color: #aaa;
    line-height: 1.5;
}

.footer {
    padding: 20px 52px;
    border-top: 1px solid #e8e4dc;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.agent-info { line-height: 1.6; }

.agent-name {
    font-family: 'Playfair Display', serif;
    font-size: 15px;
    font-weight: 700;
}

.agent-contact {
    font-size: 11px;
    color: #666;
}

.qr-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
}

.qr-wrap img {
    width: 72px;
    height: 72px;
}

.qr-label {
    font-size: 8px;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

.disclaimer {
    font-size: 7.5px;
    color: #bbb;
    text-align: center;
    padding: 0 52px 14px;
    line-height: 1.4;
}
"""


HIGH_TEMPLATE = Template("""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body>
<div class="page">
  <div class="accent-bar"></div>

  <div class="header">
    <div class="eyebrow">Confidential Market Analysis · {{ zip_code }}</div>
    <div class="headline">{{ headline }}</div>
    <div class="subhead">{{ address }}</div>
  </div>

  <div class="body">
    <div class="value-block">
      <div class="value-label">Estimated Market Value</div>
      <div class="value-number">${{ '{:,}'.format(point_estimate) }}</div>
      <div class="value-range">90% confidence range: ${{ '{:,}'.format(interval_low) }} – ${{ '{:,}'.format(interval_high) }}</div>
    </div>

    <div class="stats-row">
      <div class="stat">
        <div class="stat-value">${{ '{:,}'.format(neighborhood_median) }}</div>
        <div class="stat-label">Neighborhood Median</div>
      </div>
      <div class="stat">
        <div class="stat-value">{{ n_transactions }}</div>
        <div class="stat-label">Sales Analyzed</div>
      </div>
      {% if avg_dom %}
      <div class="stat">
        <div class="stat-value">{{ avg_dom }}</div>
        <div class="stat-label">Avg Days on Market</div>
      </div>
      {% endif %}
    </div>

    {% if quarters %}
    <div class="trend-block">
      <div class="trend-label">Median Sale Price · Last 12 Months</div>
      <div class="bar-chart">
        {% set max_val = medians | max %}
        {% for i in range(quarters | length) %}
        <div class="bar-wrap">
          <div class="bar-val">${{ '{:.0f}K'.format(medians[i] / 1000) }}</div>
          <div class="bar {% if loop.last %}latest{% endif %}"
               style="height: {{ [(medians[i] / max_val * 64), 4] | max }}px"></div>
          <div class="bar-label">{{ quarters[i][-2:] }}</div>
        </div>
        {% endfor %}
      </div>
    </div>
    {% endif %}

    <div class="cta-block">
      <div class="cta-question">This estimate is based on {{ n_transactions }} recent sales.<br>Want to see exactly how your home compares?</div>
      <div class="cta-sub">I can walk you through the specific sales used in this analysis and explain what makes your home different — in about 15 minutes, over coffee or at your kitchen table.</div>
    </div>
  </div>

  <div class="footer">
    <div class="agent-info">
      <div class="agent-name">{{ agent_name }}</div>
      <div class="agent-contact">{{ agent_phone }} · {{ agent_email }}</div>
      <div class="agent-contact">Licensed California Real Estate Agent</div>
    </div>
    <div class="qr-wrap">
      <img src="data:image/png;base64,{{ qr_b64 }}" />
      <div class="qr-label">Full Report</div>
    </div>
  </div>

  <div class="disclaimer">
    This analysis is based on {{ n_transactions }} recorded sales in ZIP {{ zip_code }} and is provided for informational purposes only.
    It does not constitute a formal appraisal. Market conditions change; results may vary.
  </div>
</div>
</body>
</html>
""")


MEDIUM_TEMPLATE = Template("""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body>
<div class="page">
  <div class="accent-bar"></div>

  <div class="header">
    <div class="eyebrow">Neighborhood Market Summary · {{ zip_code }}</div>
    <div class="headline">What homes like yours are selling for right now</div>
    <div class="subhead">{{ address }}</div>
  </div>

  <div class="body">
    <div class="value-block">
      <div class="value-label">Estimated Value Range</div>
      <div class="value-number">${{ '{:.0f}K'.format(interval_low / 1000) }} – ${{ '{:.0f}K'.format(interval_high / 1000) }}</div>
      <div class="value-range">Based on {{ n_transactions }} comparable sales in {{ zip_code }}</div>
    </div>

    <div class="stats-row">
      <div class="stat">
        <div class="stat-value">${{ '{:,}'.format(neighborhood_median) }}</div>
        <div class="stat-label">Area Median Price</div>
      </div>
      <div class="stat">
        <div class="stat-value">{{ n_transactions }}</div>
        <div class="stat-label">Sales Analyzed</div>
      </div>
      {% if avg_dom %}
      <div class="stat">
        <div class="stat-value">{{ avg_dom }}</div>
        <div class="stat-label">Avg Days on Market</div>
      </div>
      {% endif %}
    </div>

    {% if quarters %}
    <div class="trend-block">
      <div class="trend-label">Median Sale Price · Recent Trend</div>
      <div class="bar-chart">
        {% set max_val = medians | max %}
        {% for i in range(quarters | length) %}
        <div class="bar-wrap">
          <div class="bar-val">${{ '{:.0f}K'.format(medians[i] / 1000) }}</div>
          <div class="bar {% if loop.last %}latest{% endif %}"
               style="height: {{ [(medians[i] / max_val * 64), 4] | max }}px"></div>
          <div class="bar-label">{{ quarters[i][-2:] }}</div>
        </div>
        {% endfor %}
      </div>
    </div>
    {% endif %}

    <div class="cta-block">
      <div class="cta-question">A more precise number depends on your home's specific condition and features.</div>
      <div class="cta-sub">I'd be glad to do a proper analysis once I've seen your home — no cost, no obligation. Most sellers find it changes how they think about timing and pricing.</div>
    </div>
  </div>

  <div class="footer">
    <div class="agent-info">
      <div class="agent-name">{{ agent_name }}</div>
      <div class="agent-contact">{{ agent_phone }} · {{ agent_email }}</div>
      <div class="agent-contact">Licensed California Real Estate Agent</div>
    </div>
    <div class="qr-wrap">
      <img src="data:image/png;base64,{{ qr_b64 }}" />
      <div class="qr-label">Learn More</div>
    </div>
  </div>

  <div class="disclaimer">
    This summary is based on recorded sales in ZIP {{ zip_code }} and is provided for informational purposes only.
    It does not constitute a formal appraisal. Actual value depends on property condition, features, and current market activity.
  </div>
</div>
</body>
</html>
""")


LOW_TEMPLATE = Template("""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body>
<div class="page">
  <div class="accent-bar"></div>

  <div class="header">
    <div class="eyebrow">Market Activity Report · {{ zip_code }}</div>
    <div class="headline">Your neighborhood is moving — here's what the data shows</div>
    <div class="subhead">{{ address }}</div>
  </div>

  <div class="body">
    <div class="value-block">
      <div class="value-label">Area Median Sale Price</div>
      <div class="value-number">${{ '{:,}'.format(neighborhood_median) }}</div>
      <div class="value-range">Based on {{ n_transactions }} sales in {{ zip_code }} · last 12 months</div>
    </div>

    <div class="stats-row">
      <div class="stat">
        <div class="stat-value">{{ n_transactions }}</div>
        <div class="stat-label">Homes Sold</div>
      </div>
      {% if avg_dom %}
      <div class="stat">
        <div class="stat-value">{{ avg_dom }}</div>
        <div class="stat-label">Avg Days on Market</div>
      </div>
      {% endif %}
      {% if trend_pct is not none %}
      <div class="stat">
        <div class="stat-value">{{ '+' if trend_pct > 0 else '' }}{{ trend_pct }}%</div>
        <div class="stat-label">Price Trend (YoY)</div>
      </div>
      {% endif %}
    </div>

    {% if quarters %}
    <div class="trend-block">
      <div class="trend-label">Median Sale Price · Recent Trend</div>
      <div class="bar-chart">
        {% set max_val = medians | max %}
        {% for i in range(quarters | length) %}
        <div class="bar-wrap">
          <div class="bar-val">${{ '{:.0f}K'.format(medians[i] / 1000) }}</div>
          <div class="bar {% if loop.last %}latest{% endif %}"
               style="height: {{ [(medians[i] / max_val * 64), 4] | max }}px"></div>
          <div class="bar-label">{{ quarters[i][-2:] }}</div>
        </div>
        {% endfor %}
      </div>
    </div>
    {% endif %}

    <div class="cta-block">
      <div class="cta-question">Every home in this area is different — yours included.</div>
      <div class="cta-sub">I'd need to see your home to give you an accurate number. There's meaningful variation in this market that a drive-by estimate won't capture. A 15-minute walkthrough is all it takes.</div>
    </div>
  </div>

  <div class="footer">
    <div class="agent-info">
      <div class="agent-name">{{ agent_name }}</div>
      <div class="agent-contact">{{ agent_phone }} · {{ agent_email }}</div>
      <div class="agent-contact">Licensed California Real Estate Agent</div>
    </div>
    <div class="qr-wrap">
      <img src="data:image/png;base64,{{ qr_b64 }}" />
      <div class="qr-label">See Nearby Sales</div>
    </div>
  </div>

  <div class="disclaimer">
    This report is based on recorded sales in ZIP {{ zip_code }} and is provided for informational purposes only.
    It does not constitute a formal appraisal or offer to purchase.
  </div>
</div>
</body>
</html>
""")

TEMPLATES = {"high": HIGH_TEMPLATE, "medium": MEDIUM_TEMPLATE, "low": LOW_TEMPLATE}

HIGH_HEADLINES = [
    "If your home is worth more than you think, does that change anything?",
    "Here's what the data says your home is worth right now.",
    "The market has moved. Here's where your home stands today.",
]


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_onepager(
    address: str,
    prop: PropertyFeatures,
    conn: sqlite3.Connection,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Generate a one-pager PDF for a single property.
    Returns the path to the generated PDF.
    """
    result = predict(prop, address=address, conn=conn)
    trend  = get_neighborhood_trend(prop.zip_code, conn)

    slug = address_slug(address or f"{prop.zip_code}-property")
    url  = f"{BASE_URL}/{slug}"
    qr_b64 = make_qr_base64(url)

    neighborhood_median = trend.get("current_median") or result.point_estimate
    tier = result.confidence_tier
    template = TEMPLATES[tier]

    context = dict(
        address=address or prop.zip_code,
        zip_code=prop.zip_code,
        point_estimate=result.point_estimate,
        interval_low=result.interval_low,
        interval_high=result.interval_high,
        neighborhood_median=neighborhood_median,
        n_transactions=result.n_transactions,
        avg_dom=trend.get("avg_dom"),
        trend_pct=trend.get("trend_pct"),
        quarters=trend.get("quarters", []),
        medians=trend.get("medians", []),
        headline=HIGH_HEADLINES[0],
        agent_name=AGENT_NAME,
        agent_phone=AGENT_PHONE,
        agent_email=AGENT_EMAIL,
        qr_b64=qr_b64,
    )

    html_str = template.render(**context)

    if output_path is None:
        output_path = OUTPUT_DIR / f"{slug}.pdf"

    HTML(string=html_str).write_pdf(
        output_path,
        stylesheets=[CSS(string=BASE_CSS)],
    )

    print(f"  ✅ [{tier.upper()}] {address or prop.zip_code} → {output_path.name}")
    return output_path


# ── Batch generation ──────────────────────────────────────────────────────────

def batch_by_zip(zip_code: str, conn: sqlite3.Connection, limit: int = 50) -> list[Path]:
    """
    Generate one-pagers for all parcels in a ZIP.
    Falls back to generating from transaction data if no parcels loaded.
    Uses transactions table as a proxy (address + last known features).
    """
    df = pd.read_sql("""
        SELECT address, zip, sqft, beds, baths, year_built, hoa, lot_sqft
        FROM transactions
        WHERE zip = ?
        ORDER BY sale_date DESC
        LIMIT ?
    """, conn, params=(zip_code, limit))

    if df.empty:
        print(f"  No transactions found for ZIP {zip_code}")
        return []

    print(f"\nGenerating {len(df)} one-pagers for ZIP {zip_code}...")
    paths = []
    for _, row in df.iterrows():
        prop = PropertyFeatures(
            sqft=int(row["sqft"]) if row["sqft"] else 1800,
            lot_sqft=int(row["lot_sqft"]) if row["lot_sqft"] else None,
            beds=int(row["beds"]) if row["beds"] else 3,
            baths=float(row["baths"]) if row["baths"] else 2.0,
            year_built=int(row["year_built"]) if row["year_built"] else 2000,
            hoa=int(row["hoa"]) if row["hoa"] else 0,
            zip_code=row["zip"],
        )
        try:
            path = render_onepager(row["address"], prop, conn)
            paths.append(path)
        except Exception as e:
            print(f"  ⚠️  Skipped {row['address']}: {e}")

    return paths


# ── Test mode ─────────────────────────────────────────────────────────────────

def run_test() -> None:
    """Generate one sample one-pager per ZIP for visual QA."""
    test_properties = [
        ("1234 Oak Creek Dr, Granite Bay CA",   PropertyFeatures(2400, 8500,  4, 3.0, 2005, 0,   "95746")),
        ("567 Willowbrook Ln, Roseville CA",    PropertyFeatures(1850, 5200,  3, 2.0, 1998, 120, "95678")),
        ("890 Sierra Vista Way, Rocklin CA",    PropertyFeatures(2100, 7000,  4, 2.5, 2010, 0,   "95677")),
        ("321 Vintage Oaks Ct, Roseville CA",   PropertyFeatures(3200, 9800,  5, 3.5, 2015, 85,  "95765")),
        ("45 Gold Hill Rd, Loomis CA",          PropertyFeatures(3800, 43560, 4, 3.0, 1990, 0,   "95650")),
        ("789 Parkside Ave, Roseville CA",      PropertyFeatures(1600, 4800,  3, 2.0, 1988, 0,   "95661")),
    ]

    conn = get_connection()
    print(f"\nGenerating test one-pagers → {OUTPUT_DIR}\n")
    for address, prop in test_properties:
        render_onepager(address, prop, conn)
    conn.close()
    print(f"\nDone. Open the PDFs in {OUTPUT_DIR}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate property one-pager PDFs")
    parser.add_argument("--test", action="store_true", help="Generate sample one-pager per ZIP")
    parser.add_argument("--zip", help="Batch generate for a ZIP code")
    parser.add_argument("--limit", type=int, default=50, help="Max one-pagers in batch mode")
    parser.add_argument("--address", help="Single property address")
    parser.add_argument("--sqft", type=int)
    parser.add_argument("--beds", type=int)
    parser.add_argument("--baths", type=float)
    parser.add_argument("--year-built", type=int, dest="year_built")
    parser.add_argument("--lot-sqft", type=int, dest="lot_sqft")
    parser.add_argument("--hoa", type=int, default=0)
    args = parser.parse_args()

    if args.test:
        run_test()

    elif args.zip and not args.address:
        conn = get_connection()
        batch_by_zip(args.zip, conn, limit=args.limit)
        conn.close()

    elif args.address and args.zip and args.sqft and args.beds and args.baths and args.year_built:
        prop = PropertyFeatures(
            sqft=args.sqft,
            lot_sqft=args.lot_sqft,
            beds=args.beds,
            baths=args.baths,
            year_built=args.year_built,
            hoa=args.hoa,
            zip_code=args.zip,
        )
        conn = get_connection()
        path = render_onepager(args.address, prop, conn)
        conn.close()
        print(f"PDF saved to: {path}")

    else:
        parser.print_help()
