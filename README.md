# RE Analytics — Local Control Panel

A Flask-based dashboard for customizing your real estate pricing model, one-pager output, and agent branding. Runs entirely local — no external services, no API keys.

## Quick Start

```bash
# 1. Install the one dependency
pip install flask

# 2. Generate demo database (or use your existing re_analytics.db)
python setup_demo.py

# 3. Launch
python app.py

# 4. Open in browser
open http://localhost:5000
```

## Integration with Your Existing Pipeline

This dashboard wraps your existing `src/` modules. To connect it to your real pipeline instead of the demo data:

### Option A: Point at your existing database
Edit `app.py` line ~20 — change `DB_PATH` to your actual `re_analytics.db` path:
```python
DB_PATH = Path("/Users/dmitriy/RealEstateTools/weekend3-14/re-analytics/re_analytics.db")
MODELS_DIR = Path("/Users/dmitriy/RealEstateTools/weekend3-14/re-analytics/models")
```

### Option B: Copy this into your existing project
Drop `app.py`, `templates/`, and `setup_demo.py` into your `re-analytics/` directory. The paths default to the same directory as `app.py`, so it'll find your `re_analytics.db` and `models/coefficients.json` automatically.

### Hook config.json into onepager.py
The dashboard saves all settings to `config.json`. Replace the hardcoded constants at the top of your `onepager.py` with:
```python
import json
with open("config.json") as f:
    config = json.load(f)

AGENT_NAME = config["agent"]["name"]
AGENT_PHONE = config["agent"]["phone"]
AGENT_EMAIL = config["agent"]["email"]
# ... etc
```

## What Each Panel Does

### Price Estimator
- Input property features → get a live price estimate
- Shows the **coefficient breakdown**: exactly how much each feature contributes
- Uses per-ZIP RMSE for the confidence interval (not global)

### One-Pager Studio
- **Design tokens**: colors, fonts — change them and see live preview
- **Section toggles**: turn QR code, trend chart, neighborhood stats, comp table on/off
- **Copy**: edit the CTA text and footer disclaimer
- **Live preview**: renders the one-pager HTML in an iframe with your current settings
- Click "Save Design" to persist to `config.json`

### Model Tuning
- **Feature toggles**: enable/disable regression features (sqft, beds, age, etc.)
- **Tier thresholds**: adjust what MAPE qualifies as high/medium/low confidence
- **Training params**: min samples per ZIP, test split %, prediction interval width
- **Per-ZIP table**: current accuracy metrics with tier badges
- **Coefficient table**: every model coefficient with human-readable interpretation
- Save changes, then re-run `python src/model.py` to retrain with new settings

### Agent & Branding
- Name, phone, email, tagline, license number, brokerage
- Used everywhere: one-pagers, landing pages, reports

### Data Explorer
- Browse transactions by ZIP
- Click a ZIP card to filter
- See avg price, sqft, DOM at a glance

## Architecture

```
app.py                  ← Flask server (all routes)
config.json             ← Persisted settings (auto-created on first save)
re_analytics.db         ← SQLite database
models/
  coefficients.json     ← Model coefficients
templates/
  dashboard.html        ← Main control panel (single-page app)
  onepager_preview.html ← One-pager live preview template
setup_demo.py           ← Demo data generator
```

All state lives in `config.json` (settings) and `re_analytics.db` (data). No build step, no npm, no React — just `python app.py`.

## Config Schema

The `config.json` file has three top-level sections:

```json
{
  "agent": {
    "name": "...",
    "phone": "...",
    "email": "...",
    "tagline": "...",
    "license_number": "...",
    "brokerage": "..."
  },
  "onepager": {
    "primary_color": "#1a2744",
    "accent_color": "#c9953c",
    "heading_font": "Playfair Display",
    "body_font": "DM Sans",
    "show_qr_code": true,
    "show_trend_chart": true,
    "show_neighborhood_stats": true,
    "show_confidence_interval": true,
    "show_comp_table": true,
    "footer_text": "...",
    "cta_text": "..."
  },
  "model": {
    "confidence_tiers": {
      "high": { "max_mape": 7.0 },
      "medium": { "max_mape": 12.0 }
    },
    "features_enabled": {
      "sqft": true,
      "lot_sqft_log": true,
      "beds": true,
      "baths": true,
      "age": true,
      "hoa": true,
      "zip_dummies": true
    },
    "min_zip_samples": 30,
    "test_split": 0.2,
    "prediction_interval_pct": 90
  }
}
```

Every field is editable from the UI. Every change persists immediately.
