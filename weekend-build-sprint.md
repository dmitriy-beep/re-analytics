# Weekend Build Sprint — Execution Checklist

**Goal:** By Sunday night, have a working end-to-end pipeline: real transaction data in a database, a validated pricing model, a one-pager generator, and the assessor data staged for lead scoring. Walk into the team leader conversation with a live demo, not a pitch.

**Current data situation:** Redfin CSV covering Granite Bay, Roseville, Rocklin and Loomis (12 ZIPs, 6 usable) is loaded into SQLite. Placer County assessor parcel data downloaded (199k rows). County recorder bulk access not available without payment — deferred. Saturday pipeline is complete through one-pager generation. Dashboard and control panel are live.

---

## Saturday Morning — Data Verification and Acquisition ✅ COMPLETE

### Redfin Transaction Data — DONE

- [x] ~~Download Redfin CSV~~ — already done for Granite Bay + Roseville
- [x] Spot check the file — confirmed headers, row count, ZIP distribution
- [x] Confirmed exact header row — see data dictionary Source 1
- [x] Raw row count: 4,836 | Cleaned row count: 4,320
- [x] Export contained 12 ZIPs (more than expected) — 6 with sufficient sample size
- [x] ZIP 95747 (11 rows), 95630 (8), 95610 (3), 95663 (1), 95662 (1), 98678 (1) — all flagged as low sample, excluded from model training

### County Assessor Parcel Data

- [x] Placer County: bulk parcel data downloaded — 199k rows, `data/raw/Public_Parcels_-7962162285201559209.csv`. Free download from GIS open data portal (gis-placercounty.opendata.arcgis.com). Full schema and column mapping documented in data-dictionary.md Source 3.
- [ ] Sacramento County: deferred — low overlap with current model ZIPs. Access via saccounty.nextrequest.com when needed.

### County Recorder Data

- [x] Investigated Sacramento and Placer County Recorder — no bulk export available from either county without payment. Both offer index-only online search. Bulk deed/loan data requires paid vendor (ParcelQuest, ATTOM). Deferred — equity signal approximated from assessor TransactionDt + FRED rate. Documented in data-dictionary.md Source 4.

---

## Saturday Midday — Database and Data Cleaning Pipeline ✅ COMPLETE

### Project Setup — DONE

- [x] Created project directory: `/Users/dmitriy/RealEstateTools/weekend3-14/re-analytics/`
- [x] Full directory structure: `data/{raw,cleaned,assessor}`, `models/`, `output/`, `src/`
- [x] Redfin CSV moved to `data/raw/redfin-favorites_2026-03-14-13-25-43.csv`
- [x] All source files in `src/`: `db.py`, `ingest.py`, `model.py`, `predict.py`, `onepager.py`
- [x] `requirements.txt` and `README.md` in project root
- [x] venv created and all dependencies installed

### Database — DONE

- [x] SQLite database initialized at `re_analytics.db`
- [x] All tables created: `transactions`, `parcels`, `model_accuracy`, `interactions`, `lead_scores`, `follow_ups`

### Cleaning and Ingest Script (`ingest.py`) — DONE

- [x] Column mapping confirmed against actual Redfin headers
- [x] Filters: residential only, $50K floor, sqft 400–10,000, $/sqft $50–$1,500
- [x] Deduplication on: address + sale_date
- [x] HOA missing → 0
- [x] Idempotent — safe to re-run, skips already-loaded rows
- [x] Loaded 4,320 rows into transactions table

---

## Saturday Afternoon — Pricing Model ✅ COMPLETE

### Feature Engineering — DONE

- [x] Features: sqft, lot_sqft_log (log-transformed), beds, baths, age, hoa, zip (one-hot, 6 ZIPs)
- [x] Low-sample ZIPs excluded from training (< 30 transactions)
- [x] 80/20 stratified split by ZIP

### Model Training — DONE

- [x] OLS via statsmodels — R² = 0.83 on test set
- [x] All coefficients statistically significant (p < 0.001) except zip_95746 (p = 0.98 — adjacent to base ZIP 95650, expected)
- [x] RMSE: $190,620 | MAE: $126,571 | MAPE: 12.5% overall
- [x] Residual plot saved to `models/residuals.png`
- [x] Coefficients saved to `models/coefficients.json`

### Per-ZIP Accuracy — DONE

| ZIP | MAPE | RMSE | Tier |
|---|---|---|---|
| 95650 | 14.6% | $302,592 | low |
| 95661 | 10.8% | $126,281 | low |
| 95677 | 11.4% | $169,287 | low |
| 95678 | 16.9% | $135,774 | low |
| 95746 | 11.4% | $291,519 | low |
| 95765 | 12.4% | $137,713 | low |

All ZIPs in low tier — expected with Redfin-only data. MLS features will push to medium/high.

### Prediction Function (`predict.py`) — DONE

- [x] Function: property features → point estimate + 90% prediction interval
- [x] Per-ZIP RMSE used for interval width (not global RMSE)
- [x] Client-ready sentence output adapts to confidence tier
- [x] Interactive mode: `python src/predict.py --interactive`
- [x] Test cases verified across all 6 ZIPs — estimates in correct range

---

## Sunday Morning — One-Pager Generator ✅ COMPLETE

### Template and PDF Generation (`onepager.py`) — DONE

- [x] Three confidence tier templates (high / medium / low) in HTML → PDF via weasyprint
- [x] Jinja2 templating with Playfair Display + DM Sans typography
- [x] Quarterly trend bar chart built from transactions table
- [x] Neighborhood stats: median price, n transactions, avg DOM
- [x] QR codes generated per property — pointing to `http://localhost:8000/value/<address-slug>`
- [x] Agent contact info configured (name, phone, email) in onepager.py constants
- [x] Test mode: `python src/onepager.py --test` generates one PDF per ZIP
- [x] Single property: `python src/onepager.py --address "..." --zip XXXXX --sqft N ...`
- [x] Batch by ZIP: `python src/onepager.py --zip 95746 --limit 50`
- [x] PDFs output to `output/` directory

**Current status:** All ZIPs using low-tier template (neighborhood data + walkthrough CTA). Design refinement pending.

**TODO before door knocking:** Update `AGENT_NAME`, `AGENT_PHONE`, `AGENT_EMAIL` constants at top of `onepager.py`. Swap `BASE_URL` from localhost to real domain when landing page is built. Or hook into `config.json` (see below).

---

## Sunday — Local Control Panel ✅ COMPLETE

### Flask Dashboard (`app.py` + `templates/`) — DONE

- [x] Flask app wrapping existing pipeline modules
- [x] Running on `localhost:5050` (port 5000 taken by macOS AirPlay)
- [x] Price Estimator panel — live prediction with full coefficient breakdown
- [x] One-Pager Studio — color pickers, font selectors, section toggles, live HTML preview in iframe
- [x] Model Tuning — feature toggles, confidence tier thresholds, training params, per-ZIP accuracy table, coefficient table
- [x] Agent & Branding — name, phone, email, tagline, license, brokerage
- [x] Data Explorer — browse transactions by ZIP, summary stats
- [x] All settings persist to `config.json` in project root
- [x] `config.json` auto-created on first save with sensible defaults

### Known issues / next steps for dashboard

- [ ] `onepager.py` still uses hardcoded constants — needs to load from `config.json` instead
- [ ] One-pager preview in dashboard uses a simplified template (`onepager_preview.html`), not the exact same templates as `onepager.py` — these should converge
- [ ] No "Generate PDF" button in dashboard yet — preview is HTML only
- [ ] No batch generation UI — still CLI only (`python src/onepager.py --zip 95746`)

---

## Sunday Afternoon — Remaining Work

### Market Data Enrichment Experiment ✅ COMPLETE (2026-03-15)

Built `src/enrich.py` to enrich transactions with two external data sources:
- **FRED 30-year mortgage rate** — auto-downloads CSV, joins by nearest Thursday ≤ sale_date. 100% fill rate.
- **Redfin market conditions** (Placer County, 4-week rolling) — weeks of supply, DOM, sale-to-list, price drops, off-market speed. 99.5% fill rate.

**Experiment results:**
- All 5 Redfin market columns are county-level averages → identical for every transaction in the same period → no property-level signal → added noise, not accuracy.
- Mortgage rate alone: -$18,912/point (p=0.054), improved 4 of 6 ZIPs by 0.1–0.6pp.
- **Decision:** Keep `mortgage_rate` only. Redfin market columns remain in DB for future lead scoring use.
- Full results documented in data-dictionary.md Sources 1b and 1c.

**Key lesson:** County-level market signals don't help a hedonic pricing model. The variance that matters is property-level (pool, garage, condition). MLS features are still the path to medium/high confidence tiers.

### Batch Route Generator

- [ ] Script: street name or neighborhood → pulls parcels → runs model → generates PDF stack
- [ ] Sort output by address number

### Assessor Data Ingest

- [ ] Build `src/ingest_parcels.py` — load Placer assessor CSV into `parcels` table
- [ ] Build `src/score_leads.py` — compute per-parcel lead scores into `lead_scores` table
- [ ] Sanity check: parcel count, purchase date distribution, tenure distribution

### Team Leader Demo Prep

- [ ] Three things to show: model stats (R² 0.83, 4,320 transactions, per-ZIP accuracy), one example one-pager, batch generation running live
- [ ] Three sentences:
  1. "I built a pricing model on public Redfin data that estimates home values within 10–17% accuracy across 6 ZIPs in Granite Bay, Roseville, and Rocklin."
  2. "It generates personalized property snapshots I use for prospecting — here's what one looks like."
  3. "With MLS data, the accuracy improves significantly and the coverage expands. Here's what I'd need."

---

## Sunday Evening — Wrap Up ✅ COMPLETE

### Commit and Document — DONE

- [x] End-to-end test: raw CSV → database → model → prediction → one-pager PDF
- [x] Write/update README
- [x] Git repo initialized and pushed to https://github.com/dmitriy-beep/re-analytics.git
- [x] `.gitignore` in place — excludes database, raw data, output PDFs, `__pycache__`
- [x] Cloned and verified working on Windows 11 PC
- [x] `requirements.txt` updated with all dependencies (pandas, statsmodels, weasyprint, qrcode, pillow, jinja2)
- [ ] Note what's next:
  - MLS data integration (pending team leader conversation)
  - Batch route generator (next on list)
  - Assessor parcel ingest + lead scoring (next on list)
  - One-pager design refinement
  - Hook onepager.py into config.json
  - Add PDF generation to dashboard
  - CRM / interaction logger (next week)
  - County recorder data (deferred — no free bulk export)
  - Landing page for QR codes
  - Investor property analyzer

### Verify Deliverables

- [x] **SQLite database** with 4,320 cleaned transactions across 6 ZIPs
- [x] **Trained pricing model** — R² 0.83, documented accuracy metrics per ZIP
- [x] **Prediction function** with point estimate and per-ZIP confidence interval
- [x] **One-pager generator** producing print-ready PDFs at appropriate confidence tiers
- [x] **Local control panel** — Flask dashboard with live estimator, one-pager preview, model tuning, settings
- [x] **Git repo** — pushed to GitHub, cloned and working on both Mac and Windows
- [ ] **Batch route generator** — not yet built
- [ ] **Assessor parcel data ingested** — downloaded, ingest script not yet built
- [ ] **Working demo** ready for team leader conversation

---

## Time Budget Summary

| Block | Status | What |
|---|---|---|
| Saturday morning | ✅ Done | Data spot-check + project setup |
| Saturday midday | ✅ Done | Cleaning pipeline, SQLite ingest |
| Saturday afternoon | ✅ Done | Pricing model, validation, accuracy gate, predict.py |
| Sunday morning | ✅ Done | One-pager template, PDF generator |
| Sunday midday | ✅ Done | Local control panel (Flask dashboard) |
| Sunday afternoon | 🔲 Remaining | Batch route generator, assessor ingest, lead scoring, demo prep |
| Sunday evening | ✅ Done | Git setup, documentation, cross-machine verification |

---

## Project File Structure (as of 2026-03-15)

```
re-analytics/
├── app.py                      # Flask dashboard — run with `python app.py`, serves on localhost:5050
├── re_analytics.db             # SQLite database (gitignored, regenerate from pipeline)
├── requirements.txt            # pip dependencies
├── README.md
├── data-dictionary.md          # Schema, sources, cleaning decisions, experiment results
├── weekend-build-sprint.md     # This file — build log and status tracker
│
├── data/
│   └── raw/
│       ├── redfin-favorites_2026-03-14-13-25-43.csv             # Redfin sold homes export (gitignored)
│       ├── MORTGAGE30US.csv                                      # FRED 30-yr mortgage rate (auto-downloaded by enrich.py)
│       ├── mortgage_rates.csv                                    # Duplicate of MORTGAGE30US.csv — can delete
│       ├── redfin_market_conditions.tsv                          # Redfin Data Center market conditions (Placer County)
│       └── Public_Parcels_-7962162285201559209.csv               # Placer County assessor parcels — 199k rows, downloaded 2026-03-15
│
├── models/
│   ├── coefficients.json       # Trained model coefficients + metadata (regenerated by model.py)
│   └── residuals.png           # Predicted vs actual scatter plot (regenerated by model.py)
│
├── output/                     # Generated one-pager PDFs (gitignored)
│
├── src/
│   ├── db.py                   # Database init + connection helper
│   ├── ingest.py               # Redfin CSV → cleaned transactions in DB
│   ├── enrich.py               # FRED mortgage rate + Redfin market conditions → enrichment columns on transactions
│   ├── model.py                # OLS hedonic model training, evaluation, --compare mode
│   ├── predict.py              # Single-property prediction with confidence interval
│   ├── onepager.py             # HTML → PDF one-pager generator (per-property or batch)
│   ├── ingest_parcels.py       # ⏳ NOT YET BUILT — Placer assessor CSV → parcels table
│   └── score_leads.py          # ⏳ NOT YET BUILT — parcels → lead_scores table
│
├── templates/
│   ├── dashboard.html          # Flask dashboard UI (Jinja2)
│   └── onepager_preview.html   # Simplified one-pager preview for dashboard iframe
│
├── check_counties.py           # One-off script — exploring Redfin market TSV county/region values
├── check_market.py             # One-off script — exploring Redfin market TSV structure
├── check_our_counties.py       # One-off script — filtering Redfin market TSV to local counties
├── check_regions.py            # One-off script — listing unique REGION_NAME values in market TSV
├── filter_counties.py          # One-off script — filtering market data to Placer County
└── setup_demo.py               # Demo setup helper (team leader conversation prep)
```

**Notes:**
- One-off `check_*.py` and `filter_*.py` scripts were used during the enrichment experiment to explore the Redfin market conditions TSV. Safe to delete or move to a `scripts/` directory — they're not part of the pipeline.
- `data/raw/mortgage_rates.csv` appears to be an earlier/duplicate download of `MORTGAGE30US.csv`. `enrich.py` only uses `MORTGAGE30US.csv`.
- `config.json` is auto-created in the project root on first dashboard save. Not shown above because it doesn't exist until you save settings. Schema documented in `data-dictionary.md`.
