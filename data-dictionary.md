# Data Dictionary

Last updated: 2026-03-15

This document tracks every data source, its columns, cleaning decisions, and known issues. Update this as you add data sources or make cleaning decisions so Claude has current context in every conversation.

---

## Current Data Status Summary

| Source | Status | Coverage | File |
|---|---|---|---|
| Redfin Sold CSV | ✅ Loaded into DB | 12 ZIPs across Granite Bay, Roseville, Rocklin, Loomis | `data/raw/redfin-favorites_2026-03-14-13-25-43.csv` |
| FRED Mortgage Rate | ✅ Enriched into transactions | 100% fill rate (4,320/4,320) | `data/raw/MORTGAGE30US.csv` |
| Redfin Market Conditions | ✅ In DB, ❌ not in model | 99.5% fill rate — tested, no predictive value for pricing | `data/raw/redfin_market_conditions.tsv` |
| Placer County Assessor | ✅ Downloaded, ⏳ not yet ingested | 199k parcels, all Placer County | `data/raw/Public_Parcels_-7962162285201559209.csv` |
| Sacramento County Assessor | ⏳ Deferred | Low overlap with current model ZIPs | — |
| County Recorder (deed of trust) | ❌ No bulk export available | Index-only online for both counties; bulk data requires paid vendor | — |
| MLS Data | ❌ Pending team leader conversation | — | — |

---

## Current Build Status (as of 2026-03-15)

The core pipeline is built and working end-to-end. All code lives at:
`/Users/dmitriy/RealEstateTools/weekend3-14/re-analytics/`

### What's Done

**Database (`src/db.py`)** — SQLite initialized with all tables: `transactions`, `parcels`, `model_accuracy`, `interactions`, `lead_scores`, `follow_ups`.

**Ingest (`src/ingest.py`)** — Redfin CSV cleaning and loading pipeline. Idempotent — safe to re-run. Loaded 4,320 transactions from `data/raw/redfin-favorites_2026-03-14-13-25-43.csv`.

**Enrichment (`src/enrich.py`)** — Enriches `transactions` with FRED mortgage rate (auto-downloads CSV) and Redfin market conditions (TSV). Adds 6 columns to DB. Idempotent. Run: `python src/enrich.py`. Status check: `python src/enrich.py --status`. See Source 1b and 1c in this document for experiment results — only `mortgage_rate` is used by the model; the five Redfin market columns are in the DB but excluded from pricing (useful for future lead scoring).

**Pricing model (`src/model.py`)** — OLS hedonic regression via statsmodels. R² = 0.83 on test set. Trained on 6 ZIPs. Features: sqft, lot_sqft_log, beds, baths, age, hoa, mortgage_rate, ZIP dummies. Coefficients saved to `models/coefficients.json`. Per-ZIP accuracy saved to `model_accuracy` table. Comparison mode: `python src/model.py --compare` trains baseline and enriched side by side.

**Prediction function (`src/predict.py`)** — Takes `PropertyFeatures` dataclass, returns point estimate + 90% prediction interval (uses per-ZIP RMSE for interval width) + confidence tier + client-ready sentence. Mortgage rate auto-fetched from DB if not provided. Test: `python src/predict.py`. Interactive: `python src/predict.py --interactive`.

**One-pager generator (`src/onepager.py`)** — HTML → PDF via weasyprint. Three confidence tier templates. Jinja2, Playfair Display + DM Sans typography. Quarterly trend bar chart from transactions table. QR codes via `qrcode` library pointing to `http://localhost:8000/value/<slug>` (localhost for now). Test: `python src/onepager.py --test`. Batch: `python src/onepager.py --zip 95746`.

**Local control panel (`app.py` + `templates/`)** — Flask dashboard running on `localhost:5050`. Five panels: Price Estimator (with coefficient breakdown and mortgage rate input), One-Pager Studio (live preview, design tokens, section toggles), Model Tuning (feature toggles including mortgage_rate, tier thresholds, training params), Agent & Branding, Data Explorer. All settings persist to `config.json`.

### Current Model Performance

All 6 ZIPs in **low confidence tier** (MAPE 10.4–16.5%). Expected — Redfin data lacks pool, solar, garage, condition features. One-pagers use neighborhood data template + walkthrough CTA. Will move to medium/high tier after MLS integration.

| ZIP | Area | Transactions | MAPE | Tier |
|---|---|---|---|---|
| 95765 | Roseville NW | 1,102 | 12.4% | low |
| 95677 | Rocklin | 735 | 11.4% | low |
| 95678 | Roseville central | 731 | 16.9% | low |
| 95661 | Roseville west | 701 | 10.8% | low |
| 95746 | Granite Bay | 669 | 11.4% | low |
| 95650 | Loomis/Granite Bay | 357 | 14.6% | low |

### What's Not Built Yet

- `src/ingest_parcels.py` — parcel ingest pipeline (next up)
- `src/score_leads.py` — lead scoring model (next up)
- Batch route generator
- One-pager design refinement
- Hook `onepager.py` into `config.json` (currently still uses hardcoded constants)
- CRM / interaction logger
- Landing page for QR codes
- Investor property analyzer

### Key Config to Update Before Going Live

In `src/onepager.py` (top of file):
- `AGENT_NAME`, `AGENT_PHONE`, `AGENT_EMAIL` — currently placeholders
- `BASE_URL` — currently `http://localhost:8000/value`, swap to real domain when ready

Or better: replace those constants with a `config.json` load (see Config Schema below).

---

## Source 1: Redfin Sold Homes CSV

**Origin:** Downloaded from redfin.com, filtered to "Sold", last 3 years
**Coverage:** Granite Bay, Roseville, Rocklin, Loomis — 12 ZIPs in export, 6 with sufficient sample size for modeling
**Geographic note:** Primarily Placer County territory. Does NOT currently include Sacramento County. Expand when MLS access is secured.
**File:** `data/raw/redfin-favorites_2026-03-14-13-25-43.csv`
**Download date:** 2026-03-14
**Row count (raw):** 4,836
**Row count (after cleaning):** 4,320

### ZIP Codes in This Export

| ZIP | City | Transactions (cleaned) | Model Status |
|---|---|---|---|
| 95765 | Roseville (northwest) | 1,102 | ✅ In model |
| 95677 | Rocklin | 735 | ✅ In model |
| 95678 | Roseville (central) | 731 | ✅ In model |
| 95661 | Roseville (west) | 701 | ✅ In model |
| 95746 | Granite Bay | 669 | ✅ In model |
| 95650 | Loomis / Granite Bay area | 357 | ✅ In model |
| 95747 | Roseville (west) | 11 | ⚠️ Excluded — low sample |
| 95630 | Folsom area | 8 | ⚠️ Excluded — low sample |
| 95610 | Citrus Heights area | 3 | ⚠️ Excluded — low sample |
| 95663 | Penryn | 1 | ⚠️ Excluded — low sample |
| 95662 | Orangevale | 1 | ⚠️ Excluded — low sample |
| 98678 | Invalid ZIP | 1 | ⚠️ Excluded — bad data |

**Note:** The export contained more ZIPs than the original 5 targeted. All 6 valid ZIPs are included in the model. Low-sample ZIPs are stored in the DB but excluded from model training.

### Actual Column Headers (confirmed from CSV)

```
SALE TYPE, SOLD DATE, PROPERTY TYPE, ADDRESS, CITY, STATE OR PROVINCE,
ZIP OR POSTAL CODE, PRICE, BEDS, BATHS, LOCATION, SQUARE FEET, LOT SIZE,
YEAR BUILT, DAYS ON MARKET, $/SQUARE FEET, HOA/MONTH, STATUS,
NEXT OPEN HOUSE START TIME, NEXT OPEN HOUSE END TIME,
URL (SEE https://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING),
SOURCE, MLS#, FAVORITE, INTERESTED, LATITUDE, LONGITUDE
```

**Note:** Column is `SOLD DATE` not `SALE DATE`. ingest.py handles this mapping correctly.

### Column Mapping

| Redfin Column | Our Schema Column | Type | Notes |
|---|---|---|---|
| SOLD DATE | sale_date | date | Mapped as-is; parsed to YYYY-MM-DD |
| PROPERTY TYPE | property_type | string | Keep: Single Family Residential, Townhouse, Condo/Co-op |
| ADDRESS | address | string | |
| CITY | city | string | |
| ZIP OR POSTAL CODE | zip | string | Kept as string, zero-padded to 5 digits |
| PRICE | sale_price | integer | |
| BEDS | beds | integer | |
| BATHS | baths | float | Half baths as 0.5 |
| SQUARE FEET | sqft | integer | |
| LOT SIZE | lot_sqft | integer | |
| YEAR BUILT | year_built | integer | |
| DAYS ON MARKET | days_on_market | integer | |
| HOA/MONTH | hoa | integer | Missing filled with 0 |
| LATITUDE | latitude | float | |
| LONGITUDE | longitude | float | |
| SALE TYPE | — | — | Dropped |
| STATE OR PROVINCE | — | — | Dropped |
| LOCATION | — | — | Dropped |
| $/SQUARE FEET | — | — | Dropped (derived) |
| STATUS | — | — | Dropped |
| NEXT OPEN HOUSE START TIME | — | — | Dropped |
| NEXT OPEN HOUSE END TIME | — | — | Dropped |
| URL (...) | — | — | Dropped |
| SOURCE | — | — | Dropped |
| MLS# | — | — | Dropped |
| FAVORITE | — | — | Dropped |
| INTERESTED | — | — | Dropped |

### Cleaning Decisions

- ✅ Dropped sales under $50,000 as likely non-arm's-length transactions
- ✅ Dropped sqft under 400 or over 10,000
- ✅ Dropped price/sqft under $50 or over $1,500
- ✅ Deduplicated on: address + sale_date
- ✅ Filtered to property types: Single Family Residential, Townhouse, Condo/Co-op
- ✅ Dropped 507 rows missing required fields (sale_price, sqft, beds, sale_date, address)
- ✅ Dropped 6 non-residential rows
- ✅ Dropped 2 rows with sqft outside bounds
- ✅ Dropped 1 row with $/sqft outside bounds

### Known Issues

- 98678 is an invalid ZIP code — one row, likely a data entry error in Redfin. Stored in DB but excluded from model.
- 507 rows (10% of raw) dropped for missing required fields — normal for Redfin exports.
- 4,318 of 4,320 cleaned rows are Single Family Residential. Model is effectively SFR-only for now.
- Redfin CSV has a "PAST SALE" metadata row before the header — ingest.py handles this with skiprows logic.

### Model Implications (current data)

Model trained on 6 ZIPs with 30+ transactions. All 6 ZIPs currently land in **low confidence tier** (MAPE 10.4–16.5%). This is expected — Redfin data lacks pool, solar, garage, condition features. MLS data will improve accuracy materially. See model_accuracy table for current per-ZIP metrics.

---

## Source 1b: FRED 30-Year Mortgage Rate (enrichment)

**Status:** ✅ Loaded — enriches `transactions` table via `src/enrich.py`
**Origin:** FRED (Federal Reserve Economic Data) — `https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US`
**File:** `data/raw/MORTGAGE30US.csv` (auto-downloaded by `enrich.py` if missing)
**Coverage:** Weekly observations, every Thursday. Full history back to 1971.
**Join method:** `merge_asof` — each transaction gets the rate from the nearest Thursday on or before `sale_date`.
**Fill rate:** 4,320 / 4,320 (100%)

### Column Added

| Column | Type | Description |
|---|---|---|
| `mortgage_rate` | REAL | 30-year fixed rate (%) at time of sale |

### Model Impact

Coefficient: **-$18,912 per percentage point** (p = 0.054). Economically meaningful — each point of rate roughly reduces buyer purchasing power by ~$20K in this price range. Borderline statistical significance because the 3-year training window only captures one major rate cycle (2022–2023 spike). Will tighten as more data accumulates.

**Kept in model** — improves 4 of 6 ZIPs, provides rate-environment adjustment when predicting at today's rates against training data that includes the low-rate era.

---

## Source 1c: Redfin Market Conditions (enrichment — TESTED, NOT IN MODEL)

**Status:** ✅ Loaded into `transactions` table but **excluded from model** after testing.
**Origin:** Redfin Data Center — market conditions download
**File:** `data/raw/redfin_market_conditions.tsv`
**Filter:** `DURATION == '4 weeks'` and `REGION_NAME == 'Placer County, CA'`
**Join method:** Row where `PERIOD_BEGIN <= sale_date <= PERIOD_END`
**Fill rate:** 4,297 / 4,320 (99.5%)

### Columns Added (in DB, not used by model)

| Column | Type | Description |
|---|---|---|
| `weeks_of_supply` | REAL | Weeks of inventory at current sales pace |
| `median_days_on_market` | REAL | Median DOM for the 4-week period |
| `sale_to_list_ratio` | REAL | Avg sale price / list price ratio |
| `pct_listings_price_drop` | REAL | % of active listings with price reductions |
| `off_market_in_two_weeks` | REAL | % of listings going under contract within 14 days |

### Why These Were Removed from the Pricing Model (2026-03-15)

All five columns tested with p-values > 0.05 (three above 0.08, two above 0.85). Adding all five increased MAPE in 2 of 6 ZIPs and worsened overall MAPE by +0.1pp. Root cause: **these are county-level 4-week rolling averages** — every transaction closing in the same window gets identical values. They capture macro timing but cannot distinguish between properties within a period. The model needs property-level variation (pool, garage, condition) to improve, not market-level signals.

Full experiment results:

| Feature | Coefficient | p-value | Verdict |
|---|---|---|---|
| `mortgage_rate` | -$33,463 | 0.005 | ✅ Kept (alone: -$18,912, p=0.054) |
| `weeks_of_supply` | -$33 | 0.986 | ❌ Noise |
| `median_days_on_market` | -$1,122 | 0.049 | ❌ Barely significant, no MAPE improvement |
| `sale_to_list_ratio` | -$2,201,734 | 0.079 | ❌ Unstable, high variance |
| `pct_listings_price_drop` | +$621,929 | 0.124 | ❌ Not significant |
| `off_market_in_two_weeks` | +$94 | 0.858 | ❌ Noise |

**Future use:** These columns remain in the DB and will be useful for the **lead scoring model** — `weeks_of_supply` and `sale_to_list_ratio` are strong signals for market temperature in prospecting conversations, just not for predicting individual sale prices.

---

## Source 2: Sacramento County Assessor Parcel Data

**Status:** ⏳ Deferred — low priority until model ZIPs expand south toward Folsom/Citrus Heights/Sacramento.
**Access method:** Sacramento County NextRequest portal (saccounty.nextrequest.com) — free bulk electronic records request, few days turnaround.
**What's available:** Assessment roll — APN, owner name (secured only), property address, mailing address, assessed values, exemption values, tax rate area, zoning code, land use code. Building characteristics (beds/baths) limited; fee charged for full characteristics.
**Note:** Owner name IS available in bulk export here, unlike Placer County free download.

---

## Source 3: Placer County Assessor Parcel Data

**Status:** ✅ Downloaded — `data/raw/Public_Parcels_-7962162285201559209.csv`. Not yet ingested into DB.
**Origin:** Placer County GIS Open Data portal — `gis-placercounty.opendata.arcgis.com` → "Public_Parcels" dataset
**Download date:** 2026-03-15
**Last updated by county:** 2026-03-15 (updated daily per portal)
**Row count (raw):** 199,000 parcels (all property types, all of Placer County including Tahoe)
**Format:** CSV with geometry size columns — no GIS software needed

### Use Code Distribution (residential categories)

| Use_Cd_N | Count | Include in ingest? |
|---|---|---|
| SINGLE FAM RES, HALF PLEX | 143,416 | ✅ Yes — standard SFR (name is misleading) |
| SINGLE FAM RES, CONDO | 7,747 | ✅ Yes |
| 2 SINGLE FAM RES, DUPLEX | 3,216 | ❌ Skip for now |
| 3 SINGLE FAM RES, TRIPLEX | 285 | ❌ Skip |
| MOBILE HOME OUTSIDE OF PARK | 342 | ❌ Skip |
| RESIDENCE ON COMMERCIAL LAND | 455 | ❌ Skip |

Core residential filter: `Use_Cd_N IN ('SINGLE FAM RES, HALF PLEX', 'SINGLE FAM RES, CONDO')` → ~151k parcels county-wide

### ZIP Distribution (SFR only)

| ZIP | SFR Parcels | In Model? |
|---|---|---|
| 95747 | 32,385 | ❌ No Redfin data yet |
| 95648 | 22,707 | ❌ No Redfin data yet |
| 95765 | 11,863 | ✅ |
| 95678 | 10,270 | ✅ |
| 95661 | 8,803 | ✅ |
| 95677 | 8,655 | ✅ |
| 95603 | 8,057 | ❌ Auburn — out of market |
| 95746 | 7,688 | ✅ |
| 95650 | 4,178 | ✅ |

6 model ZIPs = ~51,457 SFR parcels. ~8% 3-year turnover vs Redfin transactions — reasonable. Tahoe ZIPs (96xxx) — exclude from ingest, different market.

### Raw Column Headers

```
OBJECTID, GlobalID, APN, Book, BookPage, created_date, last_edited_date, ParcelLevel,
GIS_Acres, FEEPARCEL, Jurisdiction, TransactionDt, TRA, TaxableX, Tax_Cd, Tax_Cd_N,
Use_Cd, Use_Cd_N, Acres, EffectiveYr, SitusZip, StreetNum, StreetName, StreetDir,
StreetType, Sp_Apt, Community, Asmt, Asmt_Desc, LandValue, Structure, LandSF,
StructureSF, AprID, Neighborhood_Cd, Comments, MailingAdr1, MailingAdr2, MailingCity,
MailingState, MailingZip, SitusID, AsmtStatus, FormattedSitus1, FormattedSitus2,
SitusAddressFull, VOIDMapBookPage, TTC, Assessment, BkPg_Url, HistoricalAssrMap,
Shape__Area, Shape__Length
```

### Column Mapping to `parcels` Schema

| Raw Column | parcels Column | Type | Notes |
|---|---|---|---|
| APN | parcel_id | string | 12-digit, primary identifier |
| SitusAddressFull | address | string | Full formatted address |
| Community | city | string | |
| SitusZip | zip | string | Mixed types — cast str, strip, first 5 chars |
| — | owner_name | string | NOT IN FREE DOWNLOAD — requires paid data request form |
| MailingAdr1/2, MailingCity/State/Zip | — | string | Keep for absentee flag logic; not a parcels schema column |
| TransactionDt | purchase_date | date | Last transfer date. Parse datetime, extract date. 3,046 nulls. |
| — | purchase_price | integer | NOT AVAILABLE — CA non-disclosure state |
| LandValue + Structure | assessed_value | integer | Sum of both fields |
| StructureSF | sqft | integer | |
| LandSF | lot_sqft | integer | |
| EffectiveYr | year_built | integer | Proxy — verify against Redfin year_built for matched addresses |
| Use_Cd_N | property_type | string | Map to 'Single Family' / 'Condo' |
| — | latitude / longitude | real | NOT IN CSV — geocode from address if needed later |

**data_source value:** `'placer_assessor'`

### Computed Fields (derive at ingest, store in parcels or lead_scores)

- **tenure_years** — `(today - purchase_date).days / 365` — how long current owner has held. Core lead scoring input.
- **absentee_flag** — `1` if MailingZip != SitusZip, else `0`. Investor/absentee owner signal.

### Known Issues and Gotchas

- `FEEPARCEL` and `SitusZip` have mixed types — always use `low_memory=False` on read
- Owner name NOT in free GIS download (CA Public Records Act, Govt Code 6254.21). Available via paid data request form: placer.ca.gov → Assessor → Forms → "Request For Data Form", contact (530) 889-4300
- `TransactionDt` most recent batch entries show 9/9/2025 timestamps — these are GIS batch update timestamps, not transaction dates. Actual transaction dates spread across years. Verify distribution before finalizing scoring curve.
- Physical characteristics (beds, baths) currently unavailable even on per-parcel lookup due to system maintenance as of 2026-03-15
- No purchase price (CA non-disclosure)
- No pool, garage, solar — MLS only
- Tahoe ZIPs (96xxx) are a different market — exclude at ingest
- `EffectiveYr` may be year of last assessment not year built — verify against Redfin for matched addresses

---

## Source 4: County Recorder Data (deed of trust recordings)

**Status:** ❌ No bulk export available from either county without payment. Deferred.

Both Sacramento and Placer County recorders offer online index search only — document type, recording date, party names. Cannot view actual documents or export in bulk. Structured bulk deed data requires a paid vendor (ParcelQuest, ATTOM, CoreLogic).

**Decision:** Defer. Equity and rate-lock signals can be approximated from assessor `TransactionDt` + FRED mortgage rate at that date — already have both. Revisit after v1 lead scoring model is built and validated.

**Why vendors have this data:** They've built automated pipelines pulling the same public county sources — assessor bulk exports + recorder document feeds — normalized across all 58 CA counties. The product is the aggregation and normalization, not the underlying data, which is all technically public.

---

## Source 5: MLS Data

**Status:** ❌ Pending — contingent on team leader conversation.
**Why it matters:** Significantly more features than Redfin (pool, solar, garage count, lot backing, condition, etc.) and faster/more complete coverage. Expected to drop MAPE from ~12% to 7–9%, pushing most ZIPs into medium or high confidence tier.
**Design note:** The `transactions` table `data_source` column is the hook for MLS data. When MLS is added, it loads into the same table with `data_source = 'mls'`. No schema changes required.

---

## Current Model Accuracy (as of 2026-03-15)

Trained on Redfin data + FRED mortgage rate enrichment. Overall R² = 0.83.

| ZIP | N (test) | MAPE | RMSE | Tier |
|---|---|---|---|---|
| 95650 | 69 | 13.9% | $302,158 | low |
| 95661 | 138 | 10.4% | $126,113 | low |
| 95677 | 144 | 11.8% | $169,460 | low |
| 95678 | 144 | 16.5% | $135,619 | low |
| 95746 | 127 | 11.3% | $291,491 | low |
| 95765 | 201 | 12.4% | $137,823 | low |

**All ZIPs in low tier** due to missing MLS features (pool, solar, garage, condition). Mortgage rate enrichment improved 4 of 6 ZIPs by 0.1–0.6pp. One-pagers currently use neighborhood data template with walkthrough CTA. Expected to reach medium/high tier after MLS integration.

---

## SQLite Schema

### transactions table
```sql
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
    data_source TEXT NOT NULL,  -- 'redfin', 'mls', etc.
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### parcels table
```sql
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
    data_source TEXT NOT NULL,  -- 'sac_assessor', 'placer_assessor'
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### model_accuracy table
```sql
CREATE TABLE model_accuracy (
    zip TEXT PRIMARY KEY,
    n_transactions INTEGER,
    mape REAL,
    rmse REAL,
    confidence_tier TEXT,  -- 'high', 'medium', 'low'
    last_updated TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### Future tables (created in DB, not yet populated)
- `interactions` — door knock logs, showing feedback, contact info
- `lead_scores` — per-parcel propensity scores
- `follow_ups` — triggered follow-up queue

---

## Config Schema (`config.json`)

The local control panel (`app.py`) persists all settings to `config.json` in the project root. This is the single source of truth for agent info, one-pager design, and model parameters. Update via the dashboard UI or edit the file directly.

**Note:** `config.json` is auto-created on first save from the dashboard. If it doesn't exist yet, the app uses defaults defined in `app.py` → `DEFAULT_CONFIG`.

```json
{
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
    "show_qr_code": true,
    "show_trend_chart": true,
    "show_neighborhood_stats": true,
    "show_confidence_interval": true,
    "show_comp_table": true,
    "base_url": "http://localhost:5050/value",
    "footer_text": "Estimates based on local transaction data. Not an appraisal.",
    "cta_text": "Schedule a walkthrough for a refined estimate"
  },
  "model": {
    "confidence_tiers": {
      "high": { "max_mape": 7.0, "label": "High Confidence" },
      "medium": { "max_mape": 12.0, "label": "Medium Confidence" },
      "low": { "max_mape": 100.0, "label": "Low Confidence" }
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

### Config field reference

**agent** — Contact info used in one-pagers, landing pages, reports. Update via Agent & Branding panel or edit directly.

**onepager** — Design tokens and section toggles for PDF output. Colors are hex strings. Fonts must be available via Google Fonts. Boolean toggles control which sections appear in the one-pager. `base_url` is the root URL for QR codes.

**model** — Controls prediction behavior. `confidence_tiers` maps MAPE cutoffs to tier labels (determines which one-pager template a ZIP gets). `features_enabled` toggles regression features on/off — save here, then re-run `python src/model.py` to retrain. `prediction_interval_pct` controls whether the interval is 80%, 90%, or 95%.

---

## Local Environment Notes

- **Git repo:** https://github.com/dmitriy-beep/re-analytics.git (private)
- **Two machines:** Mac (primary dev) and Windows 11 PC — always `git pull` before starting work, `git push` when done
- **Gitignored (must regenerate after clone):** `re_analytics.db`, `data/raw/`, `output/`, `__pycache__/`
- **Setup after fresh clone:** Copy Redfin CSV to `data/raw/`, then run `pip install -r requirements.txt`, `python src/db.py`, `python src/ingest.py data/raw/<filename>.csv`, `python src/model.py`
- **Project path (Mac):** `/Users/dmitriy/RealEstateTools/weekend3-14/re-analytics/`
- **Project path (Windows):** `C:\RealEstateTools\weekend3-14\re-analytics\`
- **Python (Mac):** Anaconda base environment
- **Python (Windows):** Python 3.11 (system install)
- **Dashboard:** Flask on `localhost:5050` (port 5000 is taken by macOS AirPlay Receiver on Mac; 5050 works on both machines)
- **Launch:** `python app.py` from project root
- **Database:** `re_analytics.db` in project root (SQLite)
- **Templates:** `templates/dashboard.html` and `templates/onepager_preview.html` must be in `templates/` subdirectory (Flask convention)
- **Coefficients:** `models/coefficients.json` — nested structure with `coefficients` dict containing per-feature dicts with `coef`, `ci_lower`, `ci_upper`, `pvalue` fields (not flat numbers)
