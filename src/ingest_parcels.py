"""
ingest_parcels.py — Load Placer County assessor CSV into the parcels table.

Column mapping (confirmed from --peek on 2026-03-16):
  APN              → parcel_id
  FormattedSitus1  → address  (fallback: StreetNum+StreetDir+StreetName+StreetType)
  Jurisdiction     → city
  SitusZip         → zip
  TransactionDt    → purchase_date
  LandValue + Structure → assessed_value (summed)
  StructureSF      → sqft
  LandSF           → lot_sqft
  EffectiveYr      → year_built
  Use_Cd / Use_Cd_N → property_type filter

Residential filter (Use_Cd):
  INCLUDE: 01 (SFR half-plex), 02 (duplex), 04 (condo/SFR), 07 (aux improvement),
           08 (mobile home outside park)
  EXCLUDE: 03 (triplex), 05 (apartments 4+), 06 (timeshare), 09 (MH in park),
           28 (MH park), and all non-residential codes

Usage:
    python src/ingest_parcels.py                    # load/re-load parcels
    python src/ingest_parcels.py --file path.csv    # specify CSV path
    python src/ingest_parcels.py --status           # show DB parcel counts
    python src/ingest_parcels.py --peek             # print raw CSV headers
    python src/ingest_parcels.py --dry-run          # count what would load, no DB writes
    python src/ingest_parcels.py --show-use-codes   # print use code distribution after load
"""

import sys
import csv
import re
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
try:
    from db import get_connection
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from db import get_connection

DEFAULT_CSV = Path(__file__).parent.parent / "data" / "raw" / "Public_Parcels_-7962162285201559209.csv"
MODEL_ZIPS = {"95765", "95677", "95678", "95661", "95746", "95650"}
DATA_SOURCE = "placer_assessor"

# ---------------------------------------------------------------------------
# Residential filter — Placer County Use_Cd
# ---------------------------------------------------------------------------
# INCLUDED use codes (single-family and owner-occupied residential)
INCLUDE_USE_CD = {
    "01",   # SINGLE FAM RES, HALF PLEX  — largest group, core SFR
    "02",   # 2 SINGLE FAM RES, DUPLEX   — owner-occupied duplexes
    "04",   # SINGLE FAM RES, CONDO      — condos / attached SFR
    "07",   # RESIDENTIAL, AUXILIARY IMP — main house + ADU/guest house
    "08",   # MOBILE HOME OUTSIDE OF PARK — real property owner
}

# EXCLUDED use codes (multi-family, commercial, parks, timeshares)
EXCLUDE_USE_CD = {
    "03",   # TRIPLEX
    "05",   # APARTMENTS, 4 UNITS OR MORE
    "06",   # TIMESHARES
    "09",   # MOBILE HOME IN M H PARK (renting land)
    "28",   # MOBILE HOME PARK (the park itself)
}
# Anything not in INCLUDE and not in EXCLUDE will also be excluded
# (default-deny for unknown codes)


def is_residential(use_cd: str) -> bool:
    """Return True only for explicitly included residential use codes."""
    cd = str(use_cd).strip().lstrip("0") or "0"
    # Normalize: "01" → "1", "04" → "4", etc.
    return cd in {str(int(c)) for c in INCLUDE_USE_CD}


# ---------------------------------------------------------------------------
# Value cleaners
# ---------------------------------------------------------------------------
def clean_zip(raw) -> str | None:
    if not raw:
        return None
    z = re.sub(r"[^0-9]", "", str(raw))[:5]
    return z if len(z) == 5 else None


def clean_int(raw) -> int | None:
    if raw is None:
        return None
    s = str(raw).strip().replace(",", "").replace("$", "").split(".")[0]
    if not s or s in ("-", ""):
        return None
    try:
        return int(s)
    except ValueError:
        return None


def clean_date(raw) -> str | None:
    if not raw:
        return None
    s = str(raw).strip().split(" ")[0]
    if not s or s in ("0", "00/00/0000"):
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y", "%Y%m%d", "%m/%d/%y"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.year < 1900 or dt.year > datetime.now().year + 1:
                return None
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def build_address(row: dict) -> str:
    formatted = row.get("FormattedSitus1", "").strip()
    if formatted:
        return " ".join(formatted.split())
    parts = [
        row.get("StreetNum", "").strip(),
        row.get("StreetDir", "").strip(),
        row.get("StreetName", "").strip(),
        row.get("StreetType", "").strip(),
        row.get("Sp_Apt", "").strip(),
    ]
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Main ingest
# ---------------------------------------------------------------------------
def ingest(csv_path: Path, dry_run: bool = False) -> dict:
    if not csv_path.exists():
        print(f"  ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    print(f"  Loading: {csv_path}")
    print(f"  File size: {csv_path.stat().st_size / 1_000_000:.1f} MB")
    if dry_run:
        print(f"  *** DRY RUN — no DB writes ***\n")
    else:
        print()

    conn = None
    cursor = None
    if not dry_run:
        conn = get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_parcels_parcel_id ON parcels(parcel_id)"
            )
            conn.commit()
        except Exception:
            pass

    stats = {
        "total_rows": 0,
        "loaded": 0,
        "skipped_zip": 0,
        "skipped_residential": 0,
        "skipped_no_address": 0,
        "per_zip": {},
        "use_code_counter": Counter(),          # all codes seen, all ZIPs
        "use_code_loaded": Counter(),           # codes that made it into DB
        "use_code_skipped_res": Counter(),      # codes excluded by residential filter
    }

    batch = []

    def flush():
        if not batch or dry_run:
            batch.clear()
            return
        cursor.executemany("""
            INSERT OR IGNORE INTO parcels
              (parcel_id, address, city, zip, owner_name, purchase_date,
               purchase_price, assessed_value, sqft, lot_sqft, year_built,
               beds, baths, property_type, data_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, batch)
        conn.commit()
        batch.clear()

    with open(csv_path, encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["total_rows"] += 1
            if stats["total_rows"] % 10000 == 0:
                print(f"  ... {stats['total_rows']:,} rows read, {stats['loaded']:,} loaded")

            use_cd = row.get("Use_Cd", "").strip()
            use_cd_n = row.get("Use_Cd_N", "").strip()
            code_label = f"{use_cd}|{use_cd_n}"
            stats["use_code_counter"][code_label] += 1

            if not is_residential(use_cd):
                stats["skipped_residential"] += 1
                stats["use_code_skipped_res"][code_label] += 1
                continue

            zip_code = clean_zip(row.get("SitusZip", ""))
            if not zip_code or zip_code not in MODEL_ZIPS:
                stats["skipped_zip"] += 1
                continue

            address = build_address(row)
            if not address:
                stats["skipped_no_address"] += 1
                continue

            parcel_id = row.get("APN", "").strip() or None
            city = row.get("Jurisdiction", "").strip() or None
            if city:
                city = city.title()

            purchase_date = clean_date(row.get("TransactionDt", ""))

            land_val = clean_int(row.get("LandValue", "")) or 0
            structure_val = clean_int(row.get("Structure", "")) or 0
            assessed_value = (land_val + structure_val) or None

            sqft = clean_int(row.get("StructureSF", ""))
            lot_sqft = clean_int(row.get("LandSF", ""))

            year_built = clean_int(row.get("EffectiveYr", ""))
            if year_built and (year_built < 1850 or year_built > 2026):
                year_built = None

            stats["loaded"] += 1
            stats["per_zip"][zip_code] = stats["per_zip"].get(zip_code, 0) + 1
            stats["use_code_loaded"][code_label] += 1

            batch.append((
                parcel_id, address, city, zip_code,
                None, purchase_date, None, assessed_value,
                sqft, lot_sqft, year_built,
                None, None, use_cd_n, DATA_SOURCE,
            ))

            if len(batch) >= 500:
                flush()

    flush()
    if conn:
        conn.close()
    return stats


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def print_status():
    try:
        conn = get_connection()
        rows = conn.execute("""
            SELECT zip, COUNT(*) as n,
                   SUM(CASE WHEN purchase_date IS NOT NULL THEN 1 ELSE 0 END) as has_date,
                   SUM(CASE WHEN sqft IS NOT NULL THEN 1 ELSE 0 END) as has_sqft,
                   SUM(CASE WHEN assessed_value IS NOT NULL THEN 1 ELSE 0 END) as has_value
            FROM parcels GROUP BY zip ORDER BY n DESC
        """).fetchall()
        conn.close()
        if not rows:
            print("  No parcels in DB yet.")
            return
        total = sum(r[1] for r in rows)
        print(f"\n  {'ZIP':<8} {'Count':>7} {'Has Date':>10} {'Has Sqft':>10} {'Has Value':>10}")
        print(f"  {'─'*52}")
        for r in rows:
            print(f"  {r[0]:<8} {r[1]:>7,} {r[2]:>10,} {r[3]:>10,} {r[4]:>10,}")
        print(f"  {'─'*52}")
        print(f"  {'TOTAL':<8} {total:>7,}")
    except Exception as e:
        print(f"  Error: {e}")


def print_use_code_report(stats: dict):
    """Print a breakdown of included vs excluded use codes in the model ZIPs."""
    # Separate loaded vs skipped-but-in-model-ZIP
    # We only have all-ZIP counters, so we'll show loaded (which IS model-ZIP filtered)
    loaded = stats["use_code_loaded"]
    if not loaded:
        print("  No use code data — run without --dry-run or check filter.")
        return

    total_loaded = sum(loaded.values())
    print(f"\n  Use codes in loaded parcels ({total_loaded:,} total, model ZIPs only):")
    print(f"  {'─'*60}")
    print(f"  {'Code|Description':<45} {'Count':>7}  {'%':>5}")
    print(f"  {'─'*60}")
    for code, count in loaded.most_common():
        pct = count / total_loaded * 100
        print(f"  {code:<45} {count:>7,}  {pct:>4.1f}%")

    # Show what was excluded (top codes, all ZIPs)
    skipped = stats["use_code_skipped_res"]
    if skipped:
        print(f"\n  Excluded use codes (non-SFR, all ZIPs, top 10):")
        print(f"  {'─'*60}")
        for code, count in skipped.most_common(10):
            print(f"  {code:<45} {count:>7,}")


def peek_headers(csv_path: Path):
    if not csv_path.exists():
        print(f"  File not found: {csv_path}")
        return
    with open(csv_path, encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        first_row = next(reader, {})
    print(f"\n  {len(headers)} columns:")
    for i, h in enumerate(headers, 1):
        print(f"  {i:3}. {h:<40}  sample: {str(first_row.get(h, ''))[:50]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Placer County assessor CSV → parcels table")
    parser.add_argument("--file", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--status", action="store_true", help="Show DB parcel counts and exit")
    parser.add_argument("--peek", action="store_true", help="Print raw CSV headers and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count what would load without writing to DB")
    parser.add_argument("--show-use-codes", action="store_true",
                        help="Print use code breakdown after loading")
    args = parser.parse_args()

    if args.status:
        print_status()
        sys.exit(0)
    if args.peek:
        peek_headers(args.file)
        sys.exit(0)

    print(f"\n  Placer County Assessor → parcels table")
    print(f"  {'─'*45}")

    stats = ingest(args.file, dry_run=args.dry_run)

    action = "Would load" if args.dry_run else "Loaded"
    print(f"\n  {'✓ DRY RUN complete' if args.dry_run else '✅  Ingest complete'}")
    print(f"  {'─'*45}")
    print(f"  Total rows:        {stats['total_rows']:>8,}")
    print(f"  {action}:           {stats['loaded']:>8,}")
    print(f"  Skipped non-SFR:   {stats['skipped_residential']:>8,}")
    print(f"  Skipped wrong ZIP: {stats['skipped_zip']:>8,}")
    print(f"  Skipped no addr:   {stats['skipped_no_address']:>8,}")
    print(f"\n  Per-ZIP:")
    for z in sorted(stats["per_zip"]):
        print(f"    {z}: {stats['per_zip'][z]:,}")

    if args.show_use_codes:
        print_use_code_report(stats)

    if not args.dry_run:
        print(f"\n  Next: python src/score_leads.py")
