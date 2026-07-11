"""
PRISM - Module 1: Data Acquisition Layer
==========================================

Responsibility of this file (and ONLY this file, per PRISM's design rule):
    Fetch, parse, cache, and load raw data.
    NO statistics, NO scoring, NO decisions happen here.

Why this separation matters:
    Every later module (returns, volatility, covariance, overlap, sectors,
    scoring, optimization) will import from this file but will NEVER
    duplicate a network call or a CSV read inside itself. This keeps
    PRISM testable: you can unit-test Module 2's math with fake/mock
    data without ever hitting the internet.

Data source:
    NAV history      -> mfapi.in   (free, public, no API key needed)
    Holdings/Sectors  -> manually curated CSVs (see README.md for why)

Author: <you>
"""

import os
import json
import csv
import urllib.request
import urllib.parse
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = "data_cache/nav"
BASE_URL = "https://api.mfapi.in/mf"


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(scheme_code):
    return os.path.join(CACHE_DIR, f"{scheme_code}.json")


# ---------------------------------------------------------------------------
# 1. Scheme discovery — find a fund's scheme_code by name
# ---------------------------------------------------------------------------
def search_scheme(fund_name_query):
    """
    Search mfapi.in for scheme codes matching a fund name.

    This exists so you (the student) can find the correct scheme_code
    for any fund WITHOUT guessing or hardcoding wrong numbers. Run this
    interactively first, confirm the exact scheme (Direct/Growth plan
    is usually what you want), THEN put the code into fund_meta.csv.

    Example:
        >>> search_scheme("HDFC Top 100")
        [{"schemeCode": 125497, "schemeName": "HDFC Top 100 Fund - Direct Plan - Growth"}, ...]

    Time complexity: O(1) network call; result parsing is O(k) for k
    matches returned (k is small, usually < 50).
    """
    query = urllib.parse.quote(fund_name_query)
    url = f"{BASE_URL}/search?q={query}"

    with urllib.request.urlopen(url) as response:
        if response.status != 200:
            raise ConnectionError(f"Search failed for query: {fund_name_query}")
        results = json.loads(response.read().decode())

    return results  # list of {"schemeCode": int, "schemeName": str}


# ---------------------------------------------------------------------------
# 2. NAV history fetch
# ---------------------------------------------------------------------------
def fetch_nav_history(scheme_code, force_refresh=False):
    """
    Fetch full NAV (Net Asset Value) history for a mutual fund scheme.

    Parameters
    ----------
    scheme_code : int or str
        The AMFI scheme code (found via search_scheme()).
    force_refresh : bool
        If True, ignores local cache and re-fetches from the API.

    Returns
    -------
    list[dict] : [{"date": "DD-MM-YYYY", "nav": float}, ...]
                 sorted ASCENDING by date (oldest first).
                 We sort ascending because every later module (returns,
                 volatility, covariance) assumes chronological order.

    Time complexity
    ----------------
    O(T log T) where T = number of NAV records (dominated by the sort;
    T is at most a few thousand for any real fund, so this is fast).
    """
    _ensure_cache_dir()
    cache_file = _cache_path(scheme_code)

    if not force_refresh and os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)

    url = f"{BASE_URL}/{scheme_code}"
    with urllib.request.urlopen(url) as response:
        if response.status != 200:
            raise ConnectionError(f"Failed to fetch NAV data for scheme {scheme_code}")
        raw = json.loads(response.read().decode())

    records = raw.get("data", [])
    if not records:
        raise ValueError(f"No NAV data found for scheme code {scheme_code}")

    parsed = []
    for entry in records:
        parsed.append({
            "date": entry["date"],
            "nav": float(entry["nav"])
        })

    # mfapi.in returns newest-first by default; we need oldest-first
    parsed.sort(key=lambda r: datetime.strptime(r["date"], "%d-%m-%Y"))

    with open(cache_file, "w") as f:
        json.dump(parsed, f)

    return parsed


def fetch_nav_series_aligned(scheme_codes, start_date=None, end_date=None):
    """
    Fetch NAV histories for MULTIPLE funds and align them onto a common
    set of trading dates (inner join on date).

    Why this function needs to exist:
        Different mutual funds can have NAVs published on slightly
        different sets of dates (holidays, fund-specific gaps, funds
        launched at different times). If you naively zip two return
        series together without aligning dates first, Module 3's
        covariance calculation will silently produce WRONG numbers
        (misaligned days masquerading as correlated/uncorrelated days).
        This is a classic real-world data bug — handling it here means
        every later module can assume clean, aligned input.

    Parameters
    ----------
    scheme_codes : list[str/int]
    start_date, end_date : "YYYY-MM-DD" strings, optional filters

    Returns
    -------
    dict:
        {
          "dates": [d1, d2, ..., dT],              # common dates, ascending
          "nav_matrix": {scheme_code: [p1, ..., pT], ...}
        }

    Time complexity
    ----------------
    O(n * T) to fetch/parse n funds' histories, plus O(n * T) to compute
    the date intersection using set operations -> overall O(n * T).
    """
    all_histories = {}
    for code in scheme_codes:
        history = fetch_nav_history(code)
        all_histories[code] = {rec["date"]: rec["nav"] for rec in history}

    # Find the intersection of dates across ALL funds
    common_dates = set(all_histories[scheme_codes[0]].keys())
    for code in scheme_codes[1:]:
        common_dates &= set(all_histories[code].keys())

    common_dates = sorted(common_dates, key=lambda d: datetime.strptime(d, "%d-%m-%Y"))

    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        common_dates = [d for d in common_dates if datetime.strptime(d, "%d-%m-%Y") >= start_dt]
    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        common_dates = [d for d in common_dates if datetime.strptime(d, "%d-%m-%Y") <= end_dt]

    nav_matrix = {}
    for code in scheme_codes:
        nav_matrix[code] = [all_histories[code][d] for d in common_dates]

    return {"dates": common_dates, "nav_matrix": nav_matrix}


# ---------------------------------------------------------------------------
# 3. Static CSV loaders (fund metadata, holdings, sectors)
# ---------------------------------------------------------------------------
def load_fund_meta(path="fund_meta.csv"):
    """
    Loads fund universe metadata.
    Expected columns: fund_id, scheme_code, fund_name, category, risk_grade
    Returns: list[dict]
    """
    funds = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            funds.append(row)
    return funds


def load_holdings(path="holdings.csv"):
    """
    Loads manually curated stock-level holdings per fund.
    Expected columns: fund_id, stock_name, weight_pct
    Returns: dict -> {fund_id: {stock_name: weight_pct, ...}, ...}

    This feeds Module 4 (Overlap Penalty).
    """
    holdings = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fund_id = row["fund_id"]
            stock = row["stock_name"]
            weight = float(row["weight_pct"])
            holdings.setdefault(fund_id, {})[stock] = weight
    return holdings


def load_sectors(path="sectors.csv"):
    """
    Loads manually curated sector allocation per fund.
    Expected columns: fund_id, sector_name, weight_pct
    Returns: dict -> {fund_id: {sector_name: weight_pct, ...}, ...}

    This feeds Module 5 (Sector Concentration / HHI Penalty).
    """
    sectors = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fund_id = row["fund_id"]
            sector = row["sector_name"]
            weight = float(row["weight_pct"])
            sectors.setdefault(fund_id, {})[sector] = weight
    return sectors


# ---------------------------------------------------------------------------
# Simple manual test when running this file directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== PRISM Module 1: Data Layer self-test ===")

    print("\n[1] Searching for 'HDFC Top 100'...")
    try:
        results = search_scheme("HDFC Top 100")
        for r in results[:5]:
            print("   ", r["schemeCode"], "-", r["schemeName"])
    except Exception as e:
        print("   Search failed (check internet access):", e)

    print("\n[2] Fetching NAV history for scheme 125497 (HDFC Top 100 Direct Growth)...")
    try:
        history = fetch_nav_history(125497)
        print(f"   Got {len(history)} NAV records.")
        print("   First record:", history[0])
        print("   Last record :", history[-1])
    except Exception as e:
        print("   Fetch failed (check internet access):", e)

    print("\n[3] Loading fund_meta.csv / holdings.csv / sectors.csv templates...")
    try:
        meta = load_fund_meta("fund_meta.csv")
        holdings = load_holdings("holdings.csv")
        sectors = load_sectors("sectors.csv")
        print(f"   {len(meta)} funds in fund_meta.csv")
        print(f"   {len(holdings)} funds in holdings.csv")
        print(f"   {len(sectors)} funds in sectors.csv")
    except Exception as e:
        print("   CSV load failed:", e)

    print("\nModule 1 self-test complete.")
