import csv
import glob
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

DATE_FORMATS = ["%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"]


def parse_date(date_str):
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date format: {date_str!r}")


def normalize_date(date_str):
    dt = parse_date(date_str)
    return dt.strftime("%Y-%m-%d")


def _process_file(filepath, fund_codes_set):
    result = {}

    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        idx_code = header.index("Scheme Code")
        idx_nav = header.index("Net Asset Value")
        idx_date = header.index("Date")

        for row in reader:
            code = row[idx_code].strip()

            if code not in fund_codes_set:
                continue

            nav_str = row[idx_nav].strip()

            try:
                nav_value = float(nav_str)
            except ValueError:
                continue

            # Skip invalid NAV values
            if nav_value <= 0:
                continue

            try:
                date_str = normalize_date(row[idx_date].strip())
            except ValueError:
                continue

            result.setdefault(code, {})[date_str] = nav_value

    return result


def load_hk7797_nav_folder(
    folder_path,
    fund_codes,
    max_workers=8,
    trim_to_common_start=True,
    drop_non_overlapping=False,
):
    """
    trim_to_common_start:
        Trim to the period where all funds have actual NAV data.

    drop_non_overlapping:
        Automatically remove funds that do not overlap with the others.
    """

    fund_codes_set = set(fund_codes)
    raw = {code: {} for code in fund_codes}

    files = glob.glob(f"{folder_path}/data_*.csv")

    if not files:
        raise ValueError(f"No files matching 'data_*.csv' found in {folder_path}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_process_file, fp, fund_codes_set)
            for fp in files
        ]

        for future in as_completed(futures):
            file_result = future.result()

            for code, date_navs in file_result.items():
                raw[code].update(date_navs)

    codes_with_data = [c for c in fund_codes if raw[c]]
    codes_without_data = [c for c in fund_codes if not raw[c]]

    if codes_without_data:
        print("WARNING: No data found for:", codes_without_data)

    if not codes_with_data:
        raise ValueError("No matching fund codes were found.")

    fund_ranges = {}

    print("\n--- Per-fund date coverage ---")

    for code in codes_with_data:
        dates_sorted = sorted(raw[code].keys())
        fund_ranges[code] = (dates_sorted[0], dates_sorted[-1])

        print(
            f"{code}: {len(dates_sorted)} dates | "
            f"first={dates_sorted[0]} last={dates_sorted[-1]}"
        )

    print("-------------------------------\n")

    latest_start = max(r[0] for r in fund_ranges.values())
    earliest_end = min(r[1] for r in fund_ranges.values())

    if latest_start > earliest_end:

        offenders = [
            c
            for c, (first, last) in fund_ranges.items()
            if last < latest_start
        ]

        msg = (
            "No overlapping date range exists.\n"
            f"Offending funds: {offenders}"
        )

        if not drop_non_overlapping:
            raise ValueError(
                msg + "\nUse drop_non_overlapping=True to remove them."
            )

        print(msg)
        print("Auto-dropping:", offenders)

        codes_with_data = [
            c for c in codes_with_data
            if c not in offenders
        ]

        fund_ranges = {
            c: r
            for c, r in fund_ranges.items()
            if c in codes_with_data
        }

        latest_start = max(r[0] for r in fund_ranges.values())
        earliest_end = min(r[1] for r in fund_ranges.values())

    all_dates = sorted(
        set().union(*(raw[c].keys() for c in codes_with_data))
    )

    nav_matrix = {}

    for code in codes_with_data:

        navs = []
        last_known = None

        for d in all_dates:

            if d in raw[code]:
                last_known = raw[code][d]

            navs.append(last_known)

        nav_matrix[code] = navs

    if not trim_to_common_start:
        return {
            "dates": all_dates,
            "nav_matrix": nav_matrix,
        }

    start_idx = (
        all_dates.index(latest_start)
        if latest_start in all_dates
        else 0
    )

    end_idx = (
        all_dates.index(earliest_end)
        if earliest_end in all_dates
        else len(all_dates) - 1
    )

    trimmed_dates = all_dates[start_idx : end_idx + 1]

    trimmed_nav_matrix = {
        code: navs[start_idx : end_idx + 1]
        for code, navs in nav_matrix.items()
    }

    return {
        "dates": trimmed_dates,
        "nav_matrix": trimmed_nav_matrix,
    }