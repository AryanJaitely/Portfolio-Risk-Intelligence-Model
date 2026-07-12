"""
PRISM - Module 5: Sector Concentration Penalty Engine
==========================================================

Responsibility of this file:
    Measure how concentrated a PORTFOLIO (not a single fund) is across
    market sectors, using the Herfindahl-Hirschman Index (HHI) -- a
    standard, well-established concentration metric borrowed from
    antitrust economics and widely reused in portfolio risk analysis.

What is standard (attributed, not invented):
    - HHI itself: HHI = sum(p_s^2) over sector exposures p_s
    - The normalization HHI* = (HHI - 1/N) / (1 - 1/N)

What PRISM adapts (the actual engineering work of this module):
    - Aggregating fund-level sector disclosures into a single
      PORTFOLIO-level sector exposure vector, weighted by portfolio
      weights (see portfolio_sector_exposure)
    - Correctly handling under-disclosed sector data via an explicit
      "Unclassified/Other" bucket, which is REQUIRED for HHI's
      mathematical guarantees (bounded in [1/N, 1]) to hold, since
      those guarantees assume exposures sum to exactly 1.

Design rule followed: explicit loops, no pandas/numpy groupby tricks.
Depends on: data_layer.py (Module 1) for loading sectors.csv.
"""

EPSILON = 1e-6
UNCLASSIFIED_LABEL = "Unclassified/Other"


# ---------------------------------------------------------------------------
# 1. Fix under-disclosed fund sector data (data-correctness step)
# ---------------------------------------------------------------------------
def normalize_fund_sector_weights(fund_sector_dict):
    """
    Ensure one fund's sector weights sum to exactly 100 by assigning
    any missing mass to an explicit "Unclassified/Other" bucket.

    Why this matters mathematically: HHI's bounds (Section 2.2 of the
    README/derivation) are only guaranteed to hold in [1/N, 1] when
    exposures sum to exactly 1. If we silently ignored undisclosed
    holdings, portfolios with poor sector disclosure would show
    ARTIFICIALLY LOW HHI (looking falsely diversified) simply because
    part of their allocation was invisible to the formula, not because
    it was actually spread across sectors.

    Parameters
    ----------
    fund_sector_dict : dict {sector_name: weight_pct}  (0-100 scale)

    Returns
    -------
    dict : a NEW dict (does not mutate the input), with an added
           "Unclassified/Other" entry if needed.

    Time complexity: O(S), S = number of disclosed sectors for the fund.
    """
    result = dict(fund_sector_dict)  # copy, never mutate caller's data
    total = sum(result.values())
    residual = 100.0 - total

    if residual > EPSILON:
        result[UNCLASSIFIED_LABEL] = result.get(UNCLASSIFIED_LABEL, 0.0) + residual
    elif residual < -EPSILON:
        # Data entry error: sectors should never sum to MORE than 100%.
        raise ValueError(
            f"Fund sector weights sum to {total:.2f}%, which exceeds 100%. "
            f"Check sectors.csv for a data entry error."
        )

    return result


# ---------------------------------------------------------------------------
# 2. Portfolio-level sector exposure (fund-level -> portfolio-level)
# ---------------------------------------------------------------------------
def portfolio_sector_exposure(weights, sectors_dict, all_sector_names):
    """
    Aggregate fund-level sector disclosures into a single portfolio-
    level sector exposure vector, weighted by portfolio weights.

        p_s(w) = sum_i ( w_i * sector_i(s) )

    Parameters
    ----------
    weights : dict {fund_id: portfolio_weight}, should sum to 1
    sectors_dict : dict {fund_id: {sector_name: weight_pct}} (0-100 scale)
    all_sector_names : list[str], the FULL sector taxonomy (union of all
                        sectors that appear across all funds, PLUS the
                        "Unclassified/Other" bucket). Passed in explicitly
                        (rather than inferred only from nonzero exposures)
                        so that N in the HHI normalization always reflects
                        the true taxonomy size, not just sectors that
                        happen to be present in THIS particular portfolio.

    Returns
    -------
    dict {sector_name: exposure_fraction}, summing to ~1.0

    Time complexity: O(n * S), n = funds, S = sectors per fund (small).
    """
    exposure = {sector: 0.0 for sector in all_sector_names}

    for fund_id, w in weights.items():
        fund_sectors = normalize_fund_sector_weights(sectors_dict[fund_id])
        for sector, pct in fund_sectors.items():
            if sector not in exposure:
                # Defensive: a sector appeared in this fund but wasn't in
                # the taxonomy list we were given -- add it so we never
                # silently drop real exposure.
                exposure[sector] = 0.0
            exposure[sector] += w * (pct / 100.0)

    return exposure


# ---------------------------------------------------------------------------
# 3. Herfindahl-Hirschman Index (standard formula, unmodified)
# ---------------------------------------------------------------------------
def compute_hhi(exposure_dict):
    """
    HHI = sum(p_s^2) over all sector exposure fractions p_s.

    This is the STANDARD, unmodified HHI formula from antitrust
    economics, applied here to portfolio sector exposure fractions.

    Time complexity: O(N), N = number of sectors in the taxonomy.
    """
    return sum(p ** 2 for p in exposure_dict.values())


def normalized_hhi(hhi, num_sectors):
    """
    Rescale raw HHI (bounded in [1/N, 1]) onto a universal [0, 1] scale:

        HHI* = (HHI - 1/N) / (1 - 1/N)

    HHI* = 0  -> perfectly even exposure across all N sectors
    HHI* = 1  -> fully concentrated in a single sector

    This standard normalization (sometimes called the "normalized HHI")
    is what makes concentration comparable across portfolios/taxonomies
    that use different numbers of sectors -- necessary before this
    value can be combined with PRISM's other [0,1]-bounded penalty
    terms in Module 8.

    Time complexity: O(1)
    """
    if num_sectors <= 1:
        return 0.0  # degenerate case: only one possible sector, no such thing as concentration
    min_hhi = 1.0 / num_sectors
    return (hhi - min_hhi) / (1.0 - min_hhi)


# ---------------------------------------------------------------------------
# 4. Full sector concentration penalty (the function Module 8 will call)
# ---------------------------------------------------------------------------
def sector_concentration_penalty(weights, sectors_dict, all_sector_names):
    """
    Compute PRISM's sector concentration penalty H(w) for a portfolio.

    Returns
    -------
    dict with:
        "penalty"   : float in [0, 1], the normalized HHI (H(w) in
                      Module 8's PRISM Score equation)
        "raw_hhi"   : float in [1/N, 1], the unnormalized HHI (kept for
                      transparency/debugging -- always show your
                      intermediate numbers, not just the final score)
        "exposure"  : dict {sector: fraction}, the portfolio's actual
                      sector breakdown (feeds Module 10's explanations)

    Time complexity: O(n*S + N)
    """
    exposure = portfolio_sector_exposure(weights, sectors_dict, all_sector_names)
    raw_hhi = compute_hhi(exposure)
    penalty = normalized_hhi(raw_hhi, len(all_sector_names))

    return {
        "penalty": penalty,
        "raw_hhi": raw_hhi,
        "exposure": exposure,
    }


# ---------------------------------------------------------------------------
# 5. Rule-based explanation (early groundwork for Module 10)
# ---------------------------------------------------------------------------
def explain_sector_concentration(result, top_n=3):
    """
    Human-readable explanation of a sector concentration result,
    using the standard antitrust HHI interpretation bands (on the
    raw, 0-1 scale HHI, not the normalized version -- these bands are
    an established convention, not a PRISM invention):
        < 0.15            -> low concentration
        0.15 to 0.25      -> moderate concentration
        >= 0.25           -> high concentration

    Time complexity: O(N log N) for sorting sectors by exposure
    (N is small, so this is not a practical concern).
    """
    raw_hhi = result["raw_hhi"]
    exposure = result["exposure"]

    if raw_hhi < 0.15:
        band = "LOW concentration (well diversified across sectors)"
    elif raw_hhi < 0.25:
        band = "MODERATE concentration"
    else:
        band = "HIGH concentration (sector risk is significant)"

    top_sectors = sorted(exposure.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_desc = ", ".join(f"{s} ({p*100:.1f}%)" for s, p in top_sectors)

    return (
        f"Portfolio sector HHI = {raw_hhi:.4f} -> {band}. "
        f"Top sector exposures: {top_desc}."
    )


# ---------------------------------------------------------------------------
# Manual self-test using hand-verifiable synthetic examples
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== PRISM Module 5: Sector Concentration Penalty self-test ===\n")

    all_sectors = ["Financials", "IT", "Energy", "Healthcare", UNCLASSIFIED_LABEL]

    # Scenario A: two funds, BOTH heavily overweight Financials.
    # (Zero stock overlap possible -- Module 4 wouldn't catch this at all.)
    sectors_a = {
        "F1": {"Financials": 45.0, "IT": 20.0, "Energy": 15.0, "Healthcare": 15.0},
        "F2": {"Financials": 50.0, "IT": 15.0, "Energy": 20.0, "Healthcare": 10.0},
    }
    weights_a = {"F1": 0.5, "F2": 0.5}
    result_a = sector_concentration_penalty(weights_a, sectors_a, all_sectors)

    print("[Scenario A: both funds overweight Financials]")
    print(f"  Portfolio exposure: {result_a['exposure']}")
    print(f"  Raw HHI            : {result_a['raw_hhi']:.4f}")
    print(f"  Normalized HHI*    : {result_a['penalty']:.4f}")
    print(f"  {explain_sector_concentration(result_a)}")

    # Manual check: expected Financials exposure = 0.5*0.45 + 0.5*0.50 = 0.475
    expected_financials = 0.5 * 0.45 + 0.5 * 0.50
    print(f"  [Manual check] Expected Financials exposure = {expected_financials:.4f}, "
          f"got {result_a['exposure']['Financials']:.4f}")

    # Scenario B: same two funds, but a THIRD fund with a totally different
    # sector mix, diversifying the portfolio.
    print("\n[Scenario B: adding a diversifying 3rd fund]")
    sectors_b = dict(sectors_a)
    sectors_b["F3"] = {"IT": 10.0, "Energy": 10.0, "Healthcare": 60.0, "Financials": 20.0}
    weights_b = {"F1": 0.35, "F2": 0.35, "F3": 0.30}
    result_b = sector_concentration_penalty(weights_b, sectors_b, all_sectors)

    print(f"  Portfolio exposure: {result_b['exposure']}")
    print(f"  Raw HHI            : {result_b['raw_hhi']:.4f}")
    print(f"  Normalized HHI*    : {result_b['penalty']:.4f}")
    print(f"  {explain_sector_concentration(result_b)}")

    print(f"\n[Key result] Adding a diversifying fund should REDUCE concentration:")
    print(f"  Scenario A normalized HHI* = {result_a['penalty']:.4f}")
    print(f"  Scenario B normalized HHI* = {result_b['penalty']:.4f}")
    assert result_b["penalty"] < result_a["penalty"], "Diversifying fund should lower concentration"
    print("  Confirmed: concentration penalty decreased as expected.")

    # Edge case: fund with under-disclosed sectors (only sums to 70%)
    print("\n[Edge case: under-disclosed fund sectors]")
    under_disclosed = {"Financials": 40.0, "IT": 30.0}  # sums to 70, not 100
    fixed = normalize_fund_sector_weights(under_disclosed)
    print(f"  Original: {under_disclosed} (sums to {sum(under_disclosed.values())})")
    print(f"  Fixed   : {fixed} (sums to {sum(fixed.values())})")
