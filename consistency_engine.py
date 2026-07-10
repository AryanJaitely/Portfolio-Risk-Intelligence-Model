"""
PRISM - Module 6: Consistency Score Engine
==============================================

Responsibility of this file:
    Measure whether a fund's good performance is a STABLE, repeatable
    pattern over time, or a one-off lucky stretch that a single
    whole-history mu/sigma (Module 2) cannot distinguish.

Standard technique used (attributed, not invented):
    - Rolling-window Sharpe-like ratios (mean/vol per sub-period)
    - Coefficient of Variation (CV = std/mean) to measure relative
      (not absolute) stability across those windows

PRISM's adaptation:
    - Reciprocal saturation transform 1/(1+CV) to map the unbounded
      CV onto a bounded (0,1] reward score, consistent with every
      other PRISM component being bounded before Module 8 combines them
    - Simple linear (not pairwise) aggregation to portfolio level,
      since consistency is a per-fund attribute, not an interaction

Design rule followed: explicit loops, no pandas rolling() shortcuts.
Depends on: return_engine.py (Module 2) is NOT required directly here,
but this module expects the same kind of daily log-return list that
Module 2 produces.
"""

import math


# ---------------------------------------------------------------------------
# 1. Rolling windows
# ---------------------------------------------------------------------------
def rolling_windows(returns, window_length):
    """
    Split a return series into K non-overlapping windows of a fixed
    length. Any leftover returns that don't fill a full final window
    are discarded (kept simple and explicit, rather than padding).

    Parameters
    ----------
    returns : list[float]
    window_length : int, number of trading days per window (e.g. 63
                     for an approximate calendar quarter)

    Returns
    -------
    list[list[float]] : K windows, each of length window_length

    Time complexity: O(T)
    """
    if window_length < 2:
        raise ValueError("window_length must be at least 2 (need variance within a window).")

    num_windows = len(returns) // window_length
    if num_windows < 2:
        raise ValueError(
            f"Not enough data for consistency analysis: got {len(returns)} returns "
            f"and window_length={window_length}, which gives only {num_windows} "
            f"full window(s). Need at least 2. Provide more history or use a "
            f"smaller window_length."
        )

    windows = []
    for k in range(num_windows):
        start = k * window_length
        end = start + window_length
        windows.append(returns[start:end])

    return windows


# ---------------------------------------------------------------------------
# 2. Per-window Sharpe-like ratio
# ---------------------------------------------------------------------------
def window_sharpe(window_returns):
    """
    Mean-to-volatility ratio WITHIN one window. Risk-free rate is
    assumed ~0 here (a stated simplification -- see README) since it's
    small relative to typical equity fund daily volatility over a
    quarter-length window.

    Returns None if the window has zero volatility (undefined ratio;
    can happen with degenerate/synthetic data, essentially never with
    real NAV data) -- callers should filter out None results rather
    than treating 0 as a real Sharpe value.

    Time complexity: O(L), L = window_length
    """
    L = len(window_returns)
    if L < 2:
        raise ValueError("Window must have at least 2 returns to compute volatility.")

    mean_k = sum(window_returns) / L

    sum_sq_dev = 0.0
    for r in window_returns:
        sum_sq_dev += (r - mean_k) ** 2
    vol_k = math.sqrt(sum_sq_dev / (L - 1))

    if vol_k == 0:
        return None

    return mean_k / vol_k


# ---------------------------------------------------------------------------
# 3. Fund-level consistency score
# ---------------------------------------------------------------------------
def fund_consistency_score(returns, window_length=63):
    """
    Compute PRISM's consistency score for one fund.

    Returns
    -------
    dict with:
        "consistency" : float in (0, 1], the reward term used in
                         Module 8 (K_i in the PRISM score equation)
        "cv"           : float, the raw coefficient of variation
                         (kept for transparency/debugging)
        "num_windows"  : int, how many valid (nonzero-volatility)
                         windows were used
        "win_rate"     : float in [0,1], fraction of windows with
                         positive average return (diagnostic only,
                         feeds Module 10's explanations, not the score)

    Time complexity: O(T)  (T = len(returns); building windows is O(T),
    computing each window's Sharpe is O(L), and there are T/L windows,
    so total window-Sharpe cost is O(T) too)
    """
    windows = rolling_windows(returns, window_length)

    sharpes = []
    positive_window_count = 0
    for w in windows:
        s = window_sharpe(w)
        if s is not None:
            sharpes.append(s)
        window_mean = sum(w) / len(w)
        if window_mean > 0:
            positive_window_count += 1

    if len(sharpes) < 2:
        raise ValueError(
            "Fewer than 2 windows had defined (nonzero-volatility) Sharpe "
            "ratios -- cannot assess consistency. This should not happen "
            "with real NAV data; check your input."
        )

    K = len(sharpes)
    mu_S = sum(sharpes) / K

    sum_sq_dev = 0.0
    for s in sharpes:
        sum_sq_dev += (s - mu_S) ** 2
    sigma_S = math.sqrt(sum_sq_dev / (K - 1))

    if mu_S == 0:
        cv = math.inf
    else:
        cv = sigma_S / abs(mu_S)

    consistency = 1.0 / (1.0 + cv) if cv != math.inf else 0.0

    win_rate = positive_window_count / len(windows)

    return {
        "consistency": consistency,
        "cv": cv,
        "num_windows": K,
        "win_rate": win_rate,
    }


# ---------------------------------------------------------------------------
# 4. Portfolio-level consistency (simple weighted average, no interaction terms)
# ---------------------------------------------------------------------------
def portfolio_consistency(weights, consistency_dict):
    """
    K(w) = sum_i ( w_i * K_i )

    Unlike covariance/overlap/sector concentration, consistency is a
    per-fund attribute (not a pairwise interaction), so this is a plain
    weighted average -- same aggregation pattern as portfolio_return()
    in Module 3.

    Parameters
    ----------
    weights : dict {fund_id: weight}
    consistency_dict : dict {fund_id: result from fund_consistency_score()}

    Time complexity: O(n)
    """
    total = 0.0
    for fund_id, w in weights.items():
        total += w * consistency_dict[fund_id]["consistency"]
    return total


# ---------------------------------------------------------------------------
# 5. Rule-based explanation (same pattern as Modules 4 and 5)
# ---------------------------------------------------------------------------
def explain_consistency(fund_id, result):
    """
    Human-readable explanation of a fund's consistency result.

    Time complexity: O(1)
    """
    consistency = result["consistency"]
    win_rate = result["win_rate"]
    num_windows = result["num_windows"]

    if consistency >= 0.7:
        band = "HIGHLY consistent"
    elif consistency >= 0.4:
        band = "MODERATELY consistent"
    else:
        band = "INCONSISTENT (performance varies a lot across periods)"

    return (
        f"{fund_id}: consistency score = {consistency:.3f} -> {band}. "
        f"Positive risk-adjusted return in {win_rate*100:.0f}% of "
        f"{num_windows} analyzed periods."
    )


# ---------------------------------------------------------------------------
# Manual self-test using hand-verifiable synthetic examples
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== PRISM Module 6: Consistency Score self-test ===\n")

    # Fund X: steady performance -- same mean/vol pattern repeated in
    # every window (constructed to have IDENTICAL Sharpe every window).
    steady_window = [0.01, -0.005, 0.008, -0.003, 0.006, -0.002] * 2  # length 12
    fund_x_returns = steady_window * 5  # 5 repeats of an identical pattern -> CV should be ~0

    # Fund Y: same OVERALL average return and volatility as Fund X (over
    # the full history), but performance is front-loaded: great in the
    # first half, bad in the second half -- this should score LOWER on
    # consistency despite matching Fund X on whole-history mu/sigma.
    good_half = [0.02, 0.015, 0.018, 0.012, 0.02, 0.01] * 5   # strong, steady gains
    bad_half = [-0.025, -0.01, -0.022, -0.008, -0.018, -0.015] * 5  # losses (deliberately NOT an exact mirror, so window Sharpes don't coincidentally cancel to exactly zero -- see test_consistency_engine.py for a dedicated test of that exact edge case)
    fund_y_returns = good_half + bad_half

    result_x = fund_consistency_score(fund_x_returns, window_length=12)
    result_y = fund_consistency_score(fund_y_returns, window_length=12)

    print("[Fund X: identical performance pattern repeated every window]")
    print(f"  {explain_consistency('FundX', result_x)}")
    print(f"  CV = {result_x['cv']:.6f}")

    print("\n[Fund Y: great first half, equally bad second half]")
    print(f"  {explain_consistency('FundY', result_y)}")
    print(f"  CV = {result_y['cv']:.6f}")

    print(f"\n[Key result] Fund X consistency ({result_x['consistency']:.4f}) should be "
          f"MUCH higher than Fund Y consistency ({result_y['consistency']:.4f}), "
          f"even though both could show similar whole-history averages "
          f"if pooled naively.")
    assert result_x["consistency"] > result_y["consistency"]
    print("  Confirmed.")

    # Portfolio-level test
    print("\n[Portfolio-level consistency test]")
    consistency_dict = {"X": result_x, "Y": result_y}
    weights = {"X": 0.7, "Y": 0.3}
    port_consistency = portfolio_consistency(weights, consistency_dict)
    expected = 0.7 * result_x["consistency"] + 0.3 * result_y["consistency"]
    print(f"  Portfolio consistency K(w) = {port_consistency:.4f}")
    print(f"  Manual check (weighted avg) = {expected:.4f}")
    assert abs(port_consistency - expected) < 1e-9
