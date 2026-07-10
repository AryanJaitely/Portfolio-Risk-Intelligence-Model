"""
PRISM - Module 3: Covariance & Correlation Engine
=====================================================

Responsibility of this file:
    Given daily return series for multiple funds, compute:
        1. The pairwise covariance matrix (Sigma)
        2. The pairwise correlation matrix (rho)
        3. Portfolio-level variance/volatility for a given weight vector
        4. Portfolio-level expected return for a given weight vector

Design rule followed:
    No numpy.cov(), no numpy.corrcoef(). Every entry of every matrix is
    computed with an explicit, readable loop. Matrices are represented
    as dict-of-dicts (matrix[fund_id_a][fund_id_b] -> float) rather than
    raw 2D arrays, because fund_ids are meaningful strings (e.g. "F01")
    and dict-of-dicts avoids silent indexing bugs -- readability over
    raw performance, per PRISM's engineering philosophy.

Depends on: return_engine.py (Module 2) for mean/volatility computation.
"""

import math


# ---------------------------------------------------------------------------
# 1. Pairwise covariance
# ---------------------------------------------------------------------------
def compute_covariance(returns_i, returns_j, mean_i, mean_j):
    """
    Sample covariance between two equal-length, DATE-ALIGNED return series.

    Cov(i,j) = (1 / (T-1)) * sum( (r_i,t - mu_i) * (r_j,t - mu_j) )

    CRITICAL PRECONDITION: returns_i[t] and returns_j[t] must correspond
    to the SAME calendar day t. If your NAV series weren't date-aligned
    before computing returns (see Module 1's fetch_nav_series_aligned),
    this number will be silently wrong -- there will be no error, just
    a meaningless covariance. This is the single most common bug in
    student portfolio projects.

    Time complexity: O(T)
    """
    if len(returns_i) != len(returns_j):
        raise ValueError(
            "Return series must be the same length and date-aligned. "
            f"Got lengths {len(returns_i)} and {len(returns_j)}."
        )

    T = len(returns_i)
    if T < 2:
        raise ValueError("Need at least 2 aligned observations to compute covariance.")

    sum_product = 0.0
    for t in range(T):
        sum_product += (returns_i[t] - mean_i) * (returns_j[t] - mean_j)

    return sum_product / (T - 1)


# ---------------------------------------------------------------------------
# 2. Full covariance matrix across the fund universe
# ---------------------------------------------------------------------------
def build_covariance_matrix(returns_dict, means_dict):
    """
    Build the full n x n covariance matrix across all funds.

    Parameters
    ----------
    returns_dict : dict {fund_id: [daily returns...]}  (all must be
                   the same length and date-aligned -- see Module 1)
    means_dict   : dict {fund_id: mean_daily_return}

    Returns
    -------
    dict of dict: cov_matrix[fund_a][fund_b] -> covariance

    Time complexity: O(n^2 * T)
        n(n+1)/2 unique pairs (we exploit Cov(i,j) = Cov(j,i) and only
        compute the upper triangle), each pair costs O(T).
    """
    fund_ids = list(returns_dict.keys())
    cov_matrix = {fund_id: {} for fund_id in fund_ids}

    for a in range(len(fund_ids)):
        for b in range(a, len(fund_ids)):  # upper triangle only, a <= b
            fund_i = fund_ids[a]
            fund_j = fund_ids[b]

            cov = compute_covariance(
                returns_dict[fund_i], returns_dict[fund_j],
                means_dict[fund_i], means_dict[fund_j]
            )

            cov_matrix[fund_i][fund_j] = cov
            cov_matrix[fund_j][fund_i] = cov  # symmetry: fill both sides

    return cov_matrix


# ---------------------------------------------------------------------------
# 3. Correlation matrix (normalized covariance)
# ---------------------------------------------------------------------------
def build_correlation_matrix(cov_matrix):
    """
    Convert a covariance matrix into a correlation matrix.

    rho(i,j) = Cov(i,j) / (sigma_i * sigma_j)

    Note: sigma_i = sqrt(Cov(i,i)) -- the diagonal of the covariance
    matrix already IS each fund's variance, so we don't need to
    recompute volatility separately here. This is a nice example of
    how a well-designed data structure (the matrix) avoids redundant
    computation elsewhere.

    Returns
    -------
    dict of dict: corr_matrix[fund_a][fund_b] -> correlation in [-1, 1]

    Time complexity: O(n^2)
    """
    fund_ids = list(cov_matrix.keys())

    # Extract each fund's volatility from the diagonal of the cov matrix
    vol = {f: math.sqrt(cov_matrix[f][f]) for f in fund_ids}

    corr_matrix = {f: {} for f in fund_ids}
    for fund_i in fund_ids:
        for fund_j in fund_ids:
            denom = vol[fund_i] * vol[fund_j]
            if denom == 0:
                # A fund with zero volatility has undefined correlation
                # with anything (division by zero). We define it as 0
                # by convention -- a zero-variance fund can't "move
                # together" with anything, so treating it as uncorrelated
                # is the sensible, safe default.
                corr_matrix[fund_i][fund_j] = 0.0
            else:
                corr_matrix[fund_i][fund_j] = cov_matrix[fund_i][fund_j] / denom

    return corr_matrix


# ---------------------------------------------------------------------------
# 4. Portfolio-level formulas
# ---------------------------------------------------------------------------
def portfolio_return(weights, mean_returns):
    """
    Expected portfolio return: R_p = sum(w_i * mu_i)

    Parameters
    ----------
    weights : dict {fund_id: weight}   (weights should sum to 1)
    mean_returns : dict {fund_id: mean_daily_return}

    Time complexity: O(n)
    """
    total = 0.0
    for fund_id, w in weights.items():
        total += w * mean_returns[fund_id]
    return total


def portfolio_variance(weights, cov_matrix):
    """
    Portfolio variance: sigma_p^2 = w^T * Sigma * w
                                  = sum_i sum_j w_i * w_j * Cov(i,j)

    This is THE formula that mathematically explains why individually
    good funds can make a bad portfolio (see README section 1.4 for
    the full derivation and the diagonal/off-diagonal decomposition).

    Time complexity: O(n^2)
    """
    total = 0.0
    for fund_i, w_i in weights.items():
        for fund_j, w_j in weights.items():
            total += w_i * w_j * cov_matrix[fund_i][fund_j]
    return total


def portfolio_volatility(weights, cov_matrix):
    """
    Portfolio volatility = sqrt(portfolio variance)

    Time complexity: O(n^2) (dominated by portfolio_variance)
    """
    var = portfolio_variance(weights, cov_matrix)
    if var < 0:
        # Can only happen from floating point error on a
        # non-positive-semi-definite matrix (e.g. bad/misaligned data).
        # Clamp instead of crashing, but this is a signal to check data.
        var = 0.0
    return math.sqrt(var)


def diversification_ratio(weights, vol_dict, portfolio_vol):
    """
    A classical diversification measure (Choueifaty & Coignard, 2008),
    used here as a MACHINERY / diagnostic value, not PRISM's own
    diversification reward formula (that original formula comes in
    Module 4 -- this is a well-known benchmark ratio, and we're being
    upfront that it's not something we invented):

        DR = (weighted average of individual volatilities) / portfolio volatility

    DR = 1 means no diversification benefit at all (as if there's only
    one fund, or all funds are perfectly correlated). DR > 1 means the
    portfolio is less risky than the weighted-average individual risk
    would suggest -- diversification is doing real work.

    Time complexity: O(n)
    """
    weighted_avg_vol = sum(w * vol_dict[f] for f, w in weights.items())

    if portfolio_vol == 0:
        if weighted_avg_vol == 0:
            # Genuinely no risk anywhere -- ratio is undefined, default to 1
            return 1.0
        # Individual funds carry real risk, but it fully cancelled out
        # in the portfolio (e.g. perfect anti-correlation). The ratio
        # is mathematically unbounded in this case -- report infinity
        # rather than the misleading value 1.0.
        return math.inf

    return weighted_avg_vol / portfolio_vol


# ---------------------------------------------------------------------------
# 5. Pretty-printing helper (for debugging / report screenshots)
# ---------------------------------------------------------------------------
def print_matrix(matrix, title="Matrix", decimals=4):
    """Simple readable console print of a dict-of-dicts matrix."""
    fund_ids = list(matrix.keys())
    print(f"\n{title}")
    header = "        " + "".join(f"{f:>10}" for f in fund_ids)
    print(header)
    for fund_i in fund_ids:
        row = f"{fund_i:>8}" + "".join(
            f"{matrix[fund_i][fund_j]:>10.{decimals}f}" for fund_j in fund_ids
        )
        print(row)


# ---------------------------------------------------------------------------
# Manual self-test using a hand-verifiable synthetic example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== PRISM Module 3: Covariance & Correlation Engine self-test ===")

    # Three synthetic funds with a KNOWN relationship, so we can hand-verify:
    #   Fund A and Fund B move in perfect lockstep      -> corr should be +1
    #   Fund A and Fund C move perfectly opposite        -> corr should be -1
    returns = {
        "A": [0.01, 0.02, -0.01, 0.03, -0.02],
        "B": [0.02, 0.04, -0.02, 0.06, -0.04],   # = 2x Fund A, perfectly correlated
        "C": [-0.01, -0.02, 0.01, -0.03, 0.02],  # = -1x Fund A, perfectly anti-correlated
    }

    means = {f: sum(r) / len(r) for f, r in returns.items()}

    cov_matrix = build_covariance_matrix(returns, means)
    corr_matrix = build_correlation_matrix(cov_matrix)

    print_matrix(cov_matrix, "Covariance Matrix")
    print_matrix(corr_matrix, "Correlation Matrix")

    print("\n[Expected] corr(A,B) = +1.0 (B is a scaled copy of A)")
    print(f"[Actual]   corr(A,B) = {corr_matrix['A']['B']:.6f}")

    print("\n[Expected] corr(A,C) = -1.0 (C is an inverted copy of A)")
    print(f"[Actual]   corr(A,C) = {corr_matrix['A']['C']:.6f}")

    # Portfolio test: equal-weighted A and C (perfectly anti-correlated)
    # should have MUCH lower risk than either fund alone -- this is the
    # clearest possible demonstration of the diversification formula.
    weights = {"A": 0.5, "B": 0.0, "C": 0.5}
    p_var = portfolio_variance(weights, cov_matrix)
    p_vol = portfolio_volatility(weights, cov_matrix)

    vol_dict = {f: math.sqrt(cov_matrix[f][f]) for f in returns}
    dr = diversification_ratio(weights, vol_dict, p_vol)

    print(f"\n[Portfolio: 50% A + 50% C, perfectly anti-correlated]")
    print(f"  Fund A volatility alone : {vol_dict['A']:.6f}")
    print(f"  Fund C volatility alone : {vol_dict['C']:.6f}")
    print(f"  Portfolio volatility    : {p_vol:.6f}  <-- should be near ZERO")
    print(f"  Diversification Ratio   : {dr}  <-- mathematically unbounded (inf)")
    print("\nThis is the mathematical proof of diversification: combining")
    print("two individually risky, perfectly anti-correlated funds cancels")
    print("out portfolio risk entirely, making the diversification ratio")
    print("mathematically unbounded (division by a volatility of ~0).")

    # A second, more REALISTIC example: partial (not perfect) correlation,
    # which is what you'll actually see with real mutual fund data.
    print("\n\n=== A more realistic example: partially correlated funds ===")
    returns2 = {
        "X": [0.012, -0.008, 0.020, -0.015, 0.005, 0.018, -0.010],
        "Y": [0.008, -0.003, 0.015, -0.020, 0.010, 0.010, -0.005],
        "Z": [-0.005, 0.010, -0.008, 0.012, -0.010, -0.006, 0.015],
    }
    means2 = {f: sum(r) / len(r) for f, r in returns2.items()}
    cov2 = build_covariance_matrix(returns2, means2)
    corr2 = build_correlation_matrix(cov2)
    print_matrix(corr2, "Correlation Matrix (realistic example)")

    vol2 = {f: math.sqrt(cov2[f][f]) for f in returns2}
    equal_weights = {"X": 1 / 3, "Y": 1 / 3, "Z": 1 / 3}
    p_var2 = portfolio_variance(equal_weights, cov2)
    p_vol2 = portfolio_volatility(equal_weights, cov2)
    dr2 = diversification_ratio(equal_weights, vol2, p_vol2)

    print(f"\nIndividual volatilities : X={vol2['X']:.4f}  Y={vol2['Y']:.4f}  Z={vol2['Z']:.4f}")
    print(f"Equal-weight portfolio volatility : {p_vol2:.4f}")
    print(f"Diversification Ratio             : {dr2:.4f}")
    print("(A finite DR > 1 here shows a realistic, moderate diversification")
    print("benefit -- this is the typical case Module 4 onward will build on,")
    print("not the degenerate perfect-cancellation case above.)")
