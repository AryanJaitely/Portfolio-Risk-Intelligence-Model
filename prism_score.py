"""
PRISM - Module 8: PRISM Score (the core scoring formula)
=============================================================

This module COMBINES the outputs of Modules 3-7 into one score.
It deliberately contains almost NO new math of its own -- it reuses:
    - portfolio_return, portfolio_variance, diversification_ratio  (Module 3)
    - portfolio_overlap_penalty                                     (Module 4)
    - sector_concentration_penalty                                  (Module 5)
    - portfolio_consistency                                         (Module 6)
    - predicted return/volatility                                   (Module 7)

New in this module: the correlation penalty C(w), the risk-tolerance
coefficient mapping, and the standardization layer that makes all
these differently-scaled outputs safe to combine.

    S(w) = alpha*R(w) - beta*sigma(w) + gamma*D(w) - delta*O(w)
           - epsilon*C(w) - zeta*H(w) + eta*K(w)
"""

from covariance_engine import portfolio_return, portfolio_variance, diversification_ratio
from overlap_engine import portfolio_overlap_penalty
from sector_engine import sector_concentration_penalty
from consistency_engine import portfolio_consistency
import math


# ---------------------------------------------------------------------------
# Scale-normalization constants (Part 6 fix)
# ---------------------------------------------------------------------------
# THE BUG: R(w) and sigma(w) are raw DAILY return/volatility numbers, e.g.
# ~0.0003-0.001 in magnitude. D(w), O(w), C(w), H(w), K(w) are all already
# roughly bounded on a [0,1]-ish scale. Combining them directly means R and
# sigma are ~100-1000x SMALLER than every other term before any coefficient
# is even applied -- so no matter how alpha/beta are tuned, expected return
# can barely move the total score relative to diversification/consistency/
# sector terms. That mismatch, not the tau mechanism itself, is why the
# optimizer gravitated to ultra-low-risk debt/liquid funds: it was
# effectively optimizing D/O/C/H/K almost exclusively.
#
# THE FIX: rescale R and sigma onto the SAME rough [0,1] order of magnitude
# as the other components BEFORE combining, using fixed reference scales
# (a "typical" daily fund return / volatility for this fund universe).
# These are simple linear rescalings -- they don't change direction, sign,
# or the tau-driven risk-tolerance mechanism, just the units the additive
# formula operates in.
RETURN_REFERENCE_SCALE = 0.0003   # ~0.1%/day: a strong daily return for an
                                  # equity mutual fund in this universe
RISK_REFERENCE_SCALE = 0.01      # ~2%/day: a high (but not extreme) daily
                                  # volatility for this fund universe

# THE REWEIGHT: on top of the scale fix, apply an explicit importance
# weighting across the (now comparable-scale) components so the optimizer
# is nudged toward a balanced portfolio -- more credit for expected return,
# less domination by the risk-aversion/diversification/overlap/sector
# terms than the un-weighted formula implicitly gave them. Approximates the
# requested Return 30% / Risk 20% / Diversification 15% / Consistency 15%
# / Sector 10% / Overlap 10% split; Risk is nudged to 15% and a small 5%
# is carved out for the correlation term C(w) (not in the original 6-item
# list, but part of the existing formula) so the full set still sums to 1.
COMPONENT_IMPORTANCE = {
    "return": 0.40,
    "risk": 0.15,
    "correlation": 0.05,
    "diversification": 0.10,
    "overlap": 0.10,
    "sector": 0.10,
    "consistency": 0.10,
}



# ---------------------------------------------------------------------------
# 1. Standardization layer
# ---------------------------------------------------------------------------
def standardize_holdings_to_fractions(holdings_dict):
    """
    Convert holdings.csv-style percentages (0-100) to fractions (0-1)
    BEFORE they reach Module 4's overlap functions. Module 4's code
    works on either scale (see its docstring), but PRISM's score
    combines terms that must all live on a comparable [0,1]-ish scale,
    so this conversion happens explicitly here, at the integration
    boundary, rather than silently inside Module 4.

    Time complexity: O(n * H), n = funds, H = holdings per fund
    """
    return {
        fund_id: {stock: pct / 100.0 for stock, pct in stocks.items()}
        for fund_id, stocks in holdings_dict.items()
    }


# ---------------------------------------------------------------------------
# 2. Predicted covariance matrix (hybrid: historical correlation + predicted vol)
# ---------------------------------------------------------------------------
def build_predicted_covariance_matrix(corr_matrix, predicted_vols):
    """
    Cov_pred(i,j) = rho_hist(i,j) * sigma_pred_i * sigma_pred_j

    Standard volatility-rescaling technique: correlation structure is
    slow-moving and noisy to forecast from short data, so we keep
    Module 3's historical correlation matrix, but substitute Module 7's
    predicted (forward-looking) volatilities for the scale.

    Time complexity: O(n^2)
    """
    fund_ids = list(corr_matrix.keys())
    cov_pred = {f: {} for f in fund_ids}
    for i in fund_ids:
        for j in fund_ids:
            cov_pred[i][j] = corr_matrix[i][j] * predicted_vols[i] * predicted_vols[j]
    return cov_pred


# ---------------------------------------------------------------------------
# 3. Correlation penalty (new term, reuses Module 3's correlation matrix)
# ---------------------------------------------------------------------------
def portfolio_correlation_penalty(weights, corr_matrix):
    """
    C(w) = sum_{i != j} w_i * w_j * rho(i,j)

    Same weighted pairwise pattern used throughout PRISM (Module 3's
    portfolio_variance, Module 4's portfolio_overlap_penalty). Distinct
    from O(w): this measures STATISTICAL co-movement (correlation),
    while O(w) measures STRUCTURAL holdings redundancy.

    Note: this is intentionally allowed to go negative (if a portfolio
    is net negatively correlated) -- subtracting a negative penalty in
    the final score formula correctly REWARDS negative correlation,
    which is the mathematically correct behavior, not a bug.

    Time complexity: O(n^2)
    """
    total = 0.0
    for fund_i, w_i in weights.items():
        for fund_j, w_j in weights.items():
            if fund_i == fund_j:
                continue
            total += w_i * w_j * corr_matrix[fund_i][fund_j]
    return total


# ---------------------------------------------------------------------------
# 4. Diversification reward (reuses Module 3's diversification_ratio)
# ---------------------------------------------------------------------------
def diversification_reward(weights, predicted_vols, portfolio_vol):
    """
    D(w) = 1 - 1/DR(w), where DR is Module 3's diversification_ratio.

    DR >= 1 always for long-only weights with correlations <= 1 (portfolio
    vol can never exceed the weighted-average individual vol under these
    conditions), so D(w) is bounded in [0, 1). DR -> infinity (perfect
    cancellation) gives D -> 1.

    Time complexity: O(n) (dominated by diversification_ratio)
    """
    dr = diversification_ratio(weights, predicted_vols, portfolio_vol)
    if dr == math.inf:
        return 1.0
    return 1.0 - (1.0 / dr)


# ---------------------------------------------------------------------------
# 5. Risk-tolerance -> coefficient mapping
# ---------------------------------------------------------------------------
def risk_tolerance_weights(tau):
    """
    Linear interpolation between a CONSERVATIVE profile (tau=0) and an
    AGGRESSIVE profile (tau=1). Transparent and tunable by design --
    not a black box. The structural idea (return/consistency weight
    rise with tau; every penalty weight falls with tau) is the
    defensible part; exact numeric anchors are reasonable defaults.

    Parameters
    ----------
    tau : float in [0, 1] -- 0 = fully conservative, 1 = fully aggressive

    Time complexity: O(1)
    """
    if not (0.0 <= tau <= 1.0):
        raise ValueError("tau (risk tolerance) must be in [0, 1].")

    return {
        "alpha":   0.5 + 0.5 * tau,   # return reward:      0.5 -> 1.0
        "beta":    1.5 - 1.0 * tau,   # risk penalty:       1.5 -> 0.5
        "gamma":   1.0 - 0.5 * tau,   # diversification:    1.0 -> 0.5
        "delta":   1.0 - 0.5 * tau,   # overlap penalty:    1.0 -> 0.5
        "epsilon": 1.0 - 0.5 * tau,   # correlation penalty:1.0 -> 0.5
        "zeta":    1.0 - 0.5 * tau,   # sector penalty:     1.0 -> 0.5
        "eta":     0.8 - 0.3 * tau,   # consistency reward: 0.8 -> 0.5
    }


# ---------------------------------------------------------------------------
# 6. THE PRISM SCORE -- the orchestration function
# ---------------------------------------------------------------------------
def compute_prism_score(weights, predicted_returns, corr_matrix, predicted_vols,
                         overlap_matrix_fractions, sectors_dict, all_sector_names,
                         consistency_dict, tau):
    """
    Assemble the full PRISM score from all prior modules' outputs.

    Parameters
    ----------
    weights : dict {fund_id: portfolio_weight}, sums to 1
    predicted_returns : dict {fund_id: predicted_daily_return}   (Module 7)
    corr_matrix : dict of dict, historical correlation            (Module 3)
    predicted_vols : dict {fund_id: predicted_daily_volatility}  (Module 7)
    overlap_matrix_fractions : dict of dict, from Module 4, built on
                                FRACTION-scale holdings (see
                                standardize_holdings_to_fractions)
    sectors_dict, all_sector_names : inputs to Module 5
    consistency_dict : dict {fund_id: Module 6 result}
    tau : float in [0,1], investor risk tolerance

    Returns
    -------
    dict with "total_score" and a full breakdown of every term --
    always return the breakdown, never just the final number, so the
    score stays auditable (feeds Module 10 directly).

    Time complexity: O(n^2) -- dominated by the covariance/overlap/
    correlation matrix terms; everything else is O(n) or O(n*S).
    """
    coeffs = risk_tolerance_weights(tau)

    cov_pred = build_predicted_covariance_matrix(corr_matrix, predicted_vols)

    R = portfolio_return(weights, predicted_returns)
    sigma = portfolio_variance(weights, cov_pred) ** 0.5
    D = diversification_reward(weights, predicted_vols, sigma)
    O = portfolio_overlap_penalty(weights, overlap_matrix_fractions)
    C = portfolio_correlation_penalty(weights, corr_matrix)
    H = sector_concentration_penalty(weights, sectors_dict, all_sector_names)["penalty"]
    K = portfolio_consistency(weights, consistency_dict)

    # --- Part 6: normalize onto comparable scales, THEN weight ---------
    # R and sigma are raw daily magnitudes (~0.0003-0.001); D/O/C/H/K are
    # already ~[0,1]-ish. Rescale R and sigma so all seven terms are in
    # the same rough order of magnitude before the importance weights
    # and the existing tau-driven coefficients are applied.
    R_norm = R / RETURN_REFERENCE_SCALE
    sigma_norm = sigma / RISK_REFERENCE_SCALE
    imp = COMPONENT_IMPORTANCE
    
    total_score = (
        coeffs["alpha"] * imp["return"] * R_norm
        - coeffs["beta"] * imp["risk"] * sigma_norm
        + coeffs["gamma"] * imp["diversification"] * D
        - coeffs["delta"] * imp["overlap"] * O
        - coeffs["epsilon"] * imp["correlation"] * C
        - coeffs["zeta"] * imp["sector"] * H
        + coeffs["eta"] * imp["consistency"] * K
    )

    return {
        "total_score": total_score,
        "coefficients": coeffs,
        "component_importance": imp,
        "breakdown": {
            "R_return": R, "sigma_risk": sigma, "D_diversification": D,
            "O_overlap": O, "C_correlation": C, "H_sector": H, "K_consistency": K,
        },
    }


# ---------------------------------------------------------------------------
# Manual self-test using a small, fully hand-traceable synthetic universe
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== PRISM Module 8: PRISM Score self-test ===\n")

    weights = {"A": 0.5, "B": 0.3, "C": 0.2}

    predicted_returns = {"A": 0.0006, "B": 0.0004, "C": 0.0003}
    predicted_vols = {"A": 0.012, "B": 0.009, "C": 0.006}

    corr_matrix = {
        "A": {"A": 1.0, "B": 0.6, "C": -0.2},
        "B": {"A": 0.6, "B": 1.0, "C": 0.1},
        "C": {"A": -0.2, "B": 0.1, "C": 1.0},
    }

    holdings_pct = {
        "A": {"HDFC Bank": 9.0, "Infosys": 6.0},
        "B": {"HDFC Bank": 7.0, "ICICI Bank": 6.0},
        "C": {"ITC": 5.0, "L&T": 4.0},
    }
    holdings_fractions = standardize_holdings_to_fractions(holdings_pct)
    from overlap_engine import build_overlap_matrix
    overlap_matrix = build_overlap_matrix(holdings_fractions)

    all_sectors = ["Financials", "IT", "FMCG", "Industrials", "Unclassified/Other"]
    sectors_dict = {
        "A": {"Financials": 45.0, "IT": 20.0},
        "B": {"Financials": 50.0, "IT": 15.0},
        "C": {"FMCG": 30.0, "Industrials": 25.0},
    }

    consistency_dict = {
        "A": {"consistency": 0.8, "cv": 0.25, "num_windows": 6, "win_rate": 0.83},
        "B": {"consistency": 0.6, "cv": 0.67, "num_windows": 6, "win_rate": 0.67},
        "C": {"consistency": 0.9, "cv": 0.11, "num_windows": 6, "win_rate": 0.9},
    }

    for tau, label in [(0.0, "Conservative"), (0.5, "Balanced"), (1.0, "Aggressive")]:
        result = compute_prism_score(
            weights, predicted_returns, corr_matrix, predicted_vols,
            overlap_matrix, sectors_dict, all_sectors, consistency_dict, tau
        )
        print(f"[{label} investor, tau={tau}]")
        print(f"  Total PRISM Score : {result['total_score']:.6f}")
        print(f"  Breakdown         : {result['breakdown']}")
        print(f"  Coefficients      : {result['coefficients']}\n")
