from prism_score import (
    standardize_holdings_to_fractions, build_predicted_covariance_matrix,
    portfolio_correlation_penalty, diversification_reward,
    risk_tolerance_weights, compute_prism_score,
)
from overlap_engine import build_overlap_matrix


PREDICTED_RETURNS = {"A": 0.0006, "B": 0.0004, "C": 0.0003}
PREDICTED_VOLS = {"A": 0.012, "B": 0.009, "C": 0.006}
CORR_MATRIX = {
    "A": {"A": 1.0, "B": 0.6, "C": -0.2},
    "B": {"A": 0.6, "B": 1.0, "C": 0.1},
    "C": {"A": -0.2, "B": 0.1, "C": 1.0},
}
HOLDINGS_PCT = {
    "A": {"HDFC Bank": 9.0, "Infosys": 6.0},
    "B": {"HDFC Bank": 7.0, "ICICI Bank": 6.0},
    "C": {"ITC": 5.0, "L&T": 4.0},
}
ALL_SECTORS = ["Financials", "IT", "FMCG", "Industrials", "Unclassified/Other"]
SECTORS_DICT = {
    "A": {"Financials": 45.0, "IT": 20.0},
    "B": {"Financials": 50.0, "IT": 15.0},
    "C": {"FMCG": 30.0, "Industrials": 25.0},
}
CONSISTENCY_DICT = {
    "A": {"consistency": 0.8, "cv": 0.25, "num_windows": 6, "win_rate": 0.83},
    "B": {"consistency": 0.6, "cv": 0.67, "num_windows": 6, "win_rate": 0.67},
    "C": {"consistency": 0.9, "cv": 0.11, "num_windows": 6, "win_rate": 0.9},
}


def test_standardize_holdings_converts_scale():
    fractions = standardize_holdings_to_fractions(HOLDINGS_PCT)
    assert fractions["A"]["HDFC Bank"] == 0.09
    print("PASS: test_standardize_holdings_converts_scale")


def test_correlation_penalty_manual_check():
    weights = {"A": 0.5, "B": 0.5}
    corr = {"A": {"A": 1.0, "B": 0.6}, "B": {"A": 0.6, "B": 1.0}}
    result = portfolio_correlation_penalty(weights, corr)
    expected = 2 * 0.5 * 0.5 * 0.6
    assert abs(result - expected) < 1e-9
    print("PASS: test_correlation_penalty_manual_check")


def test_diversification_reward_bounds():
    weights = {"A": 0.5, "B": 0.5}
    vols = {"A": 0.01, "B": 0.01}
    d = diversification_reward(weights, vols, portfolio_vol=0.01)
    assert abs(d - 0.0) < 1e-9
    d2 = diversification_reward(weights, vols, portfolio_vol=0.005)
    assert abs(d2 - 0.5) < 1e-9
    print("PASS: test_diversification_reward_bounds")


def test_risk_tolerance_weights_endpoints():
    conservative = risk_tolerance_weights(0.0)
    aggressive = risk_tolerance_weights(1.0)
    assert conservative["alpha"] == 0.5 and aggressive["alpha"] == 1.0
    assert conservative["beta"] == 1.5 and aggressive["beta"] == 0.5
    assert conservative["eta"] == 0.8 and aggressive["eta"] == 0.5
    print("PASS: test_risk_tolerance_weights_endpoints")


def test_risk_tolerance_out_of_range_raises():
    try:
        risk_tolerance_weights(1.5)
        assert False
    except ValueError:
        pass
    print("PASS: test_risk_tolerance_out_of_range_raises")


def test_score_breakdown_has_all_terms():
    weights = {"A": 0.4, "B": 0.3, "C": 0.3}
    overlap_matrix = build_overlap_matrix(standardize_holdings_to_fractions(HOLDINGS_PCT))
    result = compute_prism_score(
        weights, PREDICTED_RETURNS, CORR_MATRIX, PREDICTED_VOLS,
        overlap_matrix, SECTORS_DICT, ALL_SECTORS, CONSISTENCY_DICT, tau=0.5
    )
    expected_keys = {"R_return", "sigma_risk", "D_diversification", "O_overlap",
                      "C_correlation", "H_sector", "K_consistency"}
    assert set(result["breakdown"].keys()) == expected_keys
    print("PASS: test_score_breakdown_has_all_terms")


def test_risk_tolerance_shifts_ranking_in_correct_direction():
    overlap_matrix = build_overlap_matrix(standardize_holdings_to_fractions(HOLDINGS_PCT))
    safe = {"A": 0.2, "B": 0.3, "C": 0.5}
    risky = {"A": 0.9, "B": 0.05, "C": 0.05}

    def score(w, tau):
        return compute_prism_score(
            w, PREDICTED_RETURNS, CORR_MATRIX, PREDICTED_VOLS,
            overlap_matrix, SECTORS_DICT, ALL_SECTORS, CONSISTENCY_DICT, tau
        )["total_score"]

    gap_conservative = score(risky, 0.0) - score(safe, 0.0)
    gap_aggressive = score(risky, 1.0) - score(safe, 1.0)

    assert gap_aggressive > gap_conservative, (
        f"Expected risky portfolio's relative standing to improve under "
        f"higher risk tolerance: {gap_aggressive} should be > {gap_conservative}"
    )
    print("PASS: test_risk_tolerance_shifts_ranking_in_correct_direction")


if __name__ == "__main__":
    test_standardize_holdings_converts_scale()
    test_correlation_penalty_manual_check()
    test_diversification_reward_bounds()
    test_risk_tolerance_weights_endpoints()
    test_risk_tolerance_out_of_range_raises()
    test_score_breakdown_has_all_terms()
    test_risk_tolerance_shifts_ranking_in_correct_direction()
    print("\nAll tests passed.")
