import math

from consistency_engine import (
    rolling_windows,
    window_sharpe,
    fund_consistency_score,
    portfolio_consistency,
    explain_consistency,
)


def test_rolling_windows_basic_split():
    returns = list(range(12))
    windows = rolling_windows(returns, window_length=4)
    assert len(windows) == 3
    assert windows[0] == [0, 1, 2, 3]
    assert windows[1] == [4, 5, 6, 7]
    assert windows[2] == [8, 9, 10, 11]
    print("PASS: test_rolling_windows_basic_split")


def test_rolling_windows_discards_leftover():
    returns = list(range(13))
    windows = rolling_windows(returns, window_length=4)
    assert len(windows) == 3
    print("PASS: test_rolling_windows_discards_leftover")


def test_rolling_windows_raises_if_too_few_windows():
    returns = list(range(5))
    try:
        rolling_windows(returns, window_length=4)
        assert False, "Expected ValueError for insufficient windows"
    except ValueError:
        pass
    print("PASS: test_rolling_windows_raises_if_too_few_windows")


def test_window_sharpe_zero_volatility_returns_none():
    constant_window = [0.01, 0.01, 0.01, 0.01]
    assert window_sharpe(constant_window) is None
    print("PASS: test_window_sharpe_zero_volatility_returns_none")


def test_identical_windows_give_zero_cv_and_perfect_consistency():
    pattern = [0.01, -0.005, 0.008, -0.002]
    returns = pattern * 6
    result = fund_consistency_score(returns, window_length=4)
    assert abs(result["cv"] - 0.0) < 1e-9, result["cv"]
    assert abs(result["consistency"] - 1.0) < 1e-9, result["consistency"]
    print("PASS: test_identical_windows_give_zero_cv_and_perfect_consistency")


def test_mean_sharpe_exactly_zero_gives_consistency_zero():
    window_a = [0.01, 0.02, 0.03, 0.04]
    window_b = [-x for x in window_a]
    returns = window_a + window_b

    sharpe_a = window_sharpe(window_a)
    sharpe_b = window_sharpe(window_b)
    assert abs(sharpe_a + sharpe_b) < 1e-9, "Test setup error: Sharpes should be exact negatives"

    result = fund_consistency_score(returns, window_length=4)
    assert result["cv"] == math.inf
    assert result["consistency"] == 0.0
    print("PASS: test_mean_sharpe_exactly_zero_gives_consistency_zero")


def test_more_stable_fund_scores_higher_than_less_stable_fund():
    stable_pattern = [0.012, -0.004, 0.009, -0.003] * 6
    unstable_windows = (
        [0.05, 0.04, 0.045, 0.048] +
        [-0.04, -0.05, -0.045, -0.042] +
        [0.03, 0.028, 0.032, 0.029] +
        [-0.035, -0.03, -0.038, -0.033] +
        [0.02, 0.018, 0.022, 0.019] +
        [-0.015, -0.012, -0.018, -0.01]
    )

    stable_result = fund_consistency_score(stable_pattern, window_length=4)
    unstable_result = fund_consistency_score(unstable_windows, window_length=4)

    assert stable_result["consistency"] > unstable_result["consistency"]
    print("PASS: test_more_stable_fund_scores_higher_than_less_stable_fund")


def test_win_rate_calculation():
    windows_data = (
        [0.01, 0.02, 0.01, 0.02] +
        [0.01, 0.02, 0.01, 0.02] +
        [-0.03, -0.02, -0.03, -0.02] +
        [0.01, 0.02, 0.01, 0.02]
    )
    result = fund_consistency_score(windows_data, window_length=4)
    assert abs(result["win_rate"] - 0.75) < 1e-9, result["win_rate"]
    print("PASS: test_win_rate_calculation")


def test_portfolio_consistency_matches_manual_weighted_average():
    consistency_dict = {
        "A": {"consistency": 0.9, "cv": 0.1, "num_windows": 5, "win_rate": 0.8},
        "B": {"consistency": 0.3, "cv": 2.0, "num_windows": 5, "win_rate": 0.4},
    }
    weights = {"A": 0.6, "B": 0.4}
    result = portfolio_consistency(weights, consistency_dict)
    expected = 0.6 * 0.9 + 0.4 * 0.3
    assert abs(result - expected) < 1e-9
    print("PASS: test_portfolio_consistency_matches_manual_weighted_average")


def test_explanation_reflects_band():
    high_result = {"consistency": 0.85, "win_rate": 0.9, "num_windows": 8}
    low_result = {"consistency": 0.1, "win_rate": 0.2, "num_windows": 8}
    high_text = explain_consistency("FundHigh", high_result)
    low_text = explain_consistency("FundLow", low_result)
    assert "HIGHLY" in high_text
    assert "INCONSISTENT" in low_text
    print("PASS: test_explanation_reflects_band")


if __name__ == "__main__":
    print("=== Running PRISM Module 6 verification tests ===\n")
    test_rolling_windows_basic_split()
    test_rolling_windows_discards_leftover()
    test_rolling_windows_raises_if_too_few_windows()
    test_window_sharpe_zero_volatility_returns_none()
    test_identical_windows_give_zero_cv_and_perfect_consistency()
    test_mean_sharpe_exactly_zero_gives_consistency_zero()
    test_more_stable_fund_scores_higher_than_less_stable_fund()
    test_win_rate_calculation()
    test_portfolio_consistency_matches_manual_weighted_average()
    test_explanation_reflects_band()
    print("\nAll tests passed. Consistency engine verified correct.")
