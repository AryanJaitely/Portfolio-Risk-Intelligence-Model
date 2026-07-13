from backtest_engine import (
    max_drawdown, evaluate_strategy, compute_portfolio_returns,
    generate_formation_and_test_returns,
)


def test_max_drawdown_zero_for_monotonic_gains():
    returns = [0.01] * 20
    assert max_drawdown(returns) == 0.0
    print("PASS: test_max_drawdown_zero_for_monotonic_gains")


def test_max_drawdown_detects_known_decline():
    returns = [0.02] * 5 + [-0.03] * 5 + [0.01] * 5
    mdd = max_drawdown(returns)
    assert mdd > 0.0
    assert mdd < 1.0
    print("PASS: test_max_drawdown_detects_known_decline")


def test_max_drawdown_never_negative():
    import random
    rng = random.Random(3)
    returns = [rng.uniform(-0.02, 0.02) for _ in range(100)]
    assert max_drawdown(returns) >= 0.0
    print("PASS: test_max_drawdown_never_negative")


def test_evaluate_strategy_returns_all_metrics():
    returns = [0.001, -0.0005, 0.0008, -0.0003, 0.0012] * 20
    metrics = evaluate_strategy(returns)
    for key in ["cumulative_return", "annualized_return", "annualized_volatility",
                "sharpe", "max_drawdown"]:
        assert key in metrics
    print("PASS: test_evaluate_strategy_returns_all_metrics")


def test_compute_portfolio_returns_manual_check():
    weights = {"A": 0.6, "B": 0.4}
    returns_dict = {"A": [0.01, 0.02, 0.03], "B": [0.02, -0.01, 0.01]}
    port_returns = compute_portfolio_returns(weights, returns_dict, start_idx=0, end_idx=3)
    expected = [0.6 * 0.01 + 0.4 * 0.02, 0.6 * 0.02 + 0.4 * -0.01, 0.6 * 0.03 + 0.4 * 0.01]
    for actual, exp in zip(port_returns, expected):
        assert abs(actual - exp) < 1e-9
    print("PASS: test_compute_portfolio_returns_manual_check")


def test_crisis_shock_only_affects_test_period():
    fund_ids = ["A", "B", "C"]
    means = {"A": 0.0005, "B": 0.0005, "C": 0.0003}
    vols = {"A": 0.01, "B": 0.01, "C": 0.006}
    corr = {"A": {"A": 1.0, "B": 0.1, "C": 0.0}, "B": {"A": 0.1, "B": 1.0, "C": 0.0},
            "C": {"A": 0.0, "B": 0.0, "C": 1.0}}

    formation_days, test_days = 100, 50
    returns, split_idx = generate_formation_and_test_returns(
        fund_ids, means, vols, corr, formation_days, test_days,
        crisis_shock=0.5, crisis_funds=["A"], seed=1
    )
    formation_A = returns["A"][:split_idx]
    assert all(r > -0.4 for r in formation_A), "Shock leaked into formation period!"
    print("PASS: test_crisis_shock_only_affects_test_period")


if __name__ == "__main__":
    test_max_drawdown_zero_for_monotonic_gains()
    test_max_drawdown_detects_known_decline()
    test_max_drawdown_never_negative()
    test_evaluate_strategy_returns_all_metrics()
    test_compute_portfolio_returns_manual_check()
    test_crisis_shock_only_affects_test_period()
    print("\nAll tests passed.")
