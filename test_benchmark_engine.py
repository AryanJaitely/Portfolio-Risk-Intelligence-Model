from benchmark_engine import (
    equal_weight_portfolio, generate_random_portfolio,
    average_random_portfolio_score, make_sharpe_score_fn, optimize_markowitz,
    compare_portfolios,
)
from covariance_engine import portfolio_return, portfolio_variance


def test_equal_weight_sums_to_one():
    w = equal_weight_portfolio(["A", "B", "C", "D"])
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(v == 0.25 for v in w.values())
    print("PASS: test_equal_weight_sums_to_one")


def test_random_portfolio_valid_simplex_point():
    for seed in range(20):
        w = generate_random_portfolio(["A", "B", "C"], seed=seed)
        assert abs(sum(w.values()) - 1.0) < 1e-9
        assert all(v >= 0 for v in w.values())
    print("PASS: test_random_portfolio_valid_simplex_point")


def test_random_portfolio_not_degenerate():
    samples = [generate_random_portfolio(["A", "B", "C"], seed=s) for s in range(100)]
    a_weights = [w["A"] for w in samples]
    spread = max(a_weights) - min(a_weights)
    assert spread > 0.5, f"Expected wide spread in random weights, got range {spread}"
    print("PASS: test_random_portfolio_not_degenerate")


def test_average_random_score_reasonable_bounds():
    def score_fn(weights):
        return weights["A"]

    result = average_random_portfolio_score(["A", "B", "C"], score_fn, num_samples=500, seed=1)
    assert abs(result["mean_score"] - (1 / 3)) < 0.05
    print("PASS: test_average_random_score_reasonable_bounds")


def test_markowitz_maximizes_sharpe_better_than_equal_weight():
    fund_ids = ["A", "B", "C"]
    predicted_returns = {"A": 0.0006, "B": 0.0005, "C": 0.0003}
    cov_matrix = {
        "A": {"A": 0.012**2, "B": 0.1 * 0.012 * 0.010, "C": -0.2 * 0.012 * 0.006},
        "B": {"A": 0.1 * 0.012 * 0.010, "B": 0.010**2, "C": 0.05 * 0.010 * 0.006},
        "C": {"A": -0.2 * 0.012 * 0.006, "B": 0.05 * 0.010 * 0.006, "C": 0.006**2},
    }
    markowitz_w = optimize_markowitz(fund_ids, predicted_returns, cov_matrix, num_iterations=1500, seed=1)

    equal_w = equal_weight_portfolio(fund_ids)
    sharpe_fn = make_sharpe_score_fn(predicted_returns, cov_matrix)

    assert sharpe_fn(markowitz_w) >= sharpe_fn(equal_w), "Markowitz should beat equal-weight on its OWN objective"
    print("PASS: test_markowitz_maximizes_sharpe_better_than_equal_weight")


def test_compare_portfolios_returns_all_expected_fields():
    fund_ids = ["A", "B"]
    predicted_returns = {"A": 0.0005, "B": 0.0003}
    predicted_vols = {"A": 0.01, "B": 0.008}
    cov_matrix = {"A": {"A": 0.0001, "B": 0.00002}, "B": {"A": 0.00002, "B": 0.000064}}

    def dummy_prism_score(weights):
        return weights["A"] * 2

    portfolios = {"Equal": equal_weight_portfolio(fund_ids)}
    rows = compare_portfolios(portfolios, predicted_returns, predicted_vols, cov_matrix, dummy_prism_score)

    assert len(rows) == 1
    for key in ["name", "weights", "return", "volatility", "sharpe", "prism_score"]:
        assert key in rows[0]
    print("PASS: test_compare_portfolios_returns_all_expected_fields")


if __name__ == "__main__":
    test_equal_weight_sums_to_one()
    test_random_portfolio_valid_simplex_point()
    test_random_portfolio_not_degenerate()
    test_average_random_score_reasonable_bounds()
    test_markowitz_maximizes_sharpe_better_than_equal_weight()
    test_compare_portfolios_returns_all_expected_fields()
    print("\nAll tests passed.")
