from explanation_engine import (
    classify_weight, explain_fund_allocation, explain_portfolio_summary,
    generate_full_report,
)
from prism_score import compute_prism_score, standardize_holdings_to_fractions
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


def test_classify_weight_bands():
    assert classify_weight(0.35) == "core holding"
    assert classify_weight(0.15) == "supporting holding"
    assert classify_weight(0.02) == "minor holding"
    assert classify_weight(0.001) == "excluded / negligible"
    print("PASS: test_classify_weight_bands")


def test_excluded_fund_explanation_mentions_overlap_reason():
    weights = {"A": 0.6, "B": 0.0, "C": 0.4}
    overlap_matrix = build_overlap_matrix(standardize_holdings_to_fractions(HOLDINGS_PCT))
    explanation = explain_fund_allocation(
        "B", weights, PREDICTED_RETURNS, PREDICTED_VOLS,
        CONSISTENCY_DICT, overlap_matrix, HOLDINGS_PCT
    )
    assert "excluded" in explanation.lower()
    assert "HDFC Bank" in explanation
    print("PASS: test_excluded_fund_explanation_mentions_overlap_reason")


def test_portfolio_summary_reflects_negative_correlation():
    weights = {"A": 0.5, "B": 0.0, "C": 0.5}
    overlap_matrix = build_overlap_matrix(standardize_holdings_to_fractions(HOLDINGS_PCT))
    score_result = compute_prism_score(
        weights, PREDICTED_RETURNS, CORR_MATRIX, PREDICTED_VOLS,
        overlap_matrix, SECTORS_DICT, ALL_SECTORS, CONSISTENCY_DICT, tau=0.5
    )
    summary = explain_portfolio_summary(weights, score_result, SECTORS_DICT, ALL_SECTORS)
    assert "NEGATIVE" in summary
    print("PASS: test_portfolio_summary_reflects_negative_correlation")


def test_full_report_contains_all_funds():
    weights = {"A": 0.5, "B": 0.0, "C": 0.5}
    overlap_matrix = build_overlap_matrix(standardize_holdings_to_fractions(HOLDINGS_PCT))
    score_result = compute_prism_score(
        weights, PREDICTED_RETURNS, CORR_MATRIX, PREDICTED_VOLS,
        overlap_matrix, SECTORS_DICT, ALL_SECTORS, CONSISTENCY_DICT, tau=0.5
    )
    report = generate_full_report(
        weights, PREDICTED_RETURNS, PREDICTED_VOLS, overlap_matrix,
        HOLDINGS_PCT, SECTORS_DICT, ALL_SECTORS, CONSISTENCY_DICT, score_result
    )
    for fund_id in ["A", "B", "C"]:
        assert fund_id in report
    assert "PRISM PORTFOLIO EXPLANATION REPORT" in report
    print("PASS: test_full_report_contains_all_funds")


def test_report_sorted_by_descending_weight():
    weights = {"A": 0.2, "B": 0.0, "C": 0.8}
    overlap_matrix = build_overlap_matrix(standardize_holdings_to_fractions(HOLDINGS_PCT))
    score_result = compute_prism_score(
        weights, PREDICTED_RETURNS, CORR_MATRIX, PREDICTED_VOLS,
        overlap_matrix, SECTORS_DICT, ALL_SECTORS, CONSISTENCY_DICT, tau=0.5
    )
    report = generate_full_report(
        weights, PREDICTED_RETURNS, PREDICTED_VOLS, overlap_matrix,
        HOLDINGS_PCT, SECTORS_DICT, ALL_SECTORS, CONSISTENCY_DICT, score_result
    )
    assert report.index("C —") < report.index("A —") < report.index("B —")
    print("PASS: test_report_sorted_by_descending_weight")


if __name__ == "__main__":
    test_classify_weight_bands()
    test_excluded_fund_explanation_mentions_overlap_reason()
    test_portfolio_summary_reflects_negative_correlation()
    test_full_report_contains_all_funds()
    test_report_sorted_by_descending_weight()
    print("\nAll tests passed.")
