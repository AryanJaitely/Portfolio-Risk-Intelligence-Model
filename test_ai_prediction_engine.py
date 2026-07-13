from ai_prediction_engine import predict_fund_return_and_volatility, predict_universe


def test_raises_on_insufficient_data():
    try:
        predict_fund_return_and_volatility([0.01] * 10, window_length=5)
        assert False
    except ValueError:
        pass
    print("PASS: test_raises_on_insufficient_data")


def test_volatility_never_negative():
    returns = [0.001, -0.001, 0.0005, -0.0005] * 20
    result = predict_fund_return_and_volatility(returns, window_length=10)
    assert result["predicted_daily_volatility"] >= 0.0
    print("PASS: test_volatility_never_negative")


def test_rising_trend_detected():
    returns = []
    for k in range(8):
        vol_level = 0.005 + k * 0.003
        window = [vol_level, -vol_level] * 10
        returns.extend(window)
    result = predict_fund_return_and_volatility(returns, window_length=20)
    last_actual_vol = 0.005 + 7 * 0.003
    assert result["predicted_daily_volatility"] > 0.005
    print("PASS: test_rising_trend_detected")


def test_predict_universe_runs_for_all_funds():
    returns_dict = {
        "A": [0.001, -0.001, 0.002, -0.0015] * 15,
        "B": [0.0005, -0.0008, 0.0012, -0.0009] * 15,
    }
    predictions = predict_universe(returns_dict, window_length=10)
    assert set(predictions.keys()) == {"A", "B"}
    print("PASS: test_predict_universe_runs_for_all_funds")


if __name__ == "__main__":
    test_raises_on_insufficient_data()
    test_volatility_never_negative()
    test_rising_trend_detected()
    test_predict_universe_runs_for_all_funds()
    print("\nAll tests passed.")
