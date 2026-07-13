from optimizer import random_initial_weights, generate_neighbor, simulated_annealing


def test_random_initial_weights_sum_to_one():
    weights = random_initial_weights(["A", "B", "C", "D"])
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert all(w == 0.25 for w in weights.values())
    print("PASS: test_random_initial_weights_sum_to_one")


def test_neighbor_preserves_simplex_constraint():
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    for _ in range(200):
        neighbor = generate_neighbor(weights, step_size=0.1)
        assert abs(sum(neighbor.values()) - 1.0) < 1e-9
        assert all(w >= -1e-9 for w in neighbor.values())
    print("PASS: test_neighbor_preserves_simplex_constraint")


def test_neighbor_respects_max_weight_cap():
    weights = {"A": 0.65, "B": 0.2, "C": 0.15}
    for _ in range(200):
        neighbor = generate_neighbor(weights, step_size=0.2, max_weight=0.7)
        assert all(w <= 0.7 + 1e-9 for w in neighbor.values())
    print("PASS: test_neighbor_respects_max_weight_cap")


def test_sa_never_returns_worse_than_starting_point():
    def score_fn(weights):
        return weights["A"]

    result = simulated_annealing(
        ["A", "B", "C"], score_fn, initial_temp=1.0, cooling_rate=0.98,
        num_iterations=500, step_size=0.1, seed=1
    )
    starting_score = 1 / 3
    assert result["best_score"] >= starting_score
    print("PASS: test_sa_never_returns_worse_than_starting_point")


def test_sa_converges_toward_known_optimum():
    def score_fn(weights):
        return weights["A"]

    result = simulated_annealing(
        ["A", "B", "C"], score_fn, initial_temp=1.0, cooling_rate=0.97,
        num_iterations=1500, step_size=0.1, seed=7
    )
    assert result["best_weights"]["A"] > 0.8, result["best_weights"]
    print("PASS: test_sa_converges_toward_known_optimum")


def test_sa_output_weights_always_valid():
    def score_fn(weights):
        return weights["A"] - weights["B"]

    result = simulated_annealing(
        ["A", "B", "C"], score_fn, num_iterations=300, step_size=0.05, seed=3
    )
    for w_set in [result["best_weights"], result["final_weights"]]:
        assert abs(sum(w_set.values()) - 1.0) < 1e-9
        assert all(v >= -1e-9 for v in w_set.values())
    print("PASS: test_sa_output_weights_always_valid")


def test_score_history_length_matches_iterations():
    def score_fn(weights):
        return weights["A"]

    num_iter = 100
    result = simulated_annealing(["A", "B"], score_fn, num_iterations=num_iter, seed=1)
    assert len(result["score_history"]) == num_iter + 1
    print("PASS: test_score_history_length_matches_iterations")


if __name__ == "__main__":
    test_random_initial_weights_sum_to_one()
    test_neighbor_preserves_simplex_constraint()
    test_neighbor_respects_max_weight_cap()
    test_sa_never_returns_worse_than_starting_point()
    test_sa_converges_toward_known_optimum()
    test_sa_output_weights_always_valid()
    test_score_history_length_matches_iterations()
    print("\nAll tests passed.")
