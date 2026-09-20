from sim.factory.config import load_config


def test_config_has_exact_five_destinations():
    config = load_config()
    assert set(config.bins) == {
        "bin_hard",
        "bin_semi_hard",
        "bin_soft",
        "bin_fresh",
        "bin_blue",
    }
    assert config.model_path.name == "best.pt"
    assert config.section("perception")["service_url"].startswith("http://")
    assert config.section("perception")["service_timeout_s"] > 0
    assert config.section("perception")["decision_max_age_s"] > 0
    assert config.section("belt")["object_orientation_wxyz"] == [1.0, 0.0, 0.0, 0.0]


def test_evaluation_has_two_cheeses_per_bin_and_a_reject():
    from sim.factory.perception import BIN_OF_TYPE

    sequence = load_config().raw["evaluation_sequence"]
    counts = {name: 0 for name in load_config().bins}
    for label in sequence:
        if label in BIN_OF_TYPE:
            counts[BIN_OF_TYPE[label]] += 1
    assert min(counts.values()) >= 2
    assert "not_cheese" in sequence
