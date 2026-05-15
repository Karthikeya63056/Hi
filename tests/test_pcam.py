import numpy as np

from anvil_pcam.core import PCAMEngine, evaluate_trial, predict_precision


def test_predict_precision_contract():
    query = np.zeros(64)
    query[0] = 1.0
    precision = predict_precision(query)

    assert precision.shape == (64,)
    assert np.all(precision > 0.0)
    assert np.all(precision >= 0.1)
    assert np.all(precision <= 10.0)
    np.testing.assert_allclose(precision.mean(), 1.0, atol=1e-9)


def test_adaptive_precision_changes_retrieval_path():
    engine = PCAMEngine()
    trial = evaluate_trial(engine, "A03", gaussian_sigma=0.4, mask_fraction=0.2, seed=11)

    baseline_energy = [point["energy"] for point in trial["baseline"]["trace"]]
    adaptive_energy = [point["energy"] for point in trial["adaptive"]["trace"]]

    assert trial["initialPrecisionSummary"]["anisotropy"] > 0.0
    assert baseline_energy != adaptive_energy
    assert trial["adaptive"]["trace"][-1]["targetScore"] is not None


def test_store_graph_and_trial_payload_are_demo_ready():
    engine = PCAMEngine()
    trial = evaluate_trial(engine, "A01", seed=5)

    assert len(trial["target"]["vector"]) == 64
    assert len(trial["corruptedQuery"]) == 64
    assert len(trial["initialPrecision"]) == 64
    assert trial["graph"]["nodes"]
    assert "baseline" in trial["metrics"]
    assert "adaptive" in trial["metrics"]
