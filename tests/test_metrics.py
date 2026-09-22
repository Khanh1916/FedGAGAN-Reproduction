import numpy as np

from fedgagan.metrics import correlation_distances, histogram_kl, privacy_metrics


def test_identical_arrays_have_zero_kl_and_correlation_distance():
    rng = np.random.default_rng(3)
    data = rng.normal(size=(100, 5))
    assert histogram_kl(data, data) < 1e-10
    rmse, mae = correlation_distances(data, data)
    assert rmse < 1e-10
    assert mae < 1e-10


def test_privacy_duplicate_count():
    real = np.array([[0.0, 1.0], [2.0, 3.0]])
    fake = np.array([[0.0, 1.0], [4.0, 5.0]])
    report = privacy_metrics(real, fake)
    assert report["exact_duplicate_rows"] == 1

