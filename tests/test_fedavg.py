import numpy as np

from fedgagan.aggregation import weighted_average_weights


def test_weighted_fedavg():
    first = [np.array([1.0, 3.0]), np.array([2.0])]
    second = [np.array([5.0, 7.0]), np.array([6.0])]
    result = weighted_average_weights([(first, 1), (second, 3)])
    np.testing.assert_allclose(result[0], [4.0, 6.0])
    np.testing.assert_allclose(result[1], [5.0])
