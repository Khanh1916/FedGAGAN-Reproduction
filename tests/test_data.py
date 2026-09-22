import numpy as np

from fedgagan.data import make_client_partitions


def test_iid_partition_preserves_all_rows():
    data = np.arange(120, dtype=np.float32).reshape(30, 4)
    clients = make_client_partitions(data, 4, seed=7, method="iid")
    assert sum(map(len, clients)) == len(data)
    assert max(map(len, clients)) - min(map(len, clients)) <= 1


def test_dirichlet_partition_has_no_empty_client():
    data = np.arange(200, dtype=np.float32).reshape(50, 4)
    clients = make_client_partitions(data, 5, seed=7, method="dirichlet", alpha=0.4)
    assert sum(map(len, clients)) == len(data)
    assert min(map(len, clients)) >= 2
