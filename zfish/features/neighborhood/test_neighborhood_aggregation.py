import numpy as np

import aggregation_functions as agg
from data import (
    get_empty_adjacency_matrix,
    get_full_adjacency_matrix,
    get_random_features,
    get_self_adjacency_matrix,
)
from neighborhood_aggregation import aggregate_table_dense_parallel, aggregate_table_nan


def test_empty_neighborhood_aggregation(neighborhood_aggregation_function):
    empty_nhd = get_empty_adjacency_matrix(n_obs=N_OBS)
    for aggfun in agg.get_aggregation_functions():
        result = neighborhood_aggregation_function(empty_nhd, FEAT, aggfun)
        if aggfun.__name__ == "SUM":
            assert np.allclose(result, np.full(FEAT.shape, 0.0))
        else:
            assert np.allclose(result, np.full(FEAT.shape, np.nan), equal_nan=True)


def test_self_neighborhood_aggregation(neighborhood_aggregation_function):
    self_nhd = get_self_adjacency_matrix(n_obs=N_OBS)
    for aggfun in [agg.MEAN, agg.MEDIAN, agg.MAX, agg.MIN, agg.SUM, agg.CIRCMEAN]:
        result = neighborhood_aggregation_function(self_nhd, FEAT, aggfun)
        assert np.allclose(result, FEAT)

    for aggfun in [agg.STD, agg.VAR, agg.CIRCVAR]:
        result = neighborhood_aggregation_function(self_nhd, FEAT, aggfun)
        assert np.allclose(result, np.zeros_like(FEAT))

    for aggfun in [agg.CIRCR]:
        result = neighborhood_aggregation_function(self_nhd, FEAT, aggfun)
        assert np.allclose(result, np.ones_like(FEAT))


def test_full_neighborhood_aggregation(neighborhood_aggregation_function):
    full_nhd = get_full_adjacency_matrix(n_obs=N_OBS)
    for aggfun in agg.get_aggregation_functions():
        result = neighborhood_aggregation_function(full_nhd, FEAT, aggfun)
        expected = np.array([[aggfun(FEAT[:, i]) for i in range(FEAT.shape[1])]])
        assert np.all(result == expected)


if __name__ == "__main__":
    N_OBS = 10
    N_VAR = 1
    FEAT = get_random_features(n_obs=N_OBS, n_var=N_VAR)
    print(f"{N_OBS=}")
    print(f"{N_VAR=}")
    print(f"testing `aggregate_table_dense_parallel`")
    test_empty_neighborhood_aggregation(aggregate_table_dense_parallel)
    test_self_neighborhood_aggregation(aggregate_table_dense_parallel)
    test_full_neighborhood_aggregation(aggregate_table_dense_parallel)
    print(f"testing `aggregate_table_nan`")
    test_empty_neighborhood_aggregation(aggregate_table_nan)
    test_self_neighborhood_aggregation(aggregate_table_nan)
    test_full_neighborhood_aggregation(aggregate_table_nan)

    N_OBS = 10
    N_VAR = 3
    FEAT = get_random_features(n_obs=N_OBS, n_var=N_VAR)
    print(f"{N_OBS=}")
    print(f"{N_VAR=}")
    print(f"testing `aggregate_table_dense_parallel`")
    test_empty_neighborhood_aggregation(aggregate_table_dense_parallel)
    test_self_neighborhood_aggregation(aggregate_table_dense_parallel)
    test_full_neighborhood_aggregation(aggregate_table_dense_parallel)
    print(f"testing `aggregate_table_nan`")
    test_empty_neighborhood_aggregation(aggregate_table_nan)
    test_self_neighborhood_aggregation(aggregate_table_nan)
    test_full_neighborhood_aggregation(aggregate_table_nan)

    N_OBS = 1000
    N_VAR = 3
    FEAT = get_random_features(n_obs=N_OBS, n_var=N_VAR)
    print(f"{N_OBS=}")
    print(f"{N_VAR=}")
    print(f"testing `aggregate_table_dense_parallel`")
    test_empty_neighborhood_aggregation(aggregate_table_dense_parallel)
    test_self_neighborhood_aggregation(aggregate_table_dense_parallel)
    test_full_neighborhood_aggregation(aggregate_table_dense_parallel)
    print(f"testing `aggregate_table_nan`")
    test_empty_neighborhood_aggregation(aggregate_table_nan)
    test_self_neighborhood_aggregation(aggregate_table_nan)
    test_full_neighborhood_aggregation(aggregate_table_nan)
