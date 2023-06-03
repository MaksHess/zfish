import aggregation_functions as agg
import numpy as np
from neighborhood_aggregation import aggregate_table_dense_parallel, aggregate_table_nan

from data import (
    get_empty_adjacency_matrix,
    get_full_adjacency_matrix,
    get_random_features,
    get_self_adjacency_matrix,
)


def test_empty_neighborhood_aggregation(neighborhood_aggregation_function):
    empty_nhd = get_empty_adjacency_matrix(n_obs=N_OBS)
    for aggfun in agg.get_aggregation_functions():
        result = neighborhood_aggregation_function(empty_nhd, FEAT, aggfun)
        if aggfun.__name__ == "Sum":
            assert np.allclose(result, np.full(FEAT.shape, 0.0))
        else:
            assert np.allclose(result, np.full(FEAT.shape, np.nan), equal_nan=True)


def test_self_neighborhood_aggregation(neighborhood_aggregation_function):
    self_nhd = get_self_adjacency_matrix(n_obs=N_OBS)
    for aggfun in [agg.Mean, agg.Median, agg.Max, agg.Min, agg.Sum, agg.CircMean]:
        result = neighborhood_aggregation_function(self_nhd, FEAT, aggfun)
        assert np.allclose(result, FEAT)

    for aggfun in [agg.Std, agg.Var, agg.CircVar]:
        result = neighborhood_aggregation_function(self_nhd, FEAT, aggfun)
        assert np.allclose(result, np.zeros_like(FEAT))

    for aggfun in [agg.CircR]:
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
