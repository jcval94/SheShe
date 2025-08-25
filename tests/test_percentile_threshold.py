import numpy as np
from sheshe.mbc_fast import percentile_threshold_auto


def test_small_array_partition():
    values = np.arange(100, dtype=float)[::-1]
    q = 0.2
    expected = np.sort(values)[int(np.floor(q * (len(values) - 1)))]
    auto = percentile_threshold_auto(values, q)
    part = percentile_threshold_auto(values, q, method="partition")
    assert auto == part == expected


def test_large_array_hist_mode():
    rng = np.random.default_rng(0)
    values = rng.random(2_500_000, dtype=np.float32)
    q = 0.73
    auto = percentile_threshold_auto(values, q, sample_size=None, rng_seed=0)
    hist = percentile_threshold_auto(values, q, sample_size=None, method="hist", rng_seed=0)
    assert auto == hist
    assert abs(auto - q) < 1e-2


def test_auto_switches_to_partition_after_sampling():
    rng = np.random.default_rng(0)
    values = rng.random(2_500_000, dtype=np.float32)
    q = 0.4
    auto = percentile_threshold_auto(values, q, sample_size=1000, rng_seed=0)
    part = percentile_threshold_auto(values, q, sample_size=1000, method="partition", rng_seed=0)
    hist = percentile_threshold_auto(values, q, sample_size=1000, method="hist", rng_seed=0)
    assert auto == part
    assert abs(auto - hist) > 1e-4


def test_stratified_sampling_with_y():
    values = np.concatenate([np.zeros(995), np.ones(5)])
    y = np.concatenate([np.zeros(995, dtype=int), np.ones(5, dtype=int)])
    q = 1.0
    full = percentile_threshold_auto(values, q, sample_size=None, method="partition")
    strat = percentile_threshold_auto(values, q, y=y, sample_size=20, method="partition", rng_seed=0)
    rnd = percentile_threshold_auto(values, q, sample_size=20, method="partition", rng_seed=0)
    assert strat == full == 1.0
    assert rnd == 0.0
