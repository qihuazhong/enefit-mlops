"""The live run scores with LightGBM in the main process, then retrains in worker processes. With Linux's
default "fork" those workers waited forever (the first AWS live run sat idle for 3 h); train.pool starts them fresh.
On macOS the default is already "spawn", so this test can only fail on Linux (CI, the image's test run)."""
import lightgbm as lgb
import numpy as np

from enefit import train as TR

X = np.random.default_rng(0).random((5000, 10))
Y = X @ np.arange(10)


def fit_small(_=None) -> int:
    return lgb.train({"verbose": -1, "num_threads": 2}, lgb.Dataset(X, Y), 10).num_trees()


def test_workers_run_after_lightgbm_ran_in_the_parent():
    lgb.train({"verbose": -1, "num_threads": 2}, lgb.Dataset(X, Y), 10).predict(X, num_threads=2)
    ex = TR.pool(1)
    try:
        assert ex.submit(fit_small).result(timeout=120) == 10
    finally:                       # if a worker hangs, kill it so the test fails instead of waiting forever
        for p in ex._processes.values():
            p.kill()
        ex.shutdown(wait=False)
