import numpy as np
import pytest

from marine import hull


@pytest.fixture(scope="module")
def data():
    return hull.load()


def test_series_shape(data):
    assert len(data) == 308
    assert data["hull"].nunique() == 22
    assert (data.groupby("hull").size() == 14).all()


def test_checks_find_duplicated_curve(data):
    found = hull.check(data)
    assert any("identical resistance curves" in f for f in found)


def test_vectorised_regression_matches_sklearn(data):
    X, y = data[hull.FEATURES].to_numpy(), np.log(data["rr"].to_numpy())
    m = hull.PerSpeedRegression().fit(X, y)
    direct = np.column_stack([m.models[f].predict(X[:, :-1]) for f in m.fns])
    expected = np.array([np.interp(fn, m.fns, row) for fn, row in zip(X[:, -1], direct)])
    assert np.allclose(m.predict(X), expected)


def test_monotonic_trees_never_dip_with_speed(data):
    X, y = data[hull.FEATURES].to_numpy(), np.log(data["rr"].to_numpy())
    m = hull.boosted(monotonic=True).fit(X, y)
    rng = np.random.default_rng(1)
    lo, hi = X[:, :-1].min(0), X[:, :-1].max(0)
    for form in lo + rng.random((50, 5)) * (hi - lo):
        sweep = np.column_stack([np.tile(form, (len(hull.fn_grid()), 1)), hull.fn_grid()])
        assert (np.diff(m.predict(sweep)) >= -1e-9).all()


def test_bootstrap_spread_grows_away_from_tested_hulls(data):
    X, y = data[hull.FEATURES].to_numpy(), np.log(data["rr"].to_numpy())
    b = hull.HullBootstrap(n=50).fit(X, y, data["hull"].to_numpy())
    parent = X[(data["hull"] == 0).to_numpy()][:1]
    corner = parent.copy()
    corner[0, :5] = X[:, :5].max(0)                     # every parameter at its maximum at once
    assert b.predict_dist(corner)[1][0] > b.predict_dist(parent)[1][0]


def test_optimiser_respects_fixed_length_displacement(data):
    out = hull.optimise(data, caution=1.0)
    assert out["design"]["l_disp"] == 4.78
    assert out["predicted_rr"] > 0
