import importlib.util

import numpy as np
import pandas as pd
import pytest

from marine import sail

has_xfoil = importlib.util.find_spec("xfoil") is not None


def test_driving_force_geometry():
    # beam reach: all lift drives the ship forward, drag pushes sideways
    assert np.isclose(sail.driving_force(1.0, 0.1, 90), 1.0)
    # head to wind: no lift component forward, drag pulls back
    assert np.isclose(sail.driving_force(1.0, 0.1, 0), -0.1)


def test_usable_drops_points_past_stall_and_unconverged():
    d = pd.DataFrame({"section": 0, "alpha": [0, 4, 8, 12, 16], "cl": [0.4, 0.8, 1.2, 1.1, np.nan],
                      "cd": [0.01, 0.01, 0.02, 0.05, np.nan], "converged": [True, True, True, True, False]})
    assert sail.usable(d)["alpha"].tolist() == [0, 4, 8]


def test_hybrid_network_adds_thin_aerofoil_theory():
    X = np.array([[0.04, 0.4, 0.12, 5.0, 6.5]])
    net = sail.FoilNet(hybrid=True)
    assert np.isclose(net._theory(X)[0], 2 * np.pi * np.radians(5.0 + 4.154), atol=1e-3)
    assert sail.FoilNet(hybrid=False)._theory(X)[0] == 0


def test_latin_hypercube_stays_in_design_box():
    s = sail.sample_sections(50)
    for k, (lo, hi) in sail.DESIGN.items():
        assert s[k].between(lo, hi).all()
    assert s["re"].between(*sail.RE).all()


@pytest.mark.skipif(not has_xfoil, reason="xfoil package not installed")
def test_xfoil_symmetric_section_has_no_lift_at_zero_angle():
    pol = sail.run_xfoil(0.0, 0.0, 0.12, 3e6, a_max=4).set_index("alpha")
    assert abs(pol.loc[0.0, "cl"]) < 1e-3
    assert 0.004 < pol.loc[0.0, "cd"] < 0.008
