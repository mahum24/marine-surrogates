import numpy as np

from marine import foil


def test_zero_lift_angle_matches_textbook_values():
    # thin aerofoil theory: NACA 4412 -> -4.15 deg, NACA 2412 -> -2.08 deg, symmetric -> 0
    assert abs(foil.zero_lift_angle(0.04, 0.4) - -4.15) < 0.01
    assert abs(foil.zero_lift_angle(0.02, 0.4) - -2.08) < 0.01
    assert foil.zero_lift_angle(0, 0) == 0


def test_lift_slope_is_two_pi_per_radian():
    cl = foil.thin_lift([0, 1], 0.02, 0.4)
    assert np.isclose(cl[1] - cl[0], 2 * np.pi * np.pi / 180)


def test_coordinates_closed_and_thickness_right():
    x, y = foil.coordinates(0, 0, 0.12)
    assert np.isclose(x[0], 1) and np.isclose(x[-1], 1)
    assert np.isclose(y[0], 0, atol=1e-9) and np.isclose(y[-1], 0, atol=1e-9)
    # symmetric section: maximum thickness 12% of chord
    assert abs((y.max() - y.min()) - 0.12) < 0.002
