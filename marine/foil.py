"""NACA 4-digit sections and thin aerofoil theory.

Thin aerofoil theory gives lift for an ideal, inviscid, thin section:
    Cl = 2*pi*(alpha - alpha_L0)
where alpha_L0, the angle of zero lift, depends only on the camber line. It is
exact in its own limits and is used here in two ways: as a check on the XFOIL
results at small angles, and as the physics part of the hybrid surrogate.
"""
import numpy as np


def camber_line(x, m, p):
    """Mean camber line y_c and its slope for a NACA 4-digit section (m, p as fractions of chord)."""
    x = np.asarray(x, dtype=float)
    if m == 0 or p == 0:
        return np.zeros_like(x), np.zeros_like(x)
    front = x < p
    yc = np.where(front, m / p**2 * (2 * p * x - x**2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2))
    dyc = np.where(front, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    return yc, dyc


def coordinates(m, p, t, n=81):
    """Section outline from the upper trailing edge round the nose to the lower trailing edge.

    Cosine spacing puts more points near the nose and tail, where curvature is
    highest. The trailing edge is closed.
    """
    beta = np.linspace(0, np.pi, n)
    x = (1 - np.cos(beta)) / 2
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    yc, dyc = camber_line(x, m, p)
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    return np.r_[xu[::-1], xl[1:]], np.r_[yu[::-1], yl[1:]]


def zero_lift_angle(m, p, n=2000):
    """alpha_L0 in degrees: -(1/pi) * integral over theta of dyc/dx * (cos(theta) - 1)."""
    th = (np.arange(n) + 0.5) * np.pi / n                 # midpoint rule
    _, dyc = camber_line((1 - np.cos(th)) / 2, m, p)
    return np.degrees(-np.sum(dyc * (np.cos(th) - 1)) / n)


def thin_lift(alpha_deg, m, p):
    """Lift coefficient from thin aerofoil theory."""
    return 2 * np.pi * np.radians(np.asarray(alpha_deg) - np.vectorize(zero_lift_angle)(m, p))
