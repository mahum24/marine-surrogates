"""Rigid wingsail sections: XFOIL polars, a hybrid surrogate and a section optimiser.

XFOIL (Drela, MIT) solves the flow round a 2D section with a panel method for
the outer, inviscid flow coupled to an integral boundary-layer model, including
laminar-to-turbulent transition. It is fast and well validated up to moderate
angles, and becomes unreliable past stall, so only converged points are kept and
the optimiser keeps a margin from stall.

Surrogate: a neural network that predicts lift and drag for any NACA 4-digit
section, angle and Reynolds number. In the hybrid version the network learns
only the correction to thin aerofoil theory for lift, so the physics carries the
part it gets right and the data fills in viscous and thickness effects.
"""
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.optimize import differential_evolution
from scipy.stats import qmc

from marine import foil

DESIGN = {"m": (0.0, 0.06), "p": (0.2, 0.6), "t": (0.09, 0.21)}   # camber, camber position, thickness
RE = (1e6, 1e7)
INPUTS = ["m", "p", "t", "alpha", "log_re"]


# ---------------------------------------------------------------- XFOIL

def run_xfoil(m, p, t, re, a_max=20.0, step=0.5, panels=160, n_crit=9.0):
    """Polar for one section: sweep up from 0 deg, then down from 0 deg, from a fresh boundary layer."""
    from xfoil import XFoil
    from xfoil.model import Airfoil

    xf = XFoil()
    xf.print = False
    x, y = foil.coordinates(m, p, t, n=121)
    xf.airfoil = Airfoil(x, y)
    xf.repanel(n_nodes=panels)
    xf.Re, xf.max_iter, xf.n_crit = re, 200, n_crit
    up = xf.aseq(0.0, a_max + step, step)
    xf.reset_bls()
    down = xf.aseq(-step, -4.0 - step, -step)
    a, cl, cd, cm = (np.r_[u, d] for u, d in zip(up[:4], down[:4]))
    want = np.r_[np.arange(0, a_max + step / 2, step), -np.arange(step, 4 + step / 2, step)]
    out = pd.DataFrame({"alpha": want, "cl": cl, "cd": cd, "cm": cm})
    out["converged"] = np.isfinite(a) & np.isfinite(cd) & (cd > 0)
    return out.assign(m=m, p=p, t=t, re=re).sort_values("alpha").reset_index(drop=True)


def sample_sections(n, seed=0):
    """Latin hypercube over camber, camber position, thickness and log Reynolds number."""
    lo = [DESIGN[k][0] for k in "mpt"] + [np.log10(RE[0])]
    hi = [DESIGN[k][1] for k in "mpt"] + [np.log10(RE[1])]
    s = qmc.scale(qmc.LatinHypercube(d=4, seed=seed).random(n), lo, hi)
    return pd.DataFrame({"section": range(n), "m": s[:, 0], "p": s[:, 1], "t": s[:, 2], "re": 10 ** s[:, 3]})


def build_dataset(n=300, seed=0, jobs=-1):
    sections = sample_sections(n, seed)
    polars = Parallel(n_jobs=jobs)(delayed(run_xfoil)(r.m, r.p, r.t, r.re) for r in sections.itertuples())
    return pd.concat([pl.assign(section=i) for i, pl in enumerate(polars)], ignore_index=True)


def panel_convergence(sections=((0, 0, 0.12), (0.04, 0.4, 0.12)), re=3e6, alphas=(0, 4, 8, 12),
                      panels=(80, 120, 160, 200, 260, 320)):
    """Verification: how much lift and drag change as the surface is divided into more panels."""
    rows = []
    for m, p, t in sections:
        for n in panels:
            pol = run_xfoil(m, p, t, re, a_max=max(alphas), step=0.5, panels=n).set_index("alpha")
            for a in alphas:
                rows.append({"section": naca_name(m, p, t), "panels": n, "alpha": a,
                             "cl": pol.loc[a, "cl"], "cd": pol.loc[a, "cd"]})
    out = pd.DataFrame(rows)
    finest = out[out["panels"] == max(panels)].set_index(["section", "alpha"])[["cl", "cd"]]
    out = out.join(finest, on=["section", "alpha"], rsuffix="_finest")
    out["cl_change_pct"] = 100 * (out["cl"] - out["cl_finest"]).abs() / out["cl_finest"].abs().clip(lower=0.05)
    out["cd_change_pct"] = 100 * (out["cd"] - out["cd_finest"]).abs() / out["cd_finest"]
    return out


def thin_theory_check(data, a_range=(-2.0, 4.0)):
    """Validation against theory: lift slope and zero-lift angle from XFOIL at small angles,
    compared with thin aerofoil theory (2*pi per radian, alpha_L0 from the camber line)."""
    rows = []
    for sec, g in data[data["converged"] & data["alpha"].between(*a_range)].groupby("section"):
        if len(g) < 6:
            continue
        slope, icpt = np.polyfit(np.radians(g["alpha"]), g["cl"], 1)
        r = g.iloc[0]
        rows.append({"section": sec, "m": r.m, "p": r.p, "t": r.t, "re": r.re,
                     "slope_ratio": slope / (2 * np.pi),
                     "alpha_l0_xfoil": np.degrees(-icpt / slope),
                     "alpha_l0_theory": foil.zero_lift_angle(r.m, r.p)})
    return pd.DataFrame(rows)


def naca_name(m, p, t):
    return f"NACA {round(m * 100)}{round(p * 10)}{round(t * 100):02d}"


# ---------------------------------------------------------------- surrogate

def features(df):
    return np.column_stack([df["m"], df["p"], df["t"], df["alpha"], np.log10(df["re"])])


class FoilNet:
    """Neural network for (Cl, log Cd). With hybrid=True it predicts Cl minus thin aerofoil theory."""
    def __init__(self, hybrid=True, hidden=96, epochs=2500, lr=3e-3, seed=0):
        self.hybrid, self.hidden, self.epochs, self.lr, self.seed = hybrid, hidden, epochs, lr, seed

    def _theory(self, X):
        if not self.hybrid:
            return np.zeros(len(X))
        return 2 * np.pi * np.radians(X[:, 3] - self._al0(X[:, 0], X[:, 1]))

    @staticmethod
    def _al0(m, p):
        return np.array([foil.zero_lift_angle(a, b) for a, b in zip(m, p)])

    def fit(self, X, cl, cd):
        torch.manual_seed(self.seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        target = np.column_stack([cl - self._theory(X), np.log(cd)])
        self.tmu, self.tsd = target.mean(0), target.std(0) + 1e-9
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        Yt = torch.tensor((target - self.tmu) / self.tsd, dtype=torch.float32)
        h = self.hidden
        self.net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], h), torch.nn.SiLU(), torch.nn.Linear(h, h),
                                       torch.nn.SiLU(), torch.nn.Linear(h, h), torch.nn.SiLU(), torch.nn.Linear(h, 2))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = torch.mean((self.net(Xt) - Yt) ** 2)
            loss.backward()
            opt.step()
            sched.step()
        self.net.eval()
        return self

    def predict(self, X):
        with torch.no_grad():
            out = self.net(torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)).numpy() * self.tsd + self.tmu
        return out[:, 0] + self._theory(X), np.exp(out[:, 1])


def split_sections(data, test_share=0.2, seed=0):
    secs = data["section"].unique()
    rng = np.random.default_rng(seed)
    test = rng.choice(secs, int(len(secs) * test_share), replace=False)
    return np.setdiff1d(secs, test), test


def usable(data):
    """Converged points up to the angle of maximum lift: XFOIL is not trusted beyond stall."""
    d = data[data["converged"]].copy()
    stall = d.loc[d.groupby("section")["cl"].idxmax(), ["section", "alpha"]].set_index("section")["alpha"]
    return d[d["alpha"] <= d["section"].map(stall)]


def learning_curve(data, sizes=(10, 20, 40, 80, 160, None), seeds=(0, 1, 2), epochs=2500):
    """Error on unseen sections as the number of training sections grows, plain vs hybrid network."""
    d = usable(data)
    train_secs, test_secs = split_sections(d)
    test = d[d["section"].isin(test_secs)]
    Xte = features(test)
    rows = []
    for n in sizes:
        for s in seeds:
            rng = np.random.default_rng(s)
            pick = train_secs if n is None else rng.choice(train_secs, n, replace=False)
            tr = d[d["section"].isin(pick)]
            for hybrid in (False, True):
                cl, cd = FoilNet(hybrid=hybrid, epochs=epochs, seed=s).fit(features(tr), tr["cl"].to_numpy(),
                                                                          tr["cd"].to_numpy()).predict(Xte)
                rows.append({"sections": len(pick), "seed": s, "model": "Hybrid (theory + network)" if hybrid else "Network only",
                             "cl_mae": np.abs(cl - test["cl"]).mean(),
                             "cd_median_pct": 100 * np.median(np.abs(cd - test["cd"]) / test["cd"])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- section optimiser

def driving_force(cl, cd, awa_deg):
    """Thrust coefficient along the ship's heading for apparent wind angle `awa_deg` off the bow."""
    b = np.radians(awa_deg)
    return cl * np.sin(b) - cd * np.cos(b)


def optimise_section(model, awa_deg, re=5e6, t_min=0.12, stall_margin=2.0, seed=0):
    """Best camber, camber position, thickness and angle of attack for one apparent wind angle.

    Constraints: thickness at least t_min (the section must hold a mast or spar) and
    the angle kept stall_margin degrees below the predicted angle of maximum lift,
    because the apparent wind keeps changing at sea.
    """
    grid = np.arange(-4, 20.01, 0.25)

    def polar(m, p, t):
        X = np.column_stack([np.full_like(grid, m), np.full_like(grid, p), np.full_like(grid, t), grid,
                             np.full_like(grid, np.log10(re))])
        return model.predict(X)

    def objective(v):
        m, p, t = v
        cl, cd = polar(m, p, t)
        ok = grid <= grid[np.argmax(cl)] - stall_margin
        return -driving_force(cl[ok], cd[ok], awa_deg).max() if ok.any() else 1.0

    bounds = [DESIGN["m"], DESIGN["p"], (t_min, DESIGN["t"][1])]
    res = differential_evolution(objective, bounds, seed=seed, tol=1e-6, maxiter=150, polish=False)
    m, p, t = res.x
    cl, cd = polar(m, p, t)
    ok = grid <= grid[np.argmax(cl)] - stall_margin
    i = np.flatnonzero(ok)[np.argmax(driving_force(cl[ok], cd[ok], awa_deg))]
    return {"awa": awa_deg, "m": m, "p": p, "t": t, "alpha": float(grid[i]),
            "cl": float(cl[i]), "cd": float(cd[i]), "cx_predicted": float(-res.fun)}


def verify(opt, re=5e6):
    """Run XFOIL on the optimised section and angle: does the solver agree with the surrogate?"""
    pol = run_xfoil(opt["m"], opt["p"], opt["t"], re, a_max=20, step=0.25)
    row = pol[np.isclose(pol["alpha"], opt["alpha"])].iloc[0]
    conv = pol[pol["converged"]]
    return {"cl_xfoil": float(row.cl), "cd_xfoil": float(row.cd), "converged": bool(row.converged),
            "cx_xfoil": float(driving_force(row.cl, row.cd, opt["awa"])),
            "stall_alpha_xfoil": float(conv.loc[conv["cl"].idxmax(), "alpha"])}


def best_alpha(m, p, t, awa_deg, re=5e6, stall_margin=2.0):
    """Reference: XFOIL itself, best angle for a fixed section (used for the symmetric baseline)."""
    pol = run_xfoil(m, p, t, re, a_max=20, step=0.25)
    pol = pol[pol["converged"]]
    stall = pol.loc[pol["cl"].idxmax(), "alpha"]
    ok = pol[pol["alpha"] <= stall - stall_margin]
    cx = driving_force(ok["cl"], ok["cd"], awa_deg)
    r = ok.loc[cx.idxmax()]
    return {"alpha": float(r.alpha), "cl": float(r.cl), "cd": float(r.cd), "cx": float(cx.max())}
