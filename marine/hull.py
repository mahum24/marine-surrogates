"""Residuary resistance surrogates for the Delft Systematic Yacht Hull Series.

308 towing-tank measurements: 22 hull forms, each towed at 14 Froude numbers.
Residuary resistance (wave-making plus form effects) is given per unit weight
of displacement, x 1000.

Every model is tested by leaving one whole hull out at a time, so the score
says how well it predicts a hull form it has never seen, which is what a
design tool has to do.

Physics used:
  - resistance is positive and spans four orders of magnitude, so models work on log(Rr);
  - resistance must rise with speed over this range (no hump is reached by Fn 0.45);
    the boosted trees get this as a hard monotonic constraint, and every model is
    checked for dips on a dense speed sweep of each held-out hull.
"""
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from marine import DATA

FORM = ["lcb", "cp", "l_disp", "b_t", "l_b"]          # hull form parameters
FEATURES = FORM + ["fn"]
LABELS = {"lcb": "LCB position", "cp": "Prismatic coefficient", "l_disp": "Length-displacement ratio",
          "b_t": "Beam-draught ratio", "l_b": "Length-beam ratio", "fn": "Froude number"}


def load(path=DATA / "raw" / "yacht_hydrodynamics.data") -> pd.DataFrame:
    d = pd.read_csv(path, sep=r"\s+", names=FEATURES + ["rr"]).dropna()
    d["hull"] = d.groupby(FORM, sort=False).ngroup()
    return d.reset_index(drop=True)


def check(d: pd.DataFrame) -> list[str]:
    """Data-quality checks; returns findings in words."""
    out = []
    if d[FEATURES + ["rr"]].isna().any().any():
        out.append("Missing values found.")
    if (d["rr"] <= 0).any():
        out.append(f"{(d['rr'] <= 0).sum()} non-positive resistance values.")
    counts = d.groupby("hull").size()
    if counts.nunique() != 1:
        out.append(f"Hulls have different numbers of speeds: {counts.value_counts().to_dict()}.")
    curves = d.pivot(index="hull", columns="fn", values="rr")
    falling = [h for h, r in curves.iterrows() if (np.diff(r.to_numpy()) < 0).any()]
    if falling:
        out.append(f"Resistance falls with speed for hulls {falling}.")
    dup = curves[curves.duplicated(keep=False)]
    for _, grp in dup.groupby(list(dup.columns)):
        out.append(f"Hulls {list(grp.index)} have different forms but identical resistance curves "
                   "(likely a copying error in the source table).")
    return out or ["No issues found."]


# ---------------------------------------------------------------- models (all predict log Rr)

class SpeedOnly:
    """Baseline: the average log resistance of the training hulls at each speed."""
    def fit(self, X, y):
        self.table = pd.Series(y).groupby(np.round(X[:, -1], 3)).mean()
        return self

    def predict(self, X):
        return np.interp(X[:, -1], self.table.index, self.table.to_numpy())


class PerSpeedRegression:
    """Delft-style regression: a separate linear model in the hull form parameters at each speed.

    This mirrors how the series is usually summarised in naval architecture, where
    resistance is fitted as a polynomial in the form parameters with coefficients
    that change with Froude number. Between tested speeds the prediction is
    interpolated.
    """
    def __init__(self, alpha=None):
        self.alpha = alpha

    def fit(self, X, y):
        self.fns = np.unique(np.round(X[:, -1], 3))
        reg = (lambda: Ridge(self.alpha)) if self.alpha else (lambda: RidgeCV(alphas=np.logspace(-3, 3, 25)))
        self.models = {f: make_pipeline(StandardScaler(), reg()).fit(X[np.isclose(X[:, -1], f), :-1],
                                                                     y[np.isclose(X[:, -1], f)]) for f in self.fns}
        return self

    def coefficients(self):
        """Weights and intercept per speed in the original units (scaler folded in)."""
        W, b = [], []
        for f in self.fns:
            sc, reg = self.models[f][0], self.models[f][-1]
            W.append(reg.coef_ / sc.scale_)
            b.append(reg.intercept_ - (reg.coef_ * sc.mean_ / sc.scale_).sum())
        return np.array(W), np.array(b)

    def predict(self, X):
        W, b = self.coefficients()
        return _interp_speed(X, self.fns, X[:, :-1] @ W.T + b)


def _interp_speed(X, fns, at):
    """Linear interpolation between the tested speeds; `at` holds predictions at each tested speed."""
    i = np.clip(np.searchsorted(fns, X[:, -1]) - 1, 0, len(fns) - 2)
    t = (X[:, -1] - fns[i]) / (fns[i + 1] - fns[i])
    rows = np.arange(len(X))
    return at[..., rows, i] * (1 - t) + at[..., rows, i + 1] * t


def boosted(monotonic=True):
    cst = [0] * len(FORM) + [1 if monotonic else 0]
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, min_samples_leaf=5,
                                         monotonic_cst=cst, random_state=0)


class HullBootstrap:
    """Uncertainty from resampling whole hulls: refit the per-speed regression on many
    resampled sets of hulls. Where the tested hulls disagree or are sparse, the fits
    spread out."""
    def __init__(self, n=200, alpha=1.0, seed=0):
        self.n, self.alpha, self.seed = n, alpha, seed

    def fit(self, X, y, hulls):
        rng = np.random.default_rng(self.seed)
        ids = np.unique(hulls)
        self.members = []
        for _ in range(self.n):
            pick = rng.choice(ids, len(ids), replace=True)
            rows = np.concatenate([np.flatnonzero(hulls == h) for h in pick])
            self.members.append(PerSpeedRegression(self.alpha).fit(X[rows], y[rows]))
        self.fns = self.members[0].fns
        self.W, self.b = map(np.stack, zip(*(m.coefficients() for m in self.members)))
        return self

    def predict_dist(self, X):
        at = np.einsum("nf,mkf->mnk", X[:, :-1], self.W) + self.b[:, None, :]
        p = _interp_speed(X, self.fns, at)
        return p.mean(0), p.std(0)

    def predict(self, X):
        return self.predict_dist(X)[0]


MODELS = {
    "Speed only (baseline)": SpeedOnly,
    "Delft-style regression": PerSpeedRegression,
    "Boosted trees": lambda: boosted(monotonic=False),
    "Boosted trees + monotonic speed": boosted,
}


# ---------------------------------------------------------------- evaluation

def fn_grid(n=66):
    return np.linspace(0.125, 0.45, n)


def leave_one_hull_out(d: pd.DataFrame, models=MODELS):
    """Predict every hull from models trained on the other 21.

    Returns point predictions (in Rr units), the bootstrap spread (log units), and
    how many held-out hulls get a physically impossible dip in resistance somewhere
    on a dense speed sweep.
    """
    X, y, hulls = d[FEATURES].to_numpy(), np.log(d["rr"].to_numpy()), d["hull"].to_numpy()
    preds = pd.DataFrame(index=d.index, columns=list(models), dtype=float)
    sigma = pd.Series(index=d.index, dtype=float)
    dips = pd.Series(0, index=list(models), name="hulls_with_dips")
    for h in np.unique(hulls):
        test = hulls == h
        sweep = np.column_stack([np.tile(X[test][0, :-1], (len(fn_grid()), 1)), fn_grid()])
        for name, make in models.items():
            m = make().fit(X[~test], y[~test])
            preds.loc[test, name] = m.predict(X[test])
            dips[name] += int((np.diff(m.predict(sweep)) < -1e-6).any())
        sigma[test] = HullBootstrap(n=100).fit(X[~test], y[~test], hulls[~test]).predict_dist(X[test])[1]
    return np.exp(preds), sigma, dips


def score(d: pd.DataFrame, preds: pd.DataFrame, dips: pd.Series) -> pd.DataFrame:
    y = d["rr"].to_numpy()
    fast = (d["fn"] >= 0.3).to_numpy()
    rows = []
    for name in preds:
        p = preds[name].to_numpy()
        rows.append({
            "model": name,
            "r2": 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
            "mae": np.abs(p - y).mean(),
            "median_pct_error": np.median(np.abs(p - y) / y),
            "median_pct_error_fn_0.3_plus": np.median(np.abs(p[fast] - y[fast]) / y[fast]),
            "hulls_with_dips": int(dips[name]),
        })
    return pd.DataFrame(rows)


def interval_coverage(d, preds, sigma, model="Delft-style regression", level=0.8):
    """Scale the bootstrap spread on the other hulls, then check coverage on each hull (cross-conformal)."""
    y = np.log(d["rr"].to_numpy())
    z = np.abs(y - np.log(preds[model].to_numpy())) / sigma.to_numpy()
    hit, scale = [], []
    for h in d["hull"].unique():
        test = (d["hull"] == h).to_numpy()
        k = np.quantile(z[~test], level)
        hit.extend(z[test] <= k)
        scale.append(k)
    return {"raw_coverage": float(np.mean(z <= 1.2816)),       # +/-1.28 spread, before scaling
            "calibrated_coverage": float(np.mean(hit)), "scale_factor": float(np.median(scale))}


# ---------------------------------------------------------------- design optimisation

def optimise(d: pd.DataFrame, fn=0.35, l_disp=4.78, caution=1.0, radius=None, spread_scale=1.0,
             model: HullBootstrap | None = None, seed=0):
    """Find the hull form with the lowest predicted resistance at a design speed.

    Length-displacement ratio is fixed (same length and displacement as the parent
    hull), so the optimiser cannot simply make the boat slimmer. The objective is
    the upper end of the prediction (mean + caution x spread): where the model is
    unsure, usually where several parameters change together in ways the series
    never tested, the design is penalised instead of trusted. `spread_scale` is the
    calibration factor from `interval_coverage`, so the reported range is honest.

    The bootstrap only measures how much the tested hulls disagree; it cannot see
    interactions the series never tested. So `radius` adds a trust region: the
    design must stay within that many standard deviations of a tested hull form.
    """
    X, y = d[FEATURES].to_numpy(), np.log(d["rr"].to_numpy())
    model = model or HullBootstrap(seed=seed).fit(X, y, d["hull"].to_numpy())
    free = ["lcb", "cp", "b_t", "l_b"]
    bounds = [(d[c].min(), d[c].max()) for c in free]

    def full(v):
        row = dict(zip(free, v), l_disp=l_disp, fn=fn)
        return np.array([[row[c] for c in FEATURES]])

    forms = d.drop_duplicates("hull").set_index("hull")[FORM]
    scale = d[FORM].std()

    def distance(v):
        design = pd.Series(dict(zip(free, v), l_disp=l_disp))[FORM]
        return ((forms - design) / scale).pow(2).sum(1).pow(0.5)

    def objective(v):
        mean, sd = model.predict_dist(full(v))
        over = 0.0 if radius is None else max(0.0, distance(v).min() - radius)
        return float(mean[0] + caution * sd[0] + 10 * over)

    res = differential_evolution(objective, bounds, seed=seed, tol=1e-8, maxiter=200, polish=True)
    mean, sd = model.predict_dist(full(res.x))
    sd = sd * spread_scale
    tested = d[(d["fn"].round(3) == fn) & (d["l_disp"].sub(l_disp).abs() < 0.05)]
    design = pd.Series(dict(zip(free, res.x), l_disp=l_disp))[FORM]
    gap = distance(res.x)
    return {
        "caution": caution, "radius": radius, "fn": fn, "design": design.round(3).to_dict(),
        "predicted_rr": float(np.exp(mean[0])),
        "range_80": [float(np.exp(mean[0] - 1.2816 * sd[0])), float(np.exp(mean[0] + 1.2816 * sd[0]))],
        "spread_log": float(sd[0]),
        "parent_rr": float(tested.loc[tested["hull"] == 0, "rr"].iloc[0]),
        "best_tested_rr": float(tested["rr"].min()), "best_tested_hull": int(tested.loc[tested["rr"].idxmin(), "hull"]),
        "nearest_tested_hull": int(gap.idxmin()), "distance_sd": float(gap.min()),
    }
