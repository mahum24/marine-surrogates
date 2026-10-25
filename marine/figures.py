"""Figures for the README. One axis per chart, thin marks, a legend whenever there is more than one series."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from marine import RESULTS, hull, sail

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2, "figure.dpi": 150,
})


def _save(fig, name):
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / name, bbox_inches="tight")
    plt.close(fig)


def hull_curves(d, preds, sigma, scale, hulls=(0, 15, 21)):
    """Held-out hulls: towing-tank points against the predictions made without them."""
    fig, axes = plt.subplots(1, len(hulls), figsize=(4 * len(hulls), 3.4), sharey=True)
    for ax, h in zip(axes, hulls):
        k = (d["hull"] == h).to_numpy()
        fn, rr = d.loc[k, "fn"], d.loc[k, "rr"]
        mid = preds.loc[k, "Delft-style regression"].to_numpy()
        band = 1.2816 * scale * sigma[k].to_numpy()
        ax.fill_between(fn, mid * np.exp(-band), mid * np.exp(band), color=BLUE, alpha=0.15, lw=0,
                        label="80% range")
        ax.plot(fn, mid, color=BLUE, label="Delft-style regression")
        ax.plot(fn, preds.loc[k, "Boosted trees + monotonic speed"], color=ORANGE, label="Boosted trees (monotonic)")
        ax.plot(fn, rr, "o", ms=5, mfc="white", mec=INK, mew=1.2, label="Towing tank")
        ax.set_yscale("log")
        ax.set_title(f"Hull {h} (not used in training)", fontsize=10, color=INK)
        ax.set_xlabel("Froude number")
    axes[0].set_ylabel("Rr per unit weight (x1000)")
    axes[0].legend(fontsize=8, loc="upper left")
    _save(fig, "hull_held_out.png")


def hull_trust(rows):
    """Predicted resistance of the optimised hull against how far it may move from tested hulls."""
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    r = [x["radius"] for x in rows]
    mid = np.array([x["predicted_rr"] for x in rows])
    lo, hi = np.array([x["range_80"] for x in rows]).T
    ax.fill_between(r, lo, hi, color=BLUE, alpha=0.15, lw=0, label="80% range")
    ax.plot(r, mid, "o-", color=BLUE, ms=6, label="Optimised hull (predicted)")
    ax.axhline(rows[0]["parent_rr"], color=MUTED, lw=1.2, ls="--")
    ax.text(r[-1], rows[0]["parent_rr"], " parent hull (measured)", va="bottom", ha="right", color=MUTED, fontsize=8)
    ax.set_xlabel("Trust radius (standard deviations from a tested hull)")
    ax.set_ylabel("Residuary resistance at Fn 0.35")
    ax.legend(fontsize=8, loc="lower left")
    _save(fig, "hull_trust_region.png")


def sail_learning(curve):
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for name, colour in [("Network only", ORANGE), ("Hybrid (theory + network)", BLUE)]:
        g = curve[curve["model"] == name].groupby("sections")["cl_mae"]
        ax.plot(g.mean().index, g.mean(), "o-", color=colour, ms=6, label=name)
        ax.fill_between(g.min().index, g.min(), g.max(), color=colour, alpha=0.12, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("Sections used for training")
    ax.set_ylabel("Lift error on unseen sections (MAE)")
    ax.legend(fontsize=8)
    _save(fig, "sail_learning_curve.png")


def sail_polars(data, model, sections):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for colour, sec in zip([BLUE, ORANGE], sections):
        g = sail.usable(data[data["section"] == sec])
        r = g.iloc[0]
        label = f"{sail.naca_name(r.m, r.p, r.t)}-like, Re {r.re / 1e6:.1f}M"
        cl, cd = model.predict(sail.features(g))
        axes[0].plot(g["alpha"], g["cl"], "o", ms=4, mfc="white", mec=colour, mew=1.2)
        axes[0].plot(g["alpha"], cl, color=colour, label=label)
        axes[1].plot(g["cd"], g["cl"], "o", ms=4, mfc="white", mec=colour, mew=1.2)
        axes[1].plot(cd, cl, color=colour)
    axes[0].set_xlabel("Angle of attack (deg)")
    axes[0].set_ylabel("Lift coefficient")
    axes[1].set_xlabel("Drag coefficient")
    axes[1].set_ylabel("Lift coefficient")
    axes[0].legend(fontsize=8, title="Lines: surrogate, points: XFOIL", title_fontsize=8)
    _save(fig, "sail_polars.png")


def sail_optimum(rows):
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    awa = [r["awa"] for r in rows]
    ax.plot(awa, [r["cx_baseline"] for r in rows], "o-", color=MUTED, ms=6, label="NACA 0015, best angle (XFOIL)")
    ax.plot(awa, [r["cx_predicted"] for r in rows], "o-", color=BLUE, ms=6, label="Optimised section (surrogate)")
    ax.plot(awa, [r["cx_xfoil"] for r in rows], "s", ms=7, mfc="white", mec=ORANGE, mew=1.5,
            label="Optimised section (XFOIL check)")
    ax.set_xlabel("Apparent wind angle off the bow (deg)")
    ax.set_ylabel("Driving force coefficient")
    ax.legend(fontsize=8)
    _save(fig, "sail_optimum.png")
