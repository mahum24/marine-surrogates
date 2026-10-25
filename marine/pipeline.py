"""Run everything end to end and write results/ (tables, summary.json, figures).

    python -m marine.pipeline            # hull part, plus the sail part if XFOIL data or XFOIL is available
    python -m marine.pipeline --hull     # hull part only
    python -m marine.pipeline --rerun-xfoil   # regenerate the XFOIL polars (needs the xfoil package)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from marine import DATA, RESULTS, figures, hull, sail

POLARS = DATA / "xfoil_polars.csv"
CONVERGENCE = DATA / "xfoil_panel_convergence.csv"
AWA = (30, 45, 60, 90, 120)


def step(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_hull():
    step("Hull: loading the Delft series and checking the data")
    d = hull.load()
    issues = hull.check(d)
    step("Hull: leave-one-hull-out test of every model")
    preds, sigma, dips = hull.leave_one_hull_out(d)
    scores = hull.score(d, preds, dips)
    cover = hull.interval_coverage(d, preds, sigma)
    step("Hull: optimising the hull form at Fn 0.35 with growing trust radius")
    X, y = d[hull.FEATURES].to_numpy(), np.log(d["rr"].to_numpy())
    boot = hull.HullBootstrap().fit(X, y, d["hull"].to_numpy())
    trust = [hull.optimise(d, radius=r, spread_scale=cover["scale_factor"], model=boot) for r in (0.5, 1.0, 1.5, 2.0, 2.5)]
    figures.hull_curves(d, preds, sigma, cover["scale_factor"])
    figures.hull_trust(trust)
    scores.to_csv(RESULTS / "hull_scores.csv", index=False)
    pd.DataFrame([{**t.pop("design"), **t} for t in [dict(x) for x in trust]]).to_csv(RESULTS / "hull_optimum.csv", index=False)
    best = scores.set_index("model")
    return {"data_checks": issues, "interval": cover,
            "median_error": best["median_pct_error"].round(4).to_dict(),
            "median_error_fn_0.3_plus": best["median_pct_error_fn_0.3_plus"].round(4).to_dict(),
            "hulls_with_dips": best["hulls_with_dips"].to_dict(),
            "optimum_radius_1": trust[1]}


def run_sail(rerun=False):
    if rerun or not POLARS.exists():
        step("Sail: running XFOIL on 300 sections")
        sail.build_dataset(300).to_csv(POLARS, index=False)
    if rerun or not CONVERGENCE.exists():
        step("Sail: panel convergence study")
        sail.panel_convergence().to_csv(CONVERGENCE, index=False)
    data, conv = pd.read_csv(POLARS), pd.read_csv(CONVERGENCE)

    step("Sail: checking XFOIL against thin aerofoil theory")
    theory = sail.thin_theory_check(data)
    theory.to_csv(RESULTS / "xfoil_vs_theory.csv", index=False)

    step("Sail: learning curve, network only vs hybrid")
    curve = sail.learning_curve(data)
    curve.to_csv(RESULTS / "sail_learning_curve.csv", index=False)

    step("Sail: final hybrid surrogate and section optimisation")
    d = sail.usable(data)
    train_secs, test_secs = sail.split_sections(d)
    tr, te = d[d["section"].isin(train_secs)], d[d["section"].isin(test_secs)]
    model = sail.FoilNet(hybrid=True).fit(sail.features(tr), tr["cl"].to_numpy(), tr["cd"].to_numpy())
    cl, cd = model.predict(sail.features(te))
    rows = []
    for awa in AWA:
        opt = sail.optimise_section(model, awa)
        check = sail.verify(opt)
        base = sail.best_alpha(0.0, 0.0, 0.15, awa)
        rows.append({**opt, **check, "cx_baseline": base["cx"], "alpha_baseline": base["alpha"]})
    opt = pd.DataFrame(rows)
    opt.to_csv(RESULTS / "sail_optimum.csv", index=False)
    figures.sail_learning(curve)
    figures.sail_polars(data, model, test_secs[:2])
    figures.sail_optimum(rows)

    full = curve[curve["sections"] == curve["sections"].max()].groupby("model")[["cl_mae", "cd_median_pct"]].mean()
    small = curve[curve["sections"] == curve["sections"].min()].groupby("model")[["cl_mae", "cd_median_pct"]].mean()
    fine = conv[conv["panels"] == 160]
    return {
        "sections": int(data["section"].nunique()), "points": int(len(data)),
        "converged_share": float(data["converged"].mean()), "usable_points": int(len(d)),
        "panel_160_max_change_pct": {"cl": float(fine["cl_change_pct"].max()), "cd": float(fine["cd_change_pct"].max())},
        "theory": {"median_slope_ratio": float(theory["slope_ratio"].median()),
                   "median_alpha_l0_gap_deg": float((theory["alpha_l0_xfoil"] - theory["alpha_l0_theory"]).abs().median())},
        "test": {"cl_mae": float(np.abs(cl - te["cl"]).mean()),
                 "cd_median_pct": float(100 * np.median(np.abs(cd - te["cd"]) / te["cd"]))},
        "learning_curve_all": full.round(4).to_dict(), "learning_curve_smallest": small.round(4).to_dict(),
        "optimum": opt[["awa", "m", "p", "t", "alpha", "cx_predicted", "cx_xfoil", "cx_baseline"]].round(4).to_dict("records"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hull", action="store_true", help="hull part only")
    ap.add_argument("--rerun-xfoil", action="store_true", help="regenerate the XFOIL data")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    summary = {"hull": run_hull()}
    if not args.hull:
        summary["sail"] = run_sail(args.rerun_xfoil)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    step("Done: results/summary.json")


if __name__ == "__main__":
    main()
