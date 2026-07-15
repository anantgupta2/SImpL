#!/usr/bin/env python3
"""Aggregate finals eval CSVs (cot_accuracy curves) into the final LSAT/RACE/QUAIL table.

Each (dataset, method) has 3 seeds, each a curve of cot_accuracy vs step (dev + test).
The dev plateau is flat to within seed-noise, so single-step argmax is unstable. We report
a ROBUST dev-selected test number = mean over the last-N steps (default last 100), plus each
method's TEST-SET MAX (per-seed peak test, averaged over seeds) as an oracle upper bound.

Columns: dev(band) = mean dev over the band; TEST(band) = mean test over the same band
(the honest dev-selected estimate); Δ = TEST vs cot16; test-max = mean of per-seed test peaks.

Usage: python scripts/eval/aggregate_finals.py [--band 100] [--step-cap 300]
"""
import argparse, csv, glob, os, re
from statistics import mean, pstdev

DEV = os.environ.get("AGG_DEV_DIR", "evaluations/finals_dev")
TEST = os.environ.get("AGG_TEST_DIR", "evaluations/finals_test")
DATASETS = ["lsat", "race", "quail"]
METHOD_ORDER = ["cotn16", "nbmarg", "flatsimpl", "flatsimplv3", "nbmargurs1",
                "flatsplit-u4c12", "flatsplit-uevery4", "flatsplitv3-u4c12",
                "flatsimpl-long", "flatsplit-u4c12-long",
                "cotn16-long", "flatsimplv3-long", "flatsplitv3-u4c12-long",
                "cotn16-a128",
                "flatsimpl-a128",
                "flatsplit-u4c12-a128", "flatsplit-u12c4", "flatsplit-u16c0",
                "flatsplitv3-u12c4", "flatsplitv3-u16c0",
                "cotn16-short", "flatsplitv3-u4c12-short"]
SEEDS = ["s123", "s234", "s345"]


def load_curve(path):
    d = {}
    if not os.path.exists(path):
        return d
    with open(path) as f:
        for r in csv.DictReader(f):
            m = re.search(r"(\d+)", r.get("step", ""))
            if not m:
                continue
            try:
                d[int(m.group(1))] = float(r["cot_accuracy"]) * 100.0
            except (KeyError, ValueError):
                pass
    return d


def seed_curves(subdir, ds, method):
    out = {}
    for s in SEEDS:
        c = load_curve(os.path.join(subdir, f"{ds}_{method}_{s}.csv"))
        if c:
            out[s] = c
    return out


def rolling_peak_idx(steps, curve_avg, win):
    """Index of the `win`-consecutive-step window with the highest mean of curve_avg."""
    return max(range(len(steps) - win + 1),
               key=lambda i: mean(curve_avg[steps[j]] for j in range(i, i + win)))


def aggregate(ds, method, win, cap):
    """Convergence = rolling-`win` peak on the SEED-AVERAGED dev curve; report the seed-averaged
    test at that window's CENTER step (the dev-selected convergence step). test-max = rolling-`win`
    peak of the seed-averaged test curve (the method's converged ceiling)."""
    dev = seed_curves(DEV, ds, method)
    test = seed_curves(TEST, ds, method)
    seeds = sorted(set(dev) & set(test))
    if not seeds:
        return None
    common = None
    for s in seeds:
        st = {k for k in dev[s] if k <= cap} & {k for k in test[s] if k <= cap}
        common = st if common is None else (common & st)
    common = sorted(common or [])
    if len(common) < win:
        return None
    dev_avg = {st: mean(dev[s][st] for s in seeds) for st in common}
    test_avg = {st: mean(test[s][st] for s in seeds) for st in common}
    i = rolling_peak_idx(common, dev_avg, win)
    w = common[i:i + win]
    center = w[len(w) // 2]
    # per-seed test at the center step -> across-seed std (honest error bar)
    test_ps = [test[s][center] for s in seeds]
    # PER-SEED convergence: each seed picks its OWN rolling-`win` dev peak, take that seed's test
    # at its own center step, then average across seeds.
    ps_vals, ps_steps = [], []
    for s in seeds:
        js = rolling_peak_idx(common, dev[s], win)
        cs = common[js:js + win][win // 2]
        ps_vals.append(test[s][cs]); ps_steps.append(cs)
    # test ceiling: best rolling-`win` mean of the seed-avg test curve
    def r3peak(cv):
        return max(mean(cv[common[j]] for j in range(k, k + win))
                   for k in range(len(common) - win + 1))
    tmax = r3peak(test_avg)
    # PER-SEED test ceiling: each seed's own LITERAL single-step max over ITS OWN test curve
    # (all its evaluated steps <= cap, NOT restricted to the across-seed common set), averaged.
    ps_tmax = [max(v for st, v in test[s].items() if st <= cap) for s in seeds]
    # ==== CANONICAL (2026-07-12): single-step dev-argmax = the checkpoint you'd actually deploy.
    # Per-seed: each run picks its OWN best-dev step and reports test there (realistic); averaged.
    # Seed-avg: pick the step with the best across-seed-mean dev, report seed-avg test there.
    da_savg_step = max(common, key=lambda k: dev_avg[k])
    devargmax_savg = test_avg[da_savg_step]
    da_ps = [test[s][max(common, key=lambda k: dev[s][k])] for s in seeds]
    return {
        "seeds": seeds, "n_common": len(common), "window": (w[0], w[-1]), "center": center,
        "dev_sel": dev_avg[center],
        "test_mean": test_avg[center], "test_std": pstdev(test_ps), "test_ps": test_ps,
        "perseed_mean": mean(ps_vals), "perseed_std": pstdev(ps_vals),
        "perseed_steps": ps_steps, "perseed_vals": ps_vals,
        "tmax": tmax, "ps_tmax_mean": mean(ps_tmax), "ps_tmax_std": pstdev(ps_tmax),
        "devargmax_savg": devargmax_savg, "da_savg_step": da_savg_step,
        "devargmax_ps_mean": mean(da_ps), "devargmax_ps_std": pstdev(da_ps),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", type=int, default=3, help="convergence window (consecutive ckpts)")
    ap.add_argument("--step-cap", type=int, default=300)
    args = ap.parse_args()

    for ds in DATASETS:
        methods = [m for m in METHOD_ORDER
                   if glob.glob(os.path.join(TEST, f"{ds}_{m}_s*.csv"))]
        if not methods:
            continue
        results = [(m, aggregate(ds, m, args.win, args.step_cap)) for m in methods]
        results = [(m, r) for m, r in results if r]
        # baseline = cot16 control (fall back to cotn16-long when the plain one isn't present)
        base_row = next((r for m, r in results if m == "cotn16"), None) \
            or next((r for m, r in results if m == "cotn16-long"), None)
        base = base_row["devargmax_ps_mean"] if base_row else None      # CANONICAL baseline
        base_sa = base_row["devargmax_savg"] if base_row else None
        print(f"\n{'='*100}\n{ds.upper()}   test cot_acc%  (step-cap {args.step_cap}).  "
              f"CANONICAL = single-step dev-argmax (the deployed checkpoint)")
        print(f"{'method':<15} {'seeds':>5}  {'DEVARGMAX(ps)':>14} {'Δ':>6}  "
              f"{'devargmax(sa)':>13} {'Δ':>6}  {'seed-tmax':>11}")
        for m, r in results:
            d = "" if base is None else f"{r['devargmax_ps_mean']-base:+.1f}"
            dsa = "" if base_sa is None else f"{r['devargmax_savg']-base_sa:+.1f}"
            star = " *" if (base is not None and r['devargmax_ps_mean']-base >= 0.3) else ""
            print(f"{m:<15} {len(r['seeds']):>5}  {r['devargmax_ps_mean']:>7.1f}±{r['devargmax_ps_std']:<4.1f} {d:>6}  "
                  f"{r['devargmax_savg']:>13.1f} {dsa:>6}  {r['ps_tmax_mean']:>8.1f}{star}")


if __name__ == "__main__":
    main()
