"""
Number-of-bins ablation of the Mondrian conformal schedule.

The reference schedule (schedule_conformal.py) calibrates one margin per period (window / other hours) x decile of
the predicted level. This script re-implements that rule with a configurable number of quantile bins
(--n_bins_list, default 5 10 20) and compares each variant with the reference on ALL cells.
The "10 bins" row is a sanity check: it should reproduce the reference.

Verdict metric: deltas against the reference schedule with its margin scaled by ONE constant alpha
(metrics_common.iso_baselines): d_unmet = unmet of the variant minus unmet of the scaled reference at equal
over-provisioning; d_over = over of the variant minus over of the scaled reference at equal unmet.
Negative = the variant is better than simply scaling the margin.

Observed with 5 bins (h = 10 / 30 / 60 min): both deltas are positive. Fewer bins give coarser margins and the
busiest cells lose coverage (tier 97-100%: cover 0.84 -> 0.80). The filtered ramp penalty that used to be here was
removed: its lambda, tuned on a held-out split, came out 0 at 30 and 60 min and ~0 at 10 min.

Output (--out_dir, default final_metrics/adaptive_bins): adaptive_bins_metrics.csv (schema of metrics_common.py,
scopes global / tier) and one frontier figure per horizon. Units: unmet / over-provisioning are % of the total TRUE
activity of the group (activity index [a.u.], NOT energy); cover is a fraction (0-1); margins in [a.u.].

Usage (from the repository root; --horizons are 1-based, 1 = 10 min):
    python src/analysis/schedule_adaptive_bins.py --horizons 1 3 6 --day_start 9 --day_end 18 \
        --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
The per-run options of schedule_conformal.py are accepted (--day_eps, --night_eps, --cal_data, ...).
"""
import argparse
import os

import numpy as np

import metrics_common as mc
import schedule_conformal as sc
from conformal_eval import conf_q

METHOD = "adaptive_bins"
MIN_GROUP = 100        # minimum calibration pairs per (period, bin) group


# ----------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------
def parse_all(argv=None):
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--out_dir", default="final_metrics/adaptive_bins")
    ap.add_argument("--horizons", type=int, nargs="*", default=[1])
    ap.add_argument("--n_bins_list", type=int, nargs="*", default=[5, 10, 20],
                    help="numbers of predicted-level quantile bins to compare")
    mc.add_args(ap)
    extra, rest = ap.parse_known_args(argv)
    args = sc.parse_args(["--day_start", "9", "--day_end", "18"] + rest)
    for k, v in vars(extra).items():
        setattr(args, k, v)
    mc.set_lambdas(args)
    return args


# ----------------------------------------------------------------------
# Mondrian conformal margins with a configurable number of bins
# ----------------------------------------------------------------------
class BinCalib:
    """Margin per (period, quantile bin of the predicted level), from the calibration residuals truth - pred."""

    def __init__(self, cal, n_bins, args):
        self.args, self.n_bins = args, n_bins
        day = sc.in_window(cal["slot"], args.day_start, args.day_end)[:, None]
        res = cal["truth"] - cal["pred"]
        self.edges = np.quantile(cal["pred"], np.linspace(0, 1, n_bins + 1)[1:-1])
        lb = np.searchsorted(self.edges, cal["pred"], side="right")
        sel = [[(lb == b) & (day == p) for p in (True, False)] for b in range(n_bins)]   # [bin][window/other]
        small = min(s.sum() for pair in sel for s in pair)
        if small < MIN_GROUP:
            raise ValueError(f"{n_bins} bins: only {small} calibration pairs in the smallest group")
        self.q_day = np.array([conf_q(res[pair[0]], args.day_eps) for pair in sel])
        self.q_night = np.array([conf_q(res[pair[1]], args.night_eps) for pair in sel])

    def bound(self, d):
        a = self.args
        day = sc.in_window(d["slot"], a.day_start, a.day_end)[:, None]
        lb = np.searchsorted(self.edges, d["pred"], side="right")
        return d["pred"] + np.where(day, self.q_day[lb], self.q_night[lb])


# ----------------------------------------------------------------------
# Evaluation and report
# ----------------------------------------------------------------------
def uo(m):
    return f"{m['unmet_%']:5.2f}/{m['overprov_%']:5.2f}"


def print_report(res, info, names, deltas, args):
    glob = res["global"]["all cells"]
    print("\n== 1) GLOBAL, ALL CELLS, all hours (cover 0-1; unmet / over-prov. in % of total true activity) ==")
    print(f"{'approach':38s} {'cover':>6s} {'unmet%':>8s} {'over%':>8s} | {'d_unmet@iso-over':>16s} "
          f"{'d_over@iso-unmet':>16s}   [negative = better than scaling the reference]")
    for n in names:
        d = deltas.get(n)
        tail = f"{d[0]:+16.3f} {d[1]:+16.3f}" if d else ""
        print(f"{n:38s} {mc.fmt(glob[n]['all hours'])} | {tail}")

    print("\n== 2) BY STATIC TIER of cells (train-mean activity), all hours: unmet% / over% ==")
    print(f"{'tier':14s} {'act%':>5s} | " + " | ".join(f"{n[:11]:^11s}" for n in names))
    for name, t in info["tier"].items():
        print(f"{name:14s} {t['share_activity_%']:5.1f} | "
              + " | ".join(uo(res["tier"][name][n]["all hours"]) for n in names))


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------
def run_one(cal, ev, args, cols):
    truth, pred = ev["truth"], ev["pred"]
    up_ref = mc.ref_bounds(cal, ev, args)
    est = {mc.K_POINT: pred, mc.K_REF: up_ref}
    for n in args.n_bins_list:
        est[f"{n} bins"] = BinCalib(cal, n, args).bound(ev)

    # deltas vs the scaled reference (equal over / equal unmet); the reference itself must give ~0
    deltas, params = {}, {}
    for name in [mc.K_REF] + [f"{n} bins" for n in args.n_bins_list]:
        io, iu, _, _ = mc.iso_baselines(pred, up_ref, truth, est[name])
        m, mo, mu = (mc.compute(e, truth) for e in (est[name], io, iu))
        deltas[name] = (m["unmet_%"] - mo["unmet_%"], m["overprov_%"] - mu["overprov_%"])
        params[name] = f"d_unmet_at_iso_over={deltas[name][0]:+.3f};d_over_at_iso_unmet={deltas[name][1]:+.3f}"

    tier, tier_mean, tier_names, _ = mc.static_tiers(args, cols)
    scopes = {"global": {"all cells": None}, "tier": mc.tier_groups(tier, tier_names, truth.shape)}
    res = mc.evaluate_scopes(est, truth, mc.day_mask(ev, args), args, scopes)
    info = {"tier": mc.tier_info(tier, tier_mean, tier_names, truth)}
    print_report(res, info, list(est), deltas, args)

    h, window = args.horizon * 10, f"{args.day_start:g}-{args.day_end:g}"
    mc.figure_frontier(os.path.join(args.out_dir, f"adaptive_bins_h{h}_w{window}_frontier.png"), pred, up_ref,
                       truth, est, f"Bins ablation vs scaled reference - horizon {h} min")
    return mc.to_rows(METHOD, h, window, res, params)


def main():
    args = parse_all()
    if not (0 < args.day_eps < 1 and 0 < args.night_eps < 1):
        raise SystemExit("--day_eps and --night_eps must be in (0, 1)")
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for h in args.horizons:
        args.horizon = h
        cal, ev, cols = mc.load_horizon(args)
        print(f"\n##### Horizon {h * 10} min | cells {ev['truth'].shape[1]} | window {args.day_start:g}-{args.day_end:g} h")
        try:
            rows += run_one(cal, ev, args, cols)
        except ValueError as e:
            raise SystemExit(str(e))
    mc.write_csv(os.path.join(args.out_dir, "adaptive_bins_metrics.csv"), rows)


if __name__ == "__main__":
    main()