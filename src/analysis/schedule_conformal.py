"""
Time-of-day schedule for the conformal upper bound (conformal by level):
    coverage 90 % (eps 0.10) between 08:00 and 21:00, coverage 75 % (eps 0.25) in the other hours.

How it works (and why it is built like this)
  The window is assigned by the clock time of the step being FORECAST. The margin is calibrated
  separately per period and per predicted-level decile (a "Mondrian" conformal split), using only the
  val residuals of that period:
        margin[period, decile]   ->   bound = prediction + margin[period(t), decile(prediction)]
  So 90 % is guaranteed (marginally) INSIDE the daytime window and 75 % inside the night one.
  Calibrating on the whole day and only switching eps by hour would NOT do that, because the
  residuals are not the same at all hours: that variant is printed too, for comparison.

Compared on the test set (ORIGINAL units):
    fixed 90 %                         always 90 %
    fixed 75 %                         always 75 %
    scheduled, calibrated per period   the proposal
    scheduled, calibrated on all day   same switch, margin from the whole-day residuals

Saved:
    schedule_summary.png   coverage and margin per clock hour + the margin table by decile
    schedule_busiest.png   N busiest cells: true (blue), prediction (orange), scheduled bound (red);
                           grey = hours with the lower coverage
    schedule_margins.json  decile edges + margin tables, ready to use at run time:
                           margin = table[period][decile_of(prediction)]

Usage:
    python schedule_conformal.py --horizon 0 \
        --cal_data datasets/MILAN_val --cal_ckpts "checkpoints_eval/MILAN_val/seed*/*"
    python schedule_conformal.py --day_eps 0.10 --night_eps 0.25 --day_start 8 --day_end 21

--horizon is 0-based (0 = 10 min ahead, 2 = 30 min). Needs only analyse_result.py and conformal_eval.py.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyse_result import STEP_H, STEPS_PER_DAY, clock_axis
from conformal_eval import N_BINS, conf_q, load_split, subset

C_TRUE, C_PRED, C_UP = "tab:blue", "tab:orange", "tab:red"
K_SCHED = "scheduled, calibrated per period"
K_SCHED_ALL = "scheduled, calibrated on all day"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default="checkpoints/STID_MILAN/seed*/*", help="runs of the EVALUATION split (test)")
    ap.add_argument("--data", default="datasets/MILAN", help="dataset of the EVALUATION split (test)")
    ap.add_argument("--cal_ckpts", default=None, help="runs of the CALIBRATION split (e.g. val view)")
    ap.add_argument("--cal_data", default=None, help="dataset view of the calibration split")
    ap.add_argument("--input_len", type=int, default=144)
    ap.add_argument("--output_len", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=0, help="0 = first step = 10 min ahead")
    ap.add_argument("--cal_frac", type=float, default=0.5, help="only used without --cal_data")
    ap.add_argument("--exclude", type=int, nargs="*", default=[])
    ap.add_argument("--csv", default="milan_raw.csv")
    ap.add_argument("--day_eps", type=float, default=0.10, help="violation rate inside the window (0.10 = 90%%)")
    ap.add_argument("--night_eps", type=float, default=0.25, help="violation rate outside it (0.25 = 75%%)")
    ap.add_argument("--day_start", type=float, default=8.0, help="start of the high-coverage window, hour of day")
    ap.add_argument("--day_end", type=float, default=21.0, help="end of the window (exclusive), hour of day")
    ap.add_argument("--n_busy", type=int, default=5, help="cells in schedule_busiest.png")
    ap.add_argument("--start", type=int, default=0, help="first time step of the busiest plot")
    ap.add_argument("--length", type=int, default=None, help="number of time steps (default: all)")
    ap.add_argument("--out", default="schedule_summary.png")
    ap.add_argument("--out_busy", default="schedule_busiest.png")
    ap.add_argument("--out_json", default="schedule_margins.json")
    return ap.parse_args()


def in_window(slot, start_h, end_h):
    """True where the clock time of the slot is inside [start_h, end_h) (wraps over midnight if start > end)."""
    h = np.asarray(slot) * 24 / STEPS_PER_DAY
    if start_h <= end_h:
        return (h >= start_h) & (h < end_h)
    return (h >= start_h) | (h < end_h)


class Calib:
    """Level bins and calibration residuals; margin tables per eps, optionally per period."""

    def __init__(self, cal, ev, args):
        self.edges = np.quantile(cal["pred"], np.linspace(0, 1, N_BINS + 1)[1:-1])
        self.lb_c = np.searchsorted(self.edges, cal["pred"], side="right")
        self.lb_e = np.searchsorted(self.edges, ev["pred"], side="right")
        self.res_c = cal["truth"] - cal["pred"]
        self.day_c = in_window(cal["slot"], args.day_start, args.day_end)[:, None]
        self.day_e = in_window(ev["slot"], args.day_start, args.day_end)[:, None]

    def table(self, eps, period=None):
        """Margin per decile. period None = all calibration windows, True = window hours, False = the rest."""
        qb = []
        for b in range(N_BINS):
            sel = self.lb_c == b
            if period is not None:
                sel = sel & (self.day_c == period)
            r = self.res_c[sel]
            if r.size == 0:
                raise SystemExit("a calibration group is empty: check --day_start / --day_end")
            qb.append(conf_q(r, eps))
        return np.array(qb)


def build_bounds(c, ev, args):
    pred, d = ev["pred"], c.day_e
    q_day, q_night = c.table(args.day_eps, True), c.table(args.night_eps, False)
    qa_day, qa_night = c.table(args.day_eps), c.table(args.night_eps)
    k_hi, k_lo = f"fixed {1 - args.day_eps:.0%} (always)", f"fixed {1 - args.night_eps:.0%} (always)"
    bounds = {
        k_hi: pred + qa_day[c.lb_e],
        k_lo: pred + qa_night[c.lb_e],
        K_SCHED: pred + np.where(d, q_day[c.lb_e], q_night[c.lb_e]),
        K_SCHED_ALL: pred + np.where(d, qa_day[c.lb_e], qa_night[c.lb_e]),
    }
    return bounds, (q_day, q_night), (k_hi, k_lo)


def stats(up, ev, rows=None):
    t, p, u = ev["truth"], ev["pred"], up
    if rows is not None:
        t, p, u = t[rows], p[rows], u[rows]
    return {"cover": (t <= u).mean(), "margin": (u - p).mean(),
            "marg_pct": 100 * (u - p).mean() / t.mean(), "wape_b": np.abs(u - t).sum() / t.sum()}


def to_full(v, idx, n):
    full = np.full(n, np.nan)
    full[idx] = v
    return full


# ----------------------------------------------------------------------
def report(c, ev, bounds, keys, args):
    day, night = c.day_e[:, 0], ~c.day_e[:, 0]
    lab_d = f"{args.day_start:g}-{args.day_end:g} h (target {1 - args.day_eps:.0%})"
    lab_n = f"other hours (target {1 - args.night_eps:.0%})"
    print(f"\n{'approach':36s} | {'ALL HOURS':^22s} | {lab_d:^28s} | {lab_n:^28s}")
    print(f"{'':36s} | {'cover':>6s} {'marg':>6s} {'marg%':>6s} | {'cover':>6s} {'marg':>7s} {'marg%':>6s}    "
          f"| {'cover':>6s} {'marg':>7s} {'marg%':>6s}")
    res = {}
    for k, up in bounds.items():
        a, dd, nn = stats(up, ev), stats(up, ev, day), stats(up, ev, night)
        res[k] = (a, dd, nn)
        tag = "  <- proposal" if k == K_SCHED else ""
        print(f"{k:36s} | {a['cover']:6.3f} {a['margin']:6.2f} {a['marg_pct']:5.1f}% | "
              f"{dd['cover']:6.3f} {dd['margin']:7.2f} {dd['marg_pct']:5.1f}%    | "
              f"{nn['cover']:6.3f} {nn['margin']:7.2f} {nn['marg_pct']:5.1f}%{tag}")
    hi = res[keys[0]]
    sc = res[K_SCHED]
    print(f"\nscheduled vs {keys[0]}: mean margin {100 * (sc[0]['margin'] / hi[0]['margin'] - 1):+.1f}% overall, "
          f"{100 * (sc[2]['margin'] / hi[2]['margin'] - 1):+.1f}% in the other hours "
          f"(coverage there {sc[2]['cover']:.3f} instead of {hi[2]['cover']:.3f})")
    print("How to read it:\n"
          "  per-period calibration should give cover ~ target INSIDE each period;\n"
          "  'calibrated on all day' shows what happens if only eps is switched: usually the hours with\n"
          "  larger errors fall below target and the quiet hours end up above it.")
    return res


def hourly(up, ev):
    hr = ev["slot"] // (STEPS_PER_DAY // 24)
    cov = np.array([(ev["truth"][hr == h] <= up[hr == h]).mean() for h in range(24)])
    mar = np.array([(up[hr == h] - ev["pred"][hr == h]).mean() for h in range(24)])
    return cov, mar


def figure_summary(c, ev, bounds, tables, keys, args):
    colors = {keys[0]: "tab:green", keys[1]: "tab:purple", K_SCHED: C_UP, K_SCHED_ALL: "tab:gray"}
    styles = {keys[0]: "-", keys[1]: "-", K_SCHED: "-", K_SCHED_ALL: "--"}
    hrs = np.arange(24)
    target = np.where(in_window(hrs * (STEPS_PER_DAY // 24), args.day_start, args.day_end),
                      1 - args.day_eps, 1 - args.night_eps)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    h_min = None

    ax = axes[0]
    ax.step(np.append(hrs, 24), np.append(target, target[-1]), where="post", color="k", ls=":", lw=1.5,
            label="target of the schedule")
    for k, up in bounds.items():
        ax.plot(hrs + 0.5, hourly(up, ev)[0], "o" + styles[k], color=colors[k], ms=4, label=k)
    ax.set_xlabel("clock hour of the forecast target")
    ax.set_ylabel("coverage on test")
    ax.set_title("(a) coverage per clock hour", fontsize=10)
    ax.set_xticks(range(0, 25, 3))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    for k, up in bounds.items():
        ax.plot(hrs + 0.5, hourly(up, ev)[1], "o" + styles[k], color=colors[k], ms=4, label=k)
    ax.set_xlabel("clock hour of the forecast target")
    ax.set_ylabel("mean margin (traffic units)")
    ax.set_title("(b) margin per clock hour (smaller = more energy saved)", fontsize=10)
    ax.set_xticks(range(0, 25, 3))
    ax.grid(alpha=0.3)

    ax = axes[2]
    q_day, q_night = tables
    lvl = np.array([ev["pred"][c.lb_e == b].mean() for b in range(N_BINS)])
    dec = np.arange(N_BINS)
    ax.plot(dec, q_day, "o-", color="tab:green", label=f"{args.day_start:g}-{args.day_end:g} h, {1 - args.day_eps:.0%}")
    ax.plot(dec, q_night, "o-", color="tab:purple", label=f"other hours, {1 - args.night_eps:.0%}")
    ax.set_xticks(dec)
    ax.set_xticklabels([f"{v:.0f}" for v in lvl], fontsize=8)
    ax.set_xlabel("predicted level (mean prediction of the decile)")
    ax.set_ylabel("margin added to the prediction")
    ax.set_title("(c) margin table used by the schedule", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(f"Coverage {1 - args.day_eps:.0%} between {args.day_start:g}:00 and {args.day_end:g}:00, "
                 f"{1 - args.night_eps:.0%} otherwise - horizon {(args.horizon + 1) * 10} min - test set", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Saved {args.out}")


def figure_busiest(ev, up, args):
    order = np.argsort(ev["truth"].mean(0))[::-1]
    cells = [int(x) for x in order[:min(args.n_busy, order.size)]]
    u = ev["day"] * STEPS_PER_DAY + ev["slot"]
    idx = u - u.min()
    n = int(idx.max()) + 1
    a = args.start
    if a >= n:
        raise ValueError(f"--start must be < {n}")
    b = n if args.length is None else min(a + args.length, n)
    x = np.arange(a, b) * STEP_H
    slot0 = int(ev["slot"][0])
    tod0 = (slot0 + a) * STEP_H % 24
    dow0 = int(ev["dow"][0]) + (slot0 + a) // STEPS_PER_DAY
    night = ~in_window((slot0 + np.arange(a, b)) % STEPS_PER_DAY, args.day_start, args.day_end)
    day_rows = in_window(ev["slot"], args.day_start, args.day_end)

    fig, axes = plt.subplots(len(cells), 1, figsize=(24 if (b - a) > 432 else 16, 3.8 * len(cells)),
                             sharex=True, squeeze=False)
    for j, ax in enumerate(axes[:, 0]):
        c = cells[j]
        t = to_full(ev["truth"][:, c], idx, n)[a:b]
        p = to_full(ev["pred"][:, c], idx, n)[a:b]
        U = to_full(up[:, c], idx, n)[a:b]
        ax.fill_between(x, 0, 1, where=night, transform=ax.get_xaxis_transform(), color="0.85", alpha=0.5, lw=0)
        ax.plot(x, t, color=C_TRUE, lw=1.3, label="true")
        ax.plot(x, p, color=C_PRED, lw=1.3, ls="--", label="pred (ensemble mean)")
        ax.plot(x, U, color=C_UP, lw=1.2,
                label=f"pred + margin ({1 - args.day_eps:.0%} {args.day_start:g}-{args.day_end:g} h, "
                      f"{1 - args.night_eps:.0%} otherwise)")
        ok = ev["truth"][:, c] <= up[:, c]
        ax.set_ylim(bottom=0)
        ax.set_ylabel("traffic")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title(f"#{j + 1} busiest (col {c}): mean traffic {ev['truth'][:, c].mean():.0f}, coverage "
                     f"{ok[day_rows].mean():.3f} in {args.day_start:g}-{args.day_end:g} h / "
                     f"{ok[~day_rows].mean():.3f} otherwise", fontsize=10, pad=14)
        clock_axis(ax, x, tod0, dow0)
    axes[-1, 0].set_xlabel("time (clock) - grey = hours with the lower coverage")
    plt.tight_layout()
    plt.savefig(args.out_busy, dpi=150)
    plt.close(fig)
    print(f"cells plotted (busiest first): {cells}")
    print(f"Saved {args.out_busy}")


def save_json(c, tables, args):
    q_day, q_night = tables
    data = {
        "horizon_min": (args.horizon + 1) * 10,
        "window_hours": [args.day_start, args.day_end],
        "eps_window": args.day_eps, "eps_outside": args.night_eps,
        "decile_edges": [float(e) for e in c.edges],
        "margin_window": [float(q) for q in q_day],
        "margin_outside": [float(q) for q in q_night],
        "usage": "decile = searchsorted(decile_edges, prediction, side='right'); "
                 "bound = prediction + margin_<period>[decile]; period = window if the clock time of the "
                 "forecast target is in [window_hours[0], window_hours[1]) else outside",
    }
    with open(args.out_json, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {args.out_json}")


def run(cal, ev, args):
    c = Calib(cal, ev, args)
    bounds, tables, keys = build_bounds(c, ev, args)
    report(c, ev, bounds, keys, args)
    figure_summary(c, ev, bounds, tables, keys, args)
    figure_busiest(ev, bounds[K_SCHED], args)
    save_json(c, tables, args)


def main():
    args = parse_args()
    if not (0 < args.day_eps < 1 and 0 < args.night_eps < 1):
        raise SystemExit("--day_eps and --night_eps must be in (0, 1)")
    ev = load_split(args.data, args.ckpts, args, "evaluation")
    if args.cal_data:
        if not args.cal_ckpts:
            raise SystemExit("--cal_data needs --cal_ckpts")
        cal = load_split(args.cal_data, args.cal_ckpts, args, "calibration")
        if cal["members"] != ev["members"]:
            raise SystemExit("different number of members between calibration and evaluation")
    else:
        cut = int(ev["pred"].shape[0] * args.cal_frac)
        cal, ev = subset(ev, slice(0, cut)), subset(ev, slice(cut, ev["pred"].shape[0]))
        print(f"FALLBACK: calibration = first {args.cal_frac:.0%} of test, evaluation = the rest")
    print(f"\n{ev['members']} members | horizon {(args.horizon + 1) * 10} min | ORIGINAL units | "
          f"calibration windows {cal['pred'].shape[0]}, evaluation windows {ev['pred'].shape[0]}")
    run(cal, ev, args)


if __name__ == "__main__":
    main()
