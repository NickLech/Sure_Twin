"""
Ablazione sul numero e sulla forma dei bin (decili uniformi o top-split) per la schedule Mondrian.

Cosa fa questo file:
- Confronta varianti del binning del livello predetto (--n_bins_list) e/o split dell'ultimo decile
  (--top_splits, che divide il decile 90-100% in k sotto-bin mantenendo i primi 9 decili).
- Confronta ogni variante con la schedule base (10 decili) e con la curva a margine scalato con alpha,
  includendo intervalli di confidenza bootstrap sui giorni (--bootstrap_iters).
- Salva adaptive_bins_metrics.csv, il grafico di frontiera e il grafico di guadagno vs base/STID.
"""
import argparse
import os

import numpy as np

import analysis.discarded.metrics_common as mc
import schedule_conformal as sc
from conformal_eval import conf_q

METHOD = "adaptive_bins"
MIN_GROUP = 100


# ----------------------------------------------------------------------
# Argomenti
# ----------------------------------------------------------------------
def parse_all(argv=None):
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--out_dir", default="final_metrics/adaptive_bins")
    ap.add_argument("--horizons", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--n_bins_list", type=int, nargs="*", default=[5, 10, 20],
                    help="numeri di bin quantili uniformi da confrontare")
    ap.add_argument("--top_splits", type=int, nargs="*", default=[0, 2, 4],
                    help="sotto-bin in cui dividere l'ultimo decile (0 = disabilitato; k -> 9 decili + k sotto-bin)")
    ap.add_argument("--bootstrap_iters", type=int, default=200,
                    help="iterazioni di bootstrap a blocchi sui giorni per i CI dei delta (0 = disabilitato)")
    ap.add_argument("--seed", type=int, default=0, help="seed per il bootstrap")
    mc.add_args(ap)
    extra, rest = ap.parse_known_args(argv)
    args = sc.parse_args(["--day_start", "9", "--day_end", "18"] + rest)
    for k, v in vars(extra).items():
        setattr(args, k, v)
    mc.set_lambdas(args)
    return args


# ----------------------------------------------------------------------
# Costruzione soglie e calibrazione Mondrian dei margini
# ----------------------------------------------------------------------
def build_edges(pred, n_bins=None, top_split=0):
    """Restituisce (edges, label_bin, n_tot_bins) per bin uniformi oppure decili + split del top decile."""
    if top_split and top_split > 0:
        dec_edges = np.quantile(pred, np.linspace(0.1, 0.9, 9))
        top_q = np.linspace(0.9, 1.0, top_split + 1)[1:-1]
        top_edges = np.quantile(pred, top_q) if top_q.size > 0 else np.array([], dtype=float)
        edges = np.concatenate([dec_edges, top_edges])
        return edges, 9 + top_split
    if n_bins is None or n_bins < 1:
        raise ValueError("n_bins deve essere >= 1")
    return np.quantile(pred, np.linspace(0, 1, n_bins + 1)[1:-1]), n_bins


class BinCalib:
    """Calibra un margine per (periodo, bin del livello predetto) sui residui truth - pred."""

    def __init__(self, cal, args, n_bins=None, top_split=0):
        self.args = args
        day = sc.in_window(cal["slot"], args.day_start, args.day_end)[:, None]
        res = cal["truth"] - cal["pred"]
        self.edges, self.n_bins = build_edges(cal["pred"], n_bins=n_bins, top_split=top_split)
        lb = np.searchsorted(self.edges, cal["pred"], side="right")
        sel = [[(lb == b) & (day == p) for p in (True, False)] for b in range(self.n_bins)]
        small = min(s.sum() for pair in sel for s in pair)
        if small < MIN_GROUP:
            raise ValueError(f"binning ({self.n_bins} bin): solo {small} coppie nel gruppo più piccolo")
        self.q_day = np.array([max(0.0, conf_q(res[pair[0]], args.day_eps)) for pair in sel])
        self.q_night = np.array([max(0.0, conf_q(res[pair[1]], args.night_eps)) for pair in sel])

    def bound(self, d):
        a = self.args
        day = sc.in_window(d["slot"], a.day_start, a.day_end)[:, None]
        lb = np.searchsorted(self.edges, d["pred"], side="right")
        return d["pred"] + np.where(day, self.q_day[lb], self.q_night[lb])


# ----------------------------------------------------------------------
# Report e esecuzione
# ----------------------------------------------------------------------
def uo(m):
    return f"{m['unmet_%']:5.2f}/{m['overprov_%']:5.2f}"


def print_report(res, info, names, deltas, cis, args):
    glob = res["global"]["all cells"]
    print("\n== 1) GLOBALE, TUTTE LE CELLE, all hours (negativo = meglio della frontiera a margine scalato) ==")
    print(f"{'approach':38s} {'cover':>6s} {'unmet%':>8s} {'over%':>8s} | "
          f"{'d_unmet@iso-over':>16s} {'CI95_unmet':>15s} | {'d_over@iso-unmet':>16s} {'CI95_over':>15s}")
    for n in names:
        d = deltas.get(n)
        ci = cis.get(n)
        if d is not None:
            ci_u = f"[{ci['d_unmet_ci95'][0]:+.3f},{ci['d_unmet_ci95'][1]:+.3f}]" if ci else "n/a"
            ci_o = f"[{ci['d_over_ci95'][0]:+.3f},{ci['d_over_ci95'][1]:+.3f}]" if ci else "n/a"
            tail = f"{d[0]:+16.3f} {ci_u:>15s} | {d[1]:+16.3f} {ci_o:>15s}"
        else:
            tail = ""
        print(f"{n:38s} {mc.fmt(glob[n]['all hours'])} | {tail}")

    print("\n== 2) PER TIER STATICO (media su train), all hours: unmet% / over% ==")
    print(f"{'tier':14s} {'act%':>5s} | " + " | ".join(f"{n[:12]:^12s}" for n in names))
    for name, t in info["tier"].items():
        print(f"{name:14s} {t['share_activity_%']:5.1f} | "
              + " | ".join(uo(res["tier"][name][n]["all hours"]) for n in names))


def run_one(cal, ev, args, cols):
    truth, pred = ev["truth"], ev["pred"]
    up_ref = mc.ref_bounds(cal, ev, args)
    est = {mc.K_POINT: pred, mc.K_REF: up_ref}
    cand_names = []

    for n in args.n_bins_list:
        label = f"{n} bins"
        if label not in est:
            est[label] = BinCalib(cal, args, n_bins=n).bound(ev)
            cand_names.append(label)

    for k in args.top_splits:
        if k and k > 0:
            label = f"deciles+top{k}"
            if label not in est:
                est[label] = BinCalib(cal, args, top_split=k).bound(ev)
                cand_names.append(label)

    deltas, cis, params = {}, {}, {}
    for name in [mc.K_REF] + cand_names:
        du, do = mc.frontier_deltas(pred, up_ref, truth, est[name])
        deltas[name] = (du, do)
        ci = mc.bootstrap_frontier_ci(pred, up_ref, truth, est[name], ev["day"],
                                     n_iter=args.bootstrap_iters, seed=args.seed)
        cis[name] = ci
        p_str = f"d_unmet_at_iso_over={du:+.3f};d_over_at_iso_unmet={do:+.3f}"
        if ci is not None:
            p_str += (f";ci95_unmet=[{ci['d_unmet_ci95'][0]:+.3f},{ci['d_unmet_ci95'][1]:+.3f}]"
                      f";ci95_over=[{ci['d_over_ci95'][0]:+.3f},{ci['d_over_ci95'][1]:+.3f}]")
        params[name] = p_str

    tier, tier_mean, tier_names, _ = mc.static_tiers(args, cols)
    scopes = {"global": {"all cells": None}, "tier": mc.tier_groups(tier, tier_names, truth.shape)}
    day = mc.day_mask(ev, args)
    res = mc.evaluate_scopes(est, truth, day, args, scopes)
    info = {"tier": mc.tier_info(tier, tier_mean, tier_names, truth)}
    print_report(res, info, list(est), deltas, cis, args)

    h, window = args.horizon * 10, f"{args.day_start:g}-{args.day_end:g}"
    base = os.path.join(args.out_dir, f"adaptive_bins_h{h}_w{window}")
    mc.figure_frontier(base + "_frontier.png", pred, up_ref, truth, est,
                       f"Bins ablation vs scaled reference - horizon {h} min")
    if cand_names:
        best_cand = cand_names[-1]
        mc.figure_gain_vs_baselines(
            base + "_gain.png", pred, up_ref, truth, day, best_cand, est[best_cand], args,
            f"{best_cand} vs schedule base & STID point - horizon {h} min ({window} h)"
        )
    return mc.to_rows(METHOD, h, window, res, params)


def main():
    args = parse_all()
    if not (0 < args.day_eps < 1 and 0 < args.night_eps < 1):
        raise SystemExit("--day_eps e --night_eps devono essere in (0, 1)")
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