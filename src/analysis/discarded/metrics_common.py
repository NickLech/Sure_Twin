"""
Funzioni condivise per metriche, baseline iso, bootstrap e grafici di confronto.

Cosa fa questo file:
- Calcola cover, unmet %, over-provisioning % e costo pesato J su scala originale [a.u.].
- Stima la frontiera ottenuta scalando il margine della schedule base con una costante alpha.
- Calcola intervalli di confidenza bootstrap a blocchi sui giorni per i delta rispetto alla frontiera.
- Genera i grafici di frontiera e di guadagno/perdita rispetto a schedule base e STID point.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import schedule_conformal as sc
from analyse_result import kept_columns, load_series

PERIODS = ("all hours", "window", "other hours")
K_POINT = "STID point prediction (no confidence)"
K_REF = sc.K_SCHED
COLUMNS = ["method", "horizon_min", "window", "scope", "group", "approach", "period",
           "cover", "unmet_%", "overprov_%", "cost_J", "switch_%", "params"]
NAN = float("nan")


# ----------------------------------------------------------------------
# Argomenti e caricamento dati
# ----------------------------------------------------------------------
def add_args(ap):
    ap.add_argument("--lam", type=float, default=None,
                    help="peso di unmet in J = lam*unmet + (1-lam)*overprov nella finestra; default 1 - day_eps")
    ap.add_argument("--lam_night", type=float, default=None,
                    help="come --lam ma fuori finestra; default 1 - night_eps")
    ap.add_argument("--tiers", type=float, nargs="*", default=[0.5, 0.75, 0.9, 0.97],
                    help="quantili della media su train per dividere le celle in tier statici")


def set_lambdas(args):
    """Risolve i pesi J di default dopo aver letto day_eps e night_eps."""
    if args.lam is None:
        args.lam = 1 - args.day_eps
    if args.lam_night is None:
        args.lam_night = 1 - args.night_eps
    if not (0 <= args.lam <= 1 and 0 <= args.lam_night <= 1):
        raise SystemExit("--lam e --lam_night devono essere in [0, 1]")


def load_horizon(args):
    """Carica split di calibrazione/valutazione e colonne delle celle tenute."""
    cal, ev = sc.load_cal_ev(args)
    cols = kept_columns(args, load_series(Path(args.data) / "test_data.npy").shape[1])
    return cal, ev, cols


def ref_bounds(cal, ev, args):
    """Calcola il bound di riferimento (schedule a 10 decili) su `ev` calibrato su `cal`."""
    bounds, _, _ = sc.build_bounds(sc.Calib(cal, ev, args), ev, args)
    return bounds[sc.K_SCHED]


# ----------------------------------------------------------------------
# Metriche
# ----------------------------------------------------------------------
def day_mask(ev, args):
    """Maschera bool [finestre, celle] per i target che cadono nella finestra ad alta copertura."""
    return np.broadcast_to(sc.in_window(ev["slot"], args.day_start, args.day_end)[:, None], ev["truth"].shape)


def compute(est, truth, mask=None):
    """Restituisce cover (0-1), unmet % e over-provisioning % sull'attività vera nella maschera."""
    t, e = (truth, est) if mask is None else (truth[mask], est[mask])
    if t.size == 0 or t.sum() <= 0:
        return {"cover": NAN, "unmet_%": NAN, "overprov_%": NAN}
    d = e - t
    return {"cover": float((t <= e).mean()),
            "unmet_%": float(100 * np.maximum(-d, 0).sum() / t.sum()),
            "overprov_%": float(100 * np.maximum(d, 0).sum() / t.sum())}


def evaluate(estimates, truth, day, args, group=None, extra=None):
    """Valuta {approccio: {periodo: metriche}} dentro `group` (None = tutte le coppie)."""
    g = np.ones(truth.shape, dtype=bool) if group is None else group
    masks = {"all hours": g, "window": g & day, "other hours": g & ~day}
    total = truth[g].sum()
    share = truth[masks["window"]].sum() / total if total > 0 else NAN
    lam = {"window": args.lam, "other hours": args.lam_night}
    out = {}
    for n, est in estimates.items():
        per = {}
        for p, m in masks.items():
            per[p] = compute(est, truth, m)
            if extra is not None:
                per[p].update(extra(n, m))
        j = {p: lam[p] * per[p]["unmet_%"] + (1 - lam[p]) * per[p]["overprov_%"] for p in lam}
        per["window"]["cost_J"], per["other hours"]["cost_J"] = j["window"], j["other hours"]
        per["all hours"]["cost_J"] = share * j["window"] + (1 - share) * j["other hours"]
        out[n] = per
    return out


def evaluate_scopes(estimates, truth, day, args, scopes, extra=None):
    """Applica evaluate() su una struttura {scope: {gruppo: maschera_or_None}}."""
    return {s: {g: evaluate(estimates, truth, day, args, m, extra) for g, m in groups.items()}
            for s, groups in scopes.items()}


def per_cell_stats(est, truth):
    """Statistiche per cella di unmet % (sull'attività della cella) e copertura."""
    u = 100 * np.maximum(truth - est, 0).sum(0) / np.maximum(truth.sum(0), 1e-12)
    cov = (truth <= est).mean(0)
    return {"unmet_med": float(np.median(u)), "unmet_p90": float(np.percentile(u, 90)),
            "unmet_p99": float(np.percentile(u, 99)), "unmet_max": float(u.max()),
            "cover_med": float(np.median(cov)), "cover_p10": float(np.percentile(cov, 10)),
            "cover_min": float(cov.min())}


def fmt(m):
    """Formatta 'cover unmet% over%' in 24 caratteri."""
    return f"{m['cover']:6.3f} {m['unmet_%']:8.2f} {m['overprov_%']:8.2f}"


# ----------------------------------------------------------------------
# Frontiera a margine scalato con alpha e controllo bootstrap sui giorni
# ----------------------------------------------------------------------
def match_alpha(pred, margin, truth, key, target):
    """Trova alpha via bisezione tale che `key` di pred + alpha*margin uguagli `target`."""
    lo, hi = 0.0, 5.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        m = compute(pred + mid * margin, truth)[key]
        below = m < target if key == "overprov_%" else m > target
        lo, hi = (mid, hi) if below else (lo, mid)
    return 0.5 * (lo + hi)


def iso_baselines(pred, ref_bound, truth, est):
    """Restituisce (bound_iso_over, bound_iso_unmet, alpha_over, alpha_unmet) rispetto a `est`."""
    margin = np.maximum(ref_bound - pred, 0.0)
    m = compute(est, truth)
    a_o = match_alpha(pred, margin, truth, "overprov_%", m["overprov_%"])
    a_u = match_alpha(pred, margin, truth, "unmet_%", m["unmet_%"])
    return pred + a_o * margin, pred + a_u * margin, a_o, a_u


def frontier_deltas(pred, ref_bound, truth, est):
    """Calcola (d_unmet@iso-over, d_over@iso-unmet); valori negativi = meglio della frontiera."""
    io, iu, _, _ = iso_baselines(pred, ref_bound, truth, est)
    m, mo, mu = (compute(e, truth) for e in (est, io, iu))
    return float(m["unmet_%"] - mo["unmet_%"]), float(m["overprov_%"] - mu["overprov_%"])


def bootstrap_frontier_ci(pred, ref_bound, truth, est, days, n_iter=200, seed=0):
    """CI al 95% (percentile) sui delta rispetto alla frontiera campionando i giorni con reinserimento."""
    if n_iter <= 0:
        return None
    uniq_days, inv = np.unique(days, return_inverse=True)
    if uniq_days.size < 2:
        return None
    day_rows = [np.flatnonzero(inv == i) for i in range(uniq_days.size)]
    rng = np.random.default_rng(seed)
    du_list, do_list = [], []
    for _ in range(n_iter):
        pick = rng.integers(0, len(day_rows), size=len(day_rows))
        rows = np.concatenate([day_rows[k] for k in pick])
        du, do = frontier_deltas(pred[rows], ref_bound[rows], truth[rows], est[rows])
        du_list.append(du)
        do_list.append(do)
    return {
        "d_unmet_ci95": (float(np.percentile(du_list, 2.5)), float(np.percentile(du_list, 97.5))),
        "d_over_ci95": (float(np.percentile(do_list, 2.5)), float(np.percentile(do_list, 97.5))),
    }


def figure_frontier(path, pred, ref_bound, truth, est, title):
    """Confronta unmet vs over-provisioning del metodo con la curva pred + alpha * margine_base."""
    margin = np.maximum(ref_bound - pred, 0.0)
    pts = [compute(pred + a * margin, truth) for a in np.linspace(0.0, 1.6, 33)]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot([p["overprov_%"] for p in pts], [p["unmet_%"] for p in pts], "--", color="gray",
            label="reference schedule x alpha")
    for n, e in est.items():
        m = compute(e, truth)
        ax.scatter(m["overprov_%"], m["unmet_%"], s=70, zorder=3, edgecolors="k", label=n)
    ax.set_xlabel("over-provisioning [% of total true activity]")
    ax.set_ylabel("unmet [% of total true activity]")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def figure_gain_vs_baselines(path, pred, ref_bound, truth, day, cand_name, cand_est, args, title):
    """Grafico stile schedule_final_metrics: scatter per periodo + delta (pp) vs schedule base e STID point."""
    masks = {"all hours": np.ones(truth.shape, dtype=bool),
             "window": day,
             "other hours": ~day}
    xt = [f"{args.day_start:g}-{args.day_end:g} h" if p == "window" else p for p in PERIODS]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4))

    # Pannello sinistro: scatter (overprov, unmet) per periodo come in schedule_final_metrics
    ax = axes[0]
    styles = [
        (K_POINT, pred, "tab:blue", "o", "STID point"),
        (K_REF, ref_bound, "tab:green", "s", "schedule base (10 decili)"),
        (cand_name, cand_est, "tab:red", "^", cand_name),
    ]
    for idx_p, (p, m) in enumerate(masks.items()):
        for label, est, col, mk, leg in styles:
            mt = compute(est, truth, m)
            ax.scatter(mt["overprov_%"], mt["unmet_%"], s=95 if label == cand_name else 70,
                       c=col, marker=mk, edgecolors="k", linewidths=0.7, zorder=3,
                       label=leg if idx_p == 0 else "")
            ax.annotate(xt[idx_p], (mt["overprov_%"], mt["unmet_%"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=7.5, color=col)
    ax.set_xlabel("over-provisioning, % of total true activity (lower = less waste)")
    ax.set_ylabel("unmet activity, % of total true activity (lower = safer)")
    ax.set_title("Trade-off per period (bottom-left = better)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    # Pannello destro: delta rispetto a schedule base e a STID point su all hours
    ax = axes[1]
    m_cand = compute(cand_est, truth, masks["all hours"])
    m_ref = compute(ref_bound, truth, masks["all hours"])
    m_pt = compute(pred, truth, masks["all hours"])
    labels_b = [
        f"Δ unmet\nvs base",
        f"Δ over\nvs base",
        f"Δ unmet\nvs STID point",
        f"Δ over\nvs STID point",
    ]
    vals_b = [
        m_cand["unmet_%"] - m_ref["unmet_%"],
        m_cand["overprov_%"] - m_ref["overprov_%"],
        m_cand["unmet_%"] - m_pt["unmet_%"],
        m_cand["overprov_%"] - m_pt["overprov_%"],
    ]
    cols_b = ["tab:orange" if v > 0 else "tab:green" for v in vals_b]
    x_b = np.arange(len(labels_b))
    bars = ax.bar(x_b, vals_b, width=0.55, color=cols_b, edgecolor="k", linewidth=0.6)
    ax.axhline(0.0, color="k", lw=1.0)
    for bar, v in zip(bars, vals_b):
        va = "bottom" if v >= 0 else "top"
        dy = 0.04 if v >= 0 else -0.04
        ax.text(bar.get_x() + bar.get_width() / 2, v + dy, f"{v:+.2f} pp",
                ha="center", va=va, fontsize=8)
    ax.set_xticks(x_b)
    ax.set_xticklabels(labels_b, fontsize=8.5)
    ax.set_ylabel("difference vs baseline [percentage points, all hours]")
    ax.set_title("Gain (< 0) or loss (> 0) vs base schedule & STID point")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ----------------------------------------------------------------------
# Tier statici delle celle (media su train, senza leakage sul test)
# ----------------------------------------------------------------------
def static_tiers(args, cols):
    """Restituisce indice tier per cella, media su train, nomi tier e serie train."""
    train = load_series(Path(args.data) / "train_data.npy")[:, cols]
    mean = train.mean(0)
    tier = np.searchsorted(np.quantile(mean, args.tiers), mean, side="right")
    cuts = [0.0] + list(args.tiers) + [1.0]
    names = [f"cells {cuts[i]:.0%}-{cuts[i + 1]:.0%}" for i in range(len(cuts) - 1)]
    return tier, mean, names, train


def tier_groups(tier, names, shape):
    return {n: np.broadcast_to((tier == i)[None, :], shape) for i, n in enumerate(names)}


def tier_info(tier, mean, names, truth):
    out = {}
    for i, n in enumerate(names):
        cell = tier == i
        out[n] = {"n_cells": int(cell.sum()), "train_mean_au": float(mean[cell].mean()),
                  "share_activity_%": float(100 * truth[:, cell].sum() / truth.sum())}
    return out


# ----------------------------------------------------------------------
# Esportazione CSV
# ----------------------------------------------------------------------
def to_rows(method, horizon_min, window, res, params=None):
    """Appiattisce {scope: {gruppo: {approccio: {periodo: metriche}}}} in righe CSV."""
    params = params or {}
    return [{"method": method, "horizon_min": horizon_min, "window": window, "scope": scope, "group": group,
             "approach": approach, "period": p, "params": params.get(approach, ""), **per[p]}
            for scope, groups in res.items() for group, apps in groups.items()
            for approach, per in apps.items() for p in PERIODS]


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, restval=NAN)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {path} ({len(rows)} rows)")