"""Comparison tables (all horizons x all windows) from the adaptive_bins_metrics.csv files.

Reads the CSVs written by schedule_adaptive_bins.py (one per window run) and writes a README-style
Markdown file with two tables, the base schedule (10 deciles) against one variant (default: deciles+top8):

  Table 1  weighted cost J per band (window / other hours), base vs variant, and relative change
  Table 2  gain of the variant against the "same schedule with the margin scaled by alpha" frontier:
           change of unmet at equal over-provisioning and of over-provisioning at equal unmet,
           with the 95% bootstrap interval (a gain is real when the interval does not contain 0)

Usage (from the repo root):
  python make_bins_table.py "final_metrics/adaptive_bins_all/w*/adaptive_bins_metrics.csv" \
      --variant deciles+top8 --out final_metrics/adaptive_bins_all/README.md

Read-only on the inputs; nothing else is touched.
"""
import argparse
import csv
import glob
import re

BASE = "scheduled, calibrated per period"


def parse_params(s):
    """'d_unmet_at_iso_over=-0.055;d_over_at_iso_unmet=-0.408;ci95_unmet=[-0.066,-0.046];ci95_over=[..]'"""
    out = {}
    for part in (s or "").split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        if v.startswith("["):
            out[k] = tuple(float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", v))
        else:
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out


def load(paths):
    """{(horizon_min, window): {(approach, period): row}} using the global scope only."""
    data = {}
    for p in paths:
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                if r["scope"] != "global":
                    continue
                key = (int(r["horizon_min"]), r["window"])
                data.setdefault(key, {})[(r["approach"], r["period"])] = r
    return data


def order(key):
    h, w = key
    s, e = (float(x) for x in w.split("-"))
    return (h, e - s, s)


def pct(new, old):
    return f"{100 * (new - old) / old:+.1f}%" if old else "n/a"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+", help="adaptive_bins_metrics.csv files (globs allowed)")
    ap.add_argument("--variant", default="deciles+top8", help="approach compared with the base schedule")
    ap.add_argument("--out", default=None, help="Markdown output file (default: print only)")
    args = ap.parse_args()

    paths = sorted({p for pat in args.csvs for p in glob.glob(pat)})
    if not paths:
        raise SystemExit("no CSV found")
    data = load(paths)
    keys = sorted(data, key=order)
    miss = [k for k in keys if (args.variant, "window") not in data[k] or (BASE, "window") not in data[k]]
    if miss:
        raise SystemExit(f"approach '{args.variant}' or the base missing for: {miss}")

    v = args.variant
    t1 = ["| Horizon | Window | J window base | J window " + v + " | Δ J window | J other hours base | "
          "J other hours " + v + " | Δ J other hours |", "|---|---|---|---|---|---|---|---|"]
    t2 = ["| Horizon | Window | Δ unmet at equal over-prov. [95% CI] | Δ over-prov. at equal unmet [95% CI] | "
          "Gain significant |", "|---|---|---|---|---|"]
    for k in keys:
        d = data[k]
        jb = {p: float(d[(BASE, p)]["cost_J"]) for p in ("window", "other hours")}
        jv = {p: float(d[(v, p)]["cost_J"]) for p in ("window", "other hours")}
        t1.append(f"| {k[0]} min | {k[1]} | {jb['window']:.2f} | {jv['window']:.2f} | {pct(jv['window'], jb['window'])} | "
                  f"{jb['other hours']:.2f} | {jv['other hours']:.2f} | {pct(jv['other hours'], jb['other hours'])} |")
        pr = parse_params(d[(v, "window")]["params"])
        if "ci95_unmet" in pr and "ci95_over" in pr:
            cu, co = pr["ci95_unmet"], pr["ci95_over"]
            sig = "yes" if (cu[1] < 0 or cu[0] > 0) and (co[1] < 0 or co[0] > 0) else "no"
            t2.append(f"| {k[0]} min | {k[1]} | {pr['d_unmet_at_iso_over']:+.3f} [{cu[0]:+.3f}, {cu[1]:+.3f}] | "
                      f"{pr['d_over_at_iso_unmet']:+.3f} [{co[0]:+.3f}, {co[1]:+.3f}] | {sig} |")
        else:
            t2.append(f"| {k[0]} min | {k[1]} | {pr.get('d_unmet_at_iso_over', float('nan')):+.3f} | "
                      f"{pr.get('d_over_at_iso_unmet', float('nan')):+.3f} | no CI (run with --bootstrap_iters) |")

    md = f"""# Base schedule vs {v}: all horizons and windows

Base schedule = *{BASE}* with 10 deciles of the forecast level (the proposal). Variant = **{v}**.
All values on the test set; costs and metrics are percentages of the total true activity index (arbitrary units, [a.u.]); they are not energy.

J = λ · unmet + (1 − λ) · over-provisioning, with λ = 1 − ε in each band (defaults: 0.90 in the window, 0.75 outside). Lower is better.

## Table 1: weighted cost J

{chr(10).join(t1)}

## Table 2: gain against the scaled-margin frontier

Each variant is compared with the base schedule whose margins are multiplied by a constant α, chosen so that it matches the variant either on over-provisioning or on unmet. Negative = the variant is better than just scaling the margin. A gain is considered real when the 95% bootstrap interval (resampling days) excludes 0. In the CSV the two measures are identical for the three time bands.

{chr(10).join(t2)}

## Limitations

- J uses λ = 1 − ε, the loss the schedule is designed to minimise: it is consistent with the method but not an independent validation. The λ sweep shows how the ranking changes with λ.
- The test set covers a limited number of days, so the bootstrap intervals are indicative.
- J for "all hours" is not shown: it is not comparable across windows (changing the window changes which hours use each λ).
"""
    print(md)
    if args.out:
        with open(args.out, "w") as f:
            f.write(md)
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
