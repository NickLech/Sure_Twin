"""
Bias diagnosis for the traffic predictor.

Usage:
    python diagnose_bias.py --train preds_train.csv --val preds_val.csv \
                            --test preds_test.csv --out bias_report

Input CSVs: columns timestamp, cell, pred, true (original scale, ensemble mean).
Sign convention: err = pred - true  (negative = underestimation).
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")


# ----------------------------------------------------------------- loading
def load(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    missing = {"timestamp", "cell", "pred", "true"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df["err"] = df["pred"] - df["true"]
    df["hour"] = df["timestamp"].dt.hour
    df["slot"] = df["timestamp"].dt.hour * 6 + df["timestamp"].dt.minute // 10
    return df


def stats(df):
    err = df["err"]
    bias = err.mean()
    mae = err.abs().mean()
    return {
        "n": len(df),
        "bias": bias,
        "median_err": err.median(),
        "mae": mae,
        "rmse": np.sqrt((err ** 2).mean()),
        "|bias|/mae": abs(bias) / mae if mae > 0 else np.nan,
        "pred/true": df["pred"].sum() / df["true"].sum(),
    }


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ------------------------------------------------- 1. where does it arise
def step1_splits(dfs):
    section("1. BIAS PER SPLIT  (where does the bias appear?)")
    table = pd.DataFrame({k: stats(v) for k, v in dfs.items()}).T
    print(table)
    print(
        "\nHow to read it:\n"
        "  - bias already negative on train/val -> model defect (loss, target, scale)\n"
        "  - bias ~0 on train/val, negative only on test -> drift between periods\n"
        "  - |bias|/mae close to 1 -> the error is almost all systematic\n"
        "  - pred/true < 1 -> total traffic underestimated by (1 - pred/true)"
    )

    section("1b. BIAS PER CELL  (uniform or concentrated?)")
    for name, df in dfs.items():
        g = df.groupby("cell").agg(
            bias=("err", "mean"), mae=("err", lambda x: x.abs().mean()),
            mean_true=("true", "mean"),
        )
        corr = g["bias"].corr(g["mean_true"])
        print(
            f"[{name}] cells with negative bias: {(g['bias'] < 0).mean():.0%} | "
            f"bias quantiles 10/50/90%: "
            f"{g['bias'].quantile(.1):.2f} / {g['bias'].median():.2f} / {g['bias'].quantile(.9):.2f} | "
            f"corr(bias, cell mean traffic) = {corr:.2f}"
        )
    print(
        "\n  corr strongly negative -> high-traffic cells are the ones underestimated.\n"
        "  Most cells negative with similar size -> global effect, not local."
    )


# ------------------------------------------- 2. why: peaks / target / loss
def step2_level(dfs, out, n_bins=10):
    section("2. BIAS vs TRAFFIC LEVEL  (are peaks being smoothed?)")
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, df in dfs.items():
        bins = pd.qcut(df["true"].rank(method="first"), n_bins, labels=False)
        g = df.groupby(bins).agg(
            true_mean=("true", "mean"), bias=("err", "mean"),
            mae=("err", lambda x: x.abs().mean()),
        )
        g["rel_bias_%"] = 100 * g["bias"] / g["true_mean"].replace(0, np.nan)
        print(f"\n[{name}]  (bin 0 = lowest traffic, bin {n_bins - 1} = highest)")
        print(g)
        ax.plot(g.index, g["bias"], marker="o", label=name)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("traffic decile (true)")
    ax.set_ylabel("mean error (pred - true)")
    ax.set_title("Bias vs traffic level")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "bias_vs_level.png"), dpi=130)
    plt.close(fig)
    print(
        "\n  Bias negative ONLY in the top deciles -> peaks smoothed (MSE / regression to the mean).\n"
        "  Bias negative in ALL deciles with similar size -> level shift (scaler, target transform, drift).\n"
        "  Bias is negative at low traffic and positive at high -> over-smoothing of everything."
    )


# --------------------------------------- 3. is it structured / correctable
def hourly_profile(df):
    g = df.groupby("hour")["err"].agg(["mean", "std", "count"])
    g["sem"] = g["std"] / np.sqrt(g["count"])
    return g


def step3_hours(dfs, out):
    section("3. BIAS BY HOUR  (is it structured and correctable?)")
    profiles = {k: hourly_profile(v) for k, v in dfs.items()}

    fig, ax = plt.subplots(figsize=(9, 4))
    for name, p in profiles.items():
        ax.plot(p.index, p["mean"], marker="o", ms=3, label=name)
        ax.fill_between(p.index, p["mean"] - 1.96 * p["sem"],
                        p["mean"] + 1.96 * p["sem"], alpha=0.15)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("hour of day")
    ax.set_ylabel("mean error (pred - true)")
    ax.set_title("Hourly bias profile (band = 95% CI of the mean)")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "bias_by_hour.png"), dpi=130)
    plt.close(fig)

    names = list(dfs)
    print("Correlation between 24h bias profiles:")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c = profiles[names[i]]["mean"].corr(profiles[names[j]]["mean"])
            print(f"  {names[i]} vs {names[j]}: {c:.2f}")
    print(
        "  High correlation (> ~0.7) -> same pattern across periods -> structured, correctable."
    )

    if "val" in dfs and "test" in dfs:
        section("3b. CORRECTION TEST  (estimate on val, apply on test)")
        val, test = dfs["val"], dfs["test"]
        rows = {"no correction": stats(test)}

        corr_hour = val.groupby("hour")["err"].mean()
        t = test.copy()
        t["pred"] = t["pred"] - t["hour"].map(corr_hour).fillna(0)
        t["err"] = t["pred"] - t["true"]
        rows["hourly correction"] = stats(t)

        corr_slot = val.groupby("slot")["err"].mean()
        t = test.copy()
        t["pred"] = t["pred"] - t["slot"].map(corr_slot).fillna(0)
        t["err"] = t["pred"] - t["true"]
        rows["10-min slot correction"] = stats(t)

        corr_global = val["err"].mean()
        t = test.copy()
        t["pred"] = t["pred"] - corr_global
        t["err"] = t["pred"] - t["true"]
        rows["global shift"] = stats(t)

        print(pd.DataFrame(rows).T[["bias", "mae", "rmse", "|bias|/mae"]])
        print(
            "\n  If the hourly correction cuts MAE clearly vs 'no correction', the bias is\n"
            "  stable between val and test -> use it as baseline and condition conformal by hour.\n"
            "  If it barely helps, the bias is drift: you need an adaptive scheme (ACI)."
        )


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=None)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="bias_report")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dfs = {}
    if args.train:
        dfs["train"] = load(args.train)
    dfs["val"] = load(args.val)
    dfs["test"] = load(args.test)

    step1_splits(dfs)
    step2_level(dfs, args.out)
    step3_hours(dfs, args.out)
    print(f"\nPlots saved in: {args.out}/")


if __name__ == "__main__":
    main()
