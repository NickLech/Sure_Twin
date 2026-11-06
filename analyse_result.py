"""
Post-training analysis for STID on MILAN.

Usage:
    python analyze_results.py                       # all checkpoints/STID_MILAN/seed*
        python analyze_results.py --logs "out_*_*.log"
        """
        import argparse
        import glob
        import json
        import re
        from pathlib import Path

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from numpy.lib.stride_tricks import sliding_window_view

        ap = argparse.ArgumentParser()
        ap.add_argument("--ckpts", default="checkpoints/STID_MILAN/seed*/*")
        ap.add_argument("--logs", default="out_*.log")
        ap.add_argument("--data", default="datasets/MILAN")
        ap.add_argument("--input_len", type=int, default=144)
        ap.add_argument("--output_len", type=int, default=12)
        ap.add_argument("--out", default="analysis.png")
        args = ap.parse_args()
        IN, OUT = args.input_len, args.output_len
        data = Path(args.data)

        # ---------- ground truth in original units ----------
        train = np.load(data / "train_data.npy")[..., 0]          # [T_train, N]
        test = np.load(data / "test_data.npy")[..., 0]            # [T_test, N]
        test_ts = np.load(data / "test_timestamps.npy")           # [T_test, 4]
        mean, std = train.mean(0), train.std(0) + 1e-8            # per-node z-score (norm_each_channel=True)
        N = test.shape[1]
        S = test.shape[0] - IN - OUT + 1


        def load(path):
            """BasicTS saves raw float32 memmaps (no .npy header) normalized -> denormalize."""
                a = np.memmap(path, dtype=np.float32, mode="r")
                    assert a.size == S * OUT * N, f"unexpected size {a.size} vs {S}x{OUT}x{N}"
                        return a.reshape(S, OUT, N) * std + mean


                        ckpt_dirs = sorted(glob.glob(args.ckpts))
                        assert ckpt_dirs, "no checkpoint dirs found"
                        targets = load(Path(ckpt_dirs[0]) / "test_results" / "targets.npy")

                        # SANITY CHECK 1: denormalized targets must equal the raw test windows
                        win = sliding_window_view(test, IN + OUT, axis=0)          # [S, N, IN+OUT]
                        true = win[:, :, IN:].transpose(0, 2, 1)                   # [S, OUT, N]
                        print(f"[check] max |targets - raw test| = {np.abs(targets - true).max():.4g} (should be ~0)")

                        # ---------- predictions: single seeds + ensemble ----------
                        preds = np.stack([load(Path(d) / "test_results" / "prediction.npy") for d in ckpt_dirs])  # [M,S,OUT,N]
                        ens_mean, ens_std = preds.mean(0), preds.std(0)

                        # ---------- baselines ----------
                        last = np.repeat(win[:, :, IN - 1][:, None, :], OUT, axis=1)                # last observed value
                        seasonal = win[:, :, :OUT].transpose(0, 2, 1)                               # same slot, 24h before


                        def metrics(p):
                            e = p - true
                                return dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), WAPE=np.abs(e).sum() / np.abs(true).sum())


                                rows = {"last value": last, "seasonal naive (24h)": seasonal}
                                for d, p in zip(ckpt_dirs, preds):
                                    rows[Path(d).name] = p
                                    if len(preds) > 1:
                                        rows["ensemble mean"] = ens_mean
                                        print(f"\n{'model':24s}{'MAE':>10s}{'RMSE':>10s}{'WAPE':>10s}   (original units)")
                                        for k, p in rows.items():
                                            m = metrics(p)
                                                print(f"{k:24s}{m['MAE']:10.3f}{m['RMSE']:10.3f}{m['WAPE']:10.3f}")

                                                # SANITY CHECK 2 (preview of module 3): spread across seeds should grow with the error
                                                if len(preds) > 1:
                                                    err = np.abs(ens_mean - true)
                                                        print(f"\n[ensemble] corr(std, |err|) = {np.corrcoef(ens_std.ravel()[::50], err.ravel()[::50])[0, 1]:.3f} (>0 = useful)")
                                                            cov = (np.abs(true - ens_mean) <= 1.645 * ens_std).mean()
                                                                print(f"[ensemble] coverage of mean ± 1.645 std = {cov:.3f} (nominal 0.90; low is expected with few members)")

                                                                # ---------- plots ----------
                                                                main = ens_mean if len(preds) > 1 else preds[0]
                                                                fig, ax = plt.subplots(2, 3, figsize=(18, 9))

                                                                # (a) learning curves from the logs -> overfitting check
                                                                for f in sorted(glob.glob(args.logs)):
                                                                    txt = open(f).read()
                                                                        tr = [float(x) for x in re.findall(r"train/loss: ([\d.eE+-]+)", txt)]
                                                                            va = [float(x) for x in re.findall(r"val/MAE: ([\d.eE+-]+)", txt)]
                                                                                if tr and va:
                                                                                        ax[0, 0].plot(tr, "--", alpha=.6)
                                                                                                ax[0, 0].plot(va, "-", label=Path(f).stem)
                                                                                                ax[0, 0].set(title="train loss (--) vs val MAE (-), normalized", xlabel="epoch")
                                                                                                ax[0, 0].legend(fontsize=6)

                                                                                                # (b) MAE per horizon vs baselines
                                                                                                for k, p in rows.items():
                                                                                                    if k.startswith("seed"):
                                                                                                            continue
                                                                                                                ax[0, 1].plot(range(1, OUT + 1), np.abs(p - true).mean((0, 2)), marker="o", label=k)
                                                                                                                ax[0, 1].set(title="MAE per horizon", xlabel="horizon (x10 min)")
                                                                                                                ax[0, 1].legend(fontsize=7)

                                                                                                                # (c) pred vs true scatter (h=1)
                                                                                                                idx = np.random.default_rng(0).choice(S * N, 20000, replace=False)
                                                                                                                t1, p1 = true[:, 0].ravel()[idx], main[:, 0].ravel()[idx]
                                                                                                                ax[0, 2].scatter(t1, p1, s=2, alpha=.3)
                                                                                                                lim = [0, max(t1.max(), p1.max())]
                                                                                                                ax[0, 2].plot(lim, lim, "r-")
                                                                                                                ax[0, 2].set(title="pred vs true (h=1)", xlabel="true", ylabel="pred")

                                                                                                                # (d) time series for busiest / median / quietest node, h=1, 3 days (consecutive windows)
                                                                                                                order = np.argsort(true[:, 0].mean(0))
                                                                                                                for n, lab in [(order[-1], "busiest"), (order[N // 2], "median"), (order[0], "quietest")]:
                                                                                                                    ax[1, 0].plot(true[:432, 0, n], label=f"true {lab}")
                                                                                                                        ax[1, 0].plot(main[:432, 0, n], "--", label=f"pred {lab}")
                                                                                                                        ax[1, 0].set(title="h=1, first 3 test days", xlabel="step (10 min)")
                                                                                                                        ax[1, 0].legend(fontsize=6)

                                                                                                                        # (e) error by time of day (matters for sleep decisions: peaks/ramps)
                                                                                                                        tod = (test_ts[IN:IN + S, 0] * 144).round().astype(int) % 144
                                                                                                                        mae_t = np.abs(main[:, 0] - true[:, 0]).mean(1)
                                                                                                                        ax[1, 1].plot([mae_t[tod == h].mean() for h in range(144)])
                                                                                                                        ax[1, 1].set(title="MAE by time of day (h=1)", xlabel="slot (x10 min)")

                                                                                                                        # (f) signed residuals (bias check)
                                                                                                                        res = (main[:, 0] - true[:, 0]).ravel()[idx]
                                                                                                                        ax[1, 2].hist(res, bins=100)
                                                                                                                        ax[1, 2].set(title=f"residuals h=1 (mean={res.mean():.2f})")

                                                                                                                        plt.tight_layout()
                                                                                                                        plt.savefig(args.out, dpi=130)
                                                                                                                        print(f"\nSaved {args.out}")
