# eval_split.py
"""
Evaluates an EXISTING STID checkpoint (one seed) on the val or train split.
Needs the dataset view created with:  python make_split_view.py --split val

Usage (one process per seed):
    python eval_split.py --split val --seed 0
    for s in $(seq 0 9); do python eval_split.py --split val --seed $s; done

Use the same --hidden / --layers as in training.
Results go to checkpoints_eval/MILAN_<split>/seed<seed>/..., NOT into checkpoints/.
"""
import argparse
import glob
import sys

from torch.optim.lr_scheduler import CosineAnnealingLR

from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.models.STID import STID, STIDConfig
from basicts.runners.callback import EarlyStopping, GradientClipping

INPUT_LEN = 144
OUTPUT_LEN = 12
NUM_EPOCHS = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--ckpt_root", default="checkpoints/STID_MILAN")
    ap.add_argument("--out_root", default="checkpoints_eval")
    args = ap.parse_args()

    # same checkpoint that produced the test results: best validation MAE
    found = sorted(glob.glob(f"{args.ckpt_root}/seed{args.seed}/*/STID_best_val_MAE.pt"))
    if len(found) != 1:
        sys.exit(f"expected exactly one STID_best_val_MAE.pt for seed{args.seed}, found: {found}")
    ckpt_path = found[0]
    print(f"checkpoint: {ckpt_path}")

    model_config = STIDConfig(
        input_len=INPUT_LEN,
        output_len=OUTPUT_LEN,
        num_features=2500,
        input_hidden_size=args.hidden,
        num_layers=args.layers,
        if_time_in_day=True,
        if_day_in_week=True,
        num_time_in_day=144,
        num_day_in_week=7,
    )

    cfg = BasicTSForecastingConfig(
        model=STID,
        model_config=model_config,
        dataset_name=f"MILAN_{args.split}",   # view where this split is exposed as 'test'
        input_len=INPUT_LEN,
        output_len=OUTPUT_LEN,
        use_timestamps=True,
        gpus="0",
        batch_size=32,
        optimizer_params={"lr": 0.001, "weight_decay": 0.0005},
        lr_scheduler=CosineAnnealingLR,
        lr_scheduler_params={"T_max": NUM_EPOCHS, "eta_min": 1e-5},
        seed=args.seed,
        num_epochs=NUM_EPOCHS,
        callbacks=[EarlyStopping(patience=10), GradientClipping(max_norm=5.0)],
        eval_horizons=[0, 5, 11],
        save_results=True,
        ckpt_save_dir=f"{args.out_root}/MILAN_{args.split}/seed{args.seed}",  # separate from the training dir
    )

    BasicTSLauncher.launch_evaluation(cfg=cfg, ckpt_path=ckpt_path, gpus="0")


if __name__ == "__main__":
    main()
