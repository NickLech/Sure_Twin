# train_STID.py
import argparse

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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    args = ap.parse_args()

    model_config = STIDConfig(
        input_len=INPUT_LEN,
        output_len=OUTPUT_LEN,
        num_features=2500,
        input_hidden_size=args.hidden,
        num_layers=args.layers,  # was passed to the wrong config before (silently ignored)
        if_time_in_day=True,
        if_day_in_week=True,
        num_time_in_day=144,
        num_day_in_week=7,
    )

    cfg = BasicTSForecastingConfig(
        model=STID,
        model_config=model_config,
        dataset_name="MILAN",
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
        eval_horizons=[0, 5, 11],  # metrics at h=1,6,12
        save_results=True,  # saves prediction.npy / targets.npy for analysis
        ckpt_save_dir=f"checkpoints/STID_MILAN/seed{args.seed}",  # one dir per seed (ensemble members)
    )

    BasicTSLauncher.launch_training(cfg)


if __name__ == "__main__":
    main()
