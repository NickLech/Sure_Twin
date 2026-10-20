# train.py

from basicts.models.STID import STID, STIDConfig
from basicts.configs import BasicTSForecastingConfig
from basicts import BasicTSLauncher
from basicts.runners.callback import EarlyStopping

def main():
    INPUT_LEN = 144
    OUTPUT_LEN = 12
    
    model_config = STIDConfig(
        input_len=INPUT_LEN,
        output_len=OUTPUT_LEN,
        num_features=2500,
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
        gpus=None,
        batch_size=32,
        optimizer_params={"lr": 0.001, "weight_decay": 0.0005},
        seed=42,
        num_epochs=60,
        num_layers=3,
        callbacks=[EarlyStopping(patience=20)],
    )

    # 3. Launch training
    BasicTSLauncher.launch_training(cfg)

if __name__ == "__main__":
    main()
