# train.py

from basicts.models.STID import STID, STIDConfig
from basicts.configs import BasicTSForecastingConfig
from basicts import BasicTSLauncher

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
        num_epochs=2,
    )

    # 3. Launch training
    BasicTSLauncher.launch_training(cfg)

if __name__ == "__main__":
    main()
