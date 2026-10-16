# train.py

from basicts.models.STID import STID, STIDConfig
from basicts.configs import BasicTSForecastingConfig
from basicts import BasicTSLauncher

def main():

    # 1. Configure the model
    model_config = STIDConfig(input_len=336, output_len=336, num_features=7)

    # 2. Configure the task
    cfg = BasicTSForecastingConfig(
        model=STID,
        model_config=model_config,
        dataset_name="ETTh1",
        gpus=None,
        num_epochs=2
    )

    # 3. Launch training
    BasicTSLauncher.launch_training(cfg)

if __name__ == "__main__":
    main()
