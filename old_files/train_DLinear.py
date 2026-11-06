# train.py
from basicts.models.DLinear import DLinear, DLinearConfig
from basicts.configs import BasicTSForecastingConfig
from basicts import BasicTSLauncher


def main():
    model_config = DLinearConfig(input_len=336, output_len=336)

    cfg = BasicTSForecastingConfig(
        model=DLinear,
        model_config=model_config,
        dataset_name="ETTh1",
        gpus=None,
        # batch_size=32,   # default, va bene per il test
        # TODO: numero di epoche (nome da trovare al passo 3), es. 2 o 3
        num_epochs = 2
    )

    BasicTSLauncher.launch_training(cfg)


if __name__ == "__main__":
    main()
