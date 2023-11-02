import click


from modules.base import LoggerConfig
from modules.base import OverSamplingConfig
from modules.mcmc.mcmc import MCMCOverSampling

@click.command()
@click.option("--config", type=str, help="Path to config file")
def main():
    config_pth = config

    with open(config, "r") as f:
        config = yaml.safe_load(f)

    os_config = OverSamplingConfig(**config["oversampling"])


    # dataset
    (X_train, y_train), (X_valid, y_valid), le = _build_Xy(os_config.dataset)
