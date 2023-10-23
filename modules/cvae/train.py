import click
import lightning as L
import yaml

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from modules.cvae.cvae import LightCVAE
from modules.cvae.data import build_datamodules
from modules.base import (
        DatasetConfig,
        ModelConfig,
        TrainingConfig
        )

L.seed_everything(1, workers=True)

@click.command()
@click.option("--config", type=str, help="Path to config file")
def main(config: str):
    with open(config, "r") as f:
        config = yaml.safe_load(f)

    # build data modules
    ds_config = DatasetConfig(**config['dataset'])
    dm = build_datamodules(
            ds_config.name,
            ds_config.val_split, 
            ds_config.batch_size,
            ds_config.num_workers)

    # model
    model_config = ModelConfig(**config['model'])
    cvae = LightCVAE(
            model_config.input_dim,
            model_config.encoder,
            model_config.decoder,
            model_config.z_dim,
            model_config.n_class)

    # training
    training_config = TrainingConfig(**config['training'])

    logger = TensorBoardLogger(
            "logs",
            name=training_config.exp_name)

    checkpointer = ModelCheckpoint(
            dirpath=training_config.checkpoint_pth,
            filename=training_config.checkpoint_fn,
            monitor=training_config.checkpoint_monitor,)

    trainer = L.Trainer(
            logger=logger,
            max_epochs=training_config.max_epochs,
            callbacks=[checkpointer, ])

    trainer.fit(cvae, dm)


if __name__ == "__main__": main()
