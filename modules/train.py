import click
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from modules.cvae import LightCVAE
from modules.data import build_datamodules


@click.command()
@click.option("--exp_name", "-N", type=str, help="Name of the experiment")
@click.option("--dataset", "-D", type=str, help="Name of the dataset")
@click.option("--batch_size", "-B", type=int, help="Batch size")
@click.option("--num_workers", "-W", type=int, help="Num workers")
@click.option("--input_dim", "-I", type=int, help="Dimension of Input")
@click.option("--z_dim", "-Z", type=int, help="Dimension of latent variable")
@click.option("--n_class", "-C", type=int, help="Number of classes")
@click.option("--val_split", "-S", type=float, help="Number of classes")
@click.option("--max_epochs", "-E", type=int, help="Max epochs")
def main(
    exp_name: str,
    dataset: str,
    batch_size: int,
    num_workers: int,
    input_dim: int,
    z_dim: int,
    n_class: int,
    val_split: float,
    max_epochs: int,
):
    # build data modules
    dm = build_datamodules(dataset, val_split, batch_size, num_workers)

    # model
    cvae = LightCVAE(input_dim, "mlp_16_8_16", "mlp_16_8_16", z_dim, n_class)
    logger = TensorBoardLogger("logs", name=exp_name)
    trainer = L.Trainer(logger=logger, max_epochs=max_epochs)
    trainer.fit(cvae, dm)


if __name__ == "__main__":
    main()
