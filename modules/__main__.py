import click
import lightning as L

from modules.cvae import LightCVAE
from modules.data import build_datamodules


@click.command()
@click.option("--exp_name", "-N", type=str, help="Name of the experiment")
@click.option("--dataset", "-D", type=str, help="Name of the dataset")
@click.option("--batch_size", "-B", type=int, help="Batch size")
@click.option("--num_workers", "-W", type=int, help="Num workers")

@click.option("--input_dim", "-I", type=int, help="Dimension of Input") 
@click.option("--dims", "-D", type=int, help="Dimension of Input") 
@click.option("--z_dim", "-Z", type=int, help="Dimension of latent variable") 
@click.option("--n_class", "-Z", type=int, help="Number of classes") 
def main(
        exp: str, 
        dataset: str,
        batch_size: int,
        num_workers: int,
        z_dims: int):

    # build data modules
    dm, le = build_datamodules(dataset, val_split, batch_size, num_workers)

    # model
    cvae = LightCVAE()
    return

if __name__ == "__main__": main()

