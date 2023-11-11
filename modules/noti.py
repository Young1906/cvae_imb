import click
from modules.bot import send_message 


@click.command()
@click.option("--msg", "-M", type=str, help="message")
def main(msg: str):
    send_message(msg)

if __name__ == "__main__": main()

