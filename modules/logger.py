"""
Author: Tu T .Do
Email: tu.dothanh1906@gmail.com
"""

import logging
import os

from modules.bot import send_message
from pydantic import BaseModel
from modules.base import LoggerConfig


# --------------------------------------------------
# HELPER FUNCTION
def clean_name(name: str) -> str:
    name = name.lower()
    name = name.split(" ")
    name = "-".join(name)

    return name


# --------------------------------------------------
# LOGGER


# Send notification to telegram
class TelegramHandler(logging.Handler):
    def __init__(self): super().__init__()

    def emit(self, record):
        msg = self.format(record)
        try:
            send_message(msg)

        except (KeyboardInterrupt, SystemExit):
            raise

        except Exception as e:
            self.handleError(e)


def build_logger(name: str, log_path: str, telegram_handler: bool):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Check if logs folder exists
    if not os.path.isdir(log_path):
        os.mkdir(log_path)

    # Formater
    formatter = logging.Formatter(
        "%(asctime)s :: [%(name)s] - %(message)s"
    )

    # FileHandler
    fh = logging.FileHandler("{pth}/{fn_name}.txt".format(
        pth=log_path,
        fn_name=clean_name(name)
    ))
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # Streamhandler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.DEBUG)
    logger.addHandler(sh)

    # TelegramHandler
    if telegram_handler == 1:
        th = TelegramHandler()
        th.setFormatter(formatter)
        th.setLevel(logging.INFO)
        logger.addHandler(th)

    return logger
