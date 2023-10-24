"""
Author: Tu T .Do
Email: tu.dothanh1906@gmail.com
"""

import logging
import os


# --------------------------------------------------
# HELPER FUNCTION
def clean_name(name: str) -> str:
    """
    clean name to a file name
    """
    name = name.lower()
    name = name.split(" ")
    name = "-".join(name)

    return name


# --------------------------------------------------
# HELPER FUNCTION

def build_logger(name: str, log_path: str):
    """
    Return a logger with name -> logpath
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Check if logs folder exists
    if not os.path.isdir(log_path):
        os.mkdir(log_path)

    # Formater
    formatter = logging.Formatter(
        "[%(name)s] %(asctime)s :: %(message)s"
    )

    # FileHandler
    fh = logging.FileHandler(f"{log_path}/{clean_name(name)}.txt")
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # Streamhandler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.DEBUG)
    logger.addHandler(sh)

    return logger
