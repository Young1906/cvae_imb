"""
BaseModel
"""

from pydantic import BaseModel
from typing import NamedTuple


class Layer(NamedTuple):
    """
    Describe layer of a MLP
    """
    h: int
    a: str


class DatasetConfig(BaseModel):
    """
    Parsing dataset config
    """
    name: str
    batch_size: int
    num_workers: int
    val_split: float


class ModelConfig(BaseModel):
    """
    Parsing model config
    """
    input_dim: int
    encoder: list[Layer]
    decoder: list[Layer]
    z_dim: int
    n_class: int


class TrainingConfig(BaseModel):
    """
    Parsing training config
    """
    max_epochs: int
    exp_name: str
    checkpoint_pth: str
    checkpoint_fn: str
    checkpoint_monitor: str


class OverSamplingConfig(BaseModel):
    """
    Parsing experiment config
    """
    dataset: str
    classifier: str

    # cvae params
    checkpoint_pth: str
    checkpoint_fn: str
    input_dim: int
    encoder: list[Layer]
    decoder: list[Layer]
    z_dim: int
    n_class: int

    # evaluation metrics
    score_avg_method: str

    # Pth to result file
    result_pth: str


class LoggerConfig(BaseModel):
    """
    Parsing logger config
    """
    logger_name: str
    logger_dir: str
