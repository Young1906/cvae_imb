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


class MCMCConfig(BaseModel):
    dataset: str
    
    sampler_d: str
    max_iter: int
    step_size: float

    classifier: str
    score_avg_method: str
    result_pth: str


class LoggerConfig(BaseModel):
    """
    Parsing logger config
    """
    logger_name: str
    logger_dir: str


class BaselineConfig(BaseModel):
    dataset: str
    sampler_name: str
    classifier: str
    score_avg_method: str
    result_pth: str


class MCMCExpConfig(BaseModel):
    sampler_d: str
    n_samples: int
    ratio: int 
    max_iter: int
    step_size: float

    classifier: str
    score_avg_method: str
    result_pth: str

