import dataclasses
from typing import Literal
# from src.data import DataSetType


@dataclasses.dataclass
class Config:
    
    ## Data
    
    data_type: Literal['cross', 'intra'] = 'intra'
    data_transform: Literal['raw', 'fft', 'mel'] = 'mel'
    # data_prefix: str = 'cross'
    
    
    ## training
    learning_rate: float = 1e-2
    batch_size: int = 128
    num_workers: int = 4
    
    max_epochs: int = 20
    val_check_interval: float = 1.0
    enable_checkpointing: bool = True
    enable_progress_bar: bool = True
    enable_model_summary: bool = False

    # accelerator: str = 'cpu'
    accelerator: str = 'gpu'
    log_every_n_steps: int = 1
    # precision = "16-mixed"
    precision = "32-true"

    # callbacks
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-3

    stochastic_weight_averaging_swa_lrs: float = 1e-3
    stochastic_weight_averaging_swa_epoch_start: int = 5

    # contrastive loss
    weight_mse: float = 1
    weight_contrastive: float = 1e-1
    # weight_contrastive: float = 0.0
    increase_contrastive_weight_every_epoch: bool = False  # whether to have a linear increase of the contrastive loss weight
    
    
    ### MODELS ###
    
    ## CNN ##
    
    model: Literal['lstm', 'resnet18', 'resnet50', 'eegnet'] = 'lstm'
    is_baseline: bool = False
    
    # num_features: int = 248
    base_channels: int = 128
    latent_dim: int = 64
    
    ## LSTM ##
    num_features: int = 64
    hidden_size: int = 8
    num_layers: int = 4
    dropout: float = 0.2
    bidirectional: bool = False