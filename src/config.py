import dataclasses
from typing import Literal


@dataclasses.dataclass
class Config:
    
    # training
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_workers: int = 4
    
    max_epochs: int = 10
    val_check_interval: float = 1.0
    enable_checkpointing: bool = True
    enable_progress_bar: bool = True
    enable_model_summary: bool = True

    accelerator: str = 'cpu'
    log_every_n_steps: int = 10
    precision = "16-mixed"

    # callbacks
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-3

    stochastic_weight_averaging_swa_lrs: float = 1e-3
    stochastic_weight_averaging_swa_epoch_start: int = 10

    # contrastive loss
    weight_mse: float = 1.0
    weight_contrastive: float = 1.0
    
    
    ### MODELS ###
    
    ## CNN ##
    
    num_features: int = 248
    base_channels: int = 128
    latent_dim: int = 64