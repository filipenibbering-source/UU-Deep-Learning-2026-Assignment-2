import lightning as L
# from src.vq_vae import VQVAE
from src.auto_encoder import AutoEncoder
from src.data import DataSet, DataSetType
from src.data import VQVAE_DataSet
from src.config import Config


from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from torch.utils.data import DataLoader

from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, ModelSummary, RichProgressBar, StochasticWeightAveraging

def train_vae(model: AutoEncoder, config: Config):

    train_dataset = VQVAE_DataSet(DataSetType.INTRA, 'train')
    train_dataset.load()
    
    val_dataset = VQVAE_DataSet(DataSetType.INTRA, 'test')
    val_dataset.load()
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers) # type: ignore
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers) # type: ignore

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        val_check_interval=config.val_check_interval,
        # logger=L.loggers.TensorBoardLogger("logs"),
        logger=TensorBoardLogger(save_dir="logs"),
        enable_checkpointing=config.enable_checkpointing,
        enable_progress_bar=config.enable_progress_bar,
        enable_model_summary=config.enable_model_summary,
        # enable_sanity_check=True,
        accelerator=config.accelerator,
        log_every_n_steps=config.log_every_n_steps,
        callbacks=[
            ModelSummary(max_depth=3),
            EarlyStopping(monitor='val_loss', patience=config.early_stopping_patience, mode='max', min_delta=config.early_stopping_min_delta, check_on_train_epoch_end=False),
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(monitor='val_loss', mode='max', save_top_k=1, save_last=True),
            RichProgressBar(),
            StochasticWeightAveraging(swa_lrs=config.stochastic_weight_averaging_swa_lrs, swa_epoch_start=config.stochastic_weight_averaging_swa_epoch_start)
        ],
        precision=config.precision, # type: ignore
    )
    trainer.fit(model, train_dataloader, val_dataloader)