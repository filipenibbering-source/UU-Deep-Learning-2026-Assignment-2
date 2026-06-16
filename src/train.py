import lightning as L
# from src.vq_vae import VQVAE
from src.auto_encoder import AutoEncoder
from src.data import DataSet, DataSetType
from src.data import VQVAE_DataSet, MelSpectrogramDataSet
from src.config import Config


from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from torch.utils.data import DataLoader
import os

from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, ModelSummary, RichProgressBar, StochasticWeightAveraging

def train_vae(model: AutoEncoder, config: Config, run_name: str = ''):

    # print('Got run name:', run_name)
    # exit()
    
    if run_name != '':
        if os.path.exists(os.path.join('logs', run_name)):
            print(f'Log folder {run_name} already exists. Skipping...')
            return

    # train_dataset = VQVAE_DataSet(DataSetType.INTRA, 'train')
    
    if config.data_type.lower().strip() == 'intra':
        train_dataset = MelSpectrogramDataSet(DataSetType.INTRA, 'train')
        
        val_dataset = MelSpectrogramDataSet(DataSetType.INTRA, 'test')
        val_dataset.get_mean_and_std(train_dataset)
    else:
        train_dataset = MelSpectrogramDataSet(DataSetType.CROSS, 'train')
        val_dataset = MelSpectrogramDataSet(DataSetType.CROSS, 'test1')
        val_dataset.get_mean_and_std(train_dataset)
        
    
    print('Got ', len(train_dataset), 'train samples and ', len(val_dataset), 'val samples')
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, persistent_workers=True) # type: ignore
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, persistent_workers=True) # type: ignore

    # print('Got ', len(train_dataloader), 'train batches and ', len(val_dataloader), 'val batches')
    
    # exit()

    trainer = L.Trainer(
        
        max_epochs=config.max_epochs,
        val_check_interval=config.val_check_interval,
        # logger=L.loggers.TensorBoardLogger("logs"),
        logger=TensorBoardLogger(save_dir="logs", name=run_name) if run_name != '' else TensorBoardLogger(save_dir="logs"),
        enable_checkpointing=config.enable_checkpointing,
        enable_progress_bar=config.enable_progress_bar,
        enable_model_summary=config.enable_model_summary,
        # enable_sanity_check=True,
        accelerator=config.accelerator,
        log_every_n_steps=config.log_every_n_steps,
        callbacks=[
            ModelSummary(max_depth=3),
            EarlyStopping(monitor='epoch_val_loss', patience=config.early_stopping_patience, mode='min', min_delta=config.early_stopping_min_delta, check_on_train_epoch_end=False),
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(monitor='epoch_val_loss', mode='min', save_top_k=1, save_last=True),
            # RichProgressBar(),
            StochasticWeightAveraging(swa_lrs=config.stochastic_weight_averaging_swa_lrs, swa_epoch_start=config.stochastic_weight_averaging_swa_epoch_start)
        ],
        precision=config.precision, # type: ignore
    )
    trainer.fit(model, train_dataloader, val_dataloader)