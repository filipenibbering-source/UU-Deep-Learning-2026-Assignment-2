"""
The idea is to use an autoencoder to learn good embeddings for each of the slices of the data.
"""

import torch
import torch.nn as nn
import lightning as L
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchmetrics import Accuracy



from src.config import Config


class BaselineModel(L.LightningModule):
    def __init__(self, model: nn.Module, config: Config):
        super(BaselineModel, self).__init__()
        
        self.model = model
        
        self.config = config
        
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.num_train_steps = 0
        self.num_val_steps = 0
        self.accuracy_train = Accuracy(task='multiclass', num_classes=4)
        self.accuracy_val = Accuracy(task='multiclass', num_classes=4)
        # self.cosine_embedding_loss = nn.CosineEmbeddingLoss()
        
        # self.weight_contrastive = config.weight_contrastive
        # self.increase_contrastive_weight_every_epoch = config.increase_contrastive_weight_every_epoch
        
        # if self.increase_contrastive_weight_every_epoch:
            # self.weight_contrastive = 0.0
            
        self.scheduler = None
        
        self.mel_transform = MelSpectrogram(
            sample_rate=2034,
            n_fft=256,
            n_mels=64,
            hop_length=128,
            power=2.0,
            f_min=0.0,
            f_max=300.0,
            # pad=3,
            # normalized=False,
            # center=True,
            # pad_mode='reflect',
            # onesided=True,
            # norm='slaney',
            # mel_scale='htk'
        )

        self.db_transform = AmplitudeToDB(stype='power', top_db=80.0)
        
    def reset_losses(self):
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.num_train_steps = 0
        self.num_val_steps = 0
        
    def forward(self, x):
        return self.model(x)
    
    def _transform_to_frequency_domain(self, x):
        """
        Written by Claude Opus, to to go a bit quicker
        """
        # x: (batch, channels, time). Returns (batch * channels, n_mels, frames),
        # matching the per-sample, per-channel caching in scripts/cache_mel_spectograms.py.
        batch, channels = x.shape[0], x.shape[1]

        # MelSpectrogram supports arbitrary leading dims, so transform the whole
        # batch at once instead of looping over samples/channels.
        x = self.mel_transform(x)  # (batch, channels, n_mels, frames)

        # AmplitudeToDB's top_db cutoff is computed per leading "batch" element.
        # The cached version applies it to each (n_mels, frames) slice on its own,
        # so reshape to (N, 1, n_mels, frames) to reproduce that exact per-slice cutoff.
        n_mels, frames = x.shape[-2], x.shape[-1]
        x = x.reshape(batch * channels, 1, n_mels, frames)
        x = self.db_transform(x)
        x = x.reshape(batch, channels, n_mels, frames)

        # Standardize per sample. The raw dB values are large in magnitude
        # (roughly [max - 80, max] dB), which saturates the LSTM gates and sits
        # far outside the input range the models expect, preventing learning.
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (std + 1e-5)

        return x
                
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
            
        x, y = batch
        # print(x.shape)
        
        x = x.transpose(1, 2)
        x = self._transform_to_frequency_domain(x)
        # x = x.transpose(1, 2)    
    
        prediction = self.model(x)
        # print(prediction)
        # print(prediction.shape)
        # print(prediction)
        
        loss = nn.functional.cross_entropy(prediction, y)
        self.log("step_train_loss_ce", loss, prog_bar=True)
        self.train_loss += loss # type: ignore
        self.num_train_steps += 1
        
        self.accuracy_train(prediction, y)
        self.log("step_train_accuracy", self.accuracy_train, prog_bar=False)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # if self.config.data_transform != 'mel':
        x, y = batch
        # else:
        #     x = batch
        #     y = None
            
            
        # print(x.shape)
        # print(x.shape)
        x = x.transpose(1, 2)
        # print(x.shape)
        
        x = self._transform_to_frequency_domain(x)
        
        prediction = self.model(x)
        # print(prediction)
        # print(prediction.shape)
        # print(prediction)
        
        loss = nn.functional.cross_entropy(prediction, y)
        
        self.accuracy_val(prediction, y)
        self.log("step_val_accuracy", self.accuracy_val, prog_bar=False)
            
        self.log("step_val_loss_ce", loss, prog_bar=True)
        # self.log("step_val_loss_contrastive", loss_contrastive)

        # if self.increase_contrastive_weight_every_epoch:
        #     loss = self.config.weight_mse * loss_mse + self.weight_contrastive * loss_contrastive
        # else:
        # loss = self.config.weight_mse * loss_mse + self.weight_contrastive * loss_contrastive
            
        
        self.val_loss += loss # type: ignore
        self.num_val_steps += 1
        return loss
    
    def on_train_epoch_end(self):
        assert (self.train_loss is not None) and (self.num_train_steps is not None), "train_loss and num_train_steps must be set"
        train_loss = self.train_loss / self.num_train_steps
        self.log("epoch_train_loss", train_loss)
        # self.reset_losses()
        
        self.num_train_steps = 0
        self.train_loss = 0.0
        
        # if self.increase_contrastive_weight_every_epoch:
        #     self.weight_contrastive += self.config.weight_contrastive / self.config.max_epochs
            
        if self.scheduler is not None:
            self.scheduler.step()
            
        self.log('accuracy_train_epoch', self.accuracy_train.compute(), prog_bar=False)
        self.accuracy_train.reset()
    
    def on_validation_epoch_end(self):
        # assert (self.train_loss is not None) and (self.num_train_steps is not None), "train_loss and num_train_steps must be set"
        assert (self.val_loss is not None) and (self.num_val_steps is not None), "val_loss and num_val_steps must be set"
        
        # train_loss = self.train_loss / self.num_train_steps
        val_loss = self.val_loss / self.num_val_steps
        
        # self.log("train_loss", train_loss)
        print("epoch_val_loss", val_loss)
        self.log("epoch_val_loss", val_loss)
        
        self.num_val_steps = 0
        self.val_loss = 0.0
        
        self.log('accuracy_val_epoch', self.accuracy_val.compute(), prog_bar=False)
        self.accuracy_val.reset()
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.max_epochs, eta_min=1e-6)
        
        return optimizer