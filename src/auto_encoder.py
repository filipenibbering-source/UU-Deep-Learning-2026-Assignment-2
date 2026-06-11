"""
The idea is to use an autoencoder to learn good embeddings for each of the slices of the data.
"""

import torch
import torch.nn as nn
import lightning as L



from src.config import Config


class AutoEncoder(L.LightningModule):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, config: Config):
        super(AutoEncoder, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
        self.config = config
        
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.num_train_steps = 0
        self.num_val_steps = 0
        self.cosine_embedding_loss = nn.CosineEmbeddingLoss()
        
    def reset_losses(self):
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.num_train_steps = 0
        self.num_val_steps = 0
        
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

    @staticmethod
    def _expand_labels(labels: torch.Tensor, num_embeddings: int, device: torch.device) -> torch.Tensor:
        labels = labels.reshape(-1).to(device)
        if labels.numel() == num_embeddings:
            return labels

        if num_embeddings % labels.numel() != 0:
            raise ValueError(
                f"Cannot align {labels.numel()} labels with {num_embeddings} embeddings."
            )

        repeats = num_embeddings // labels.numel()
        return labels.repeat_interleave(repeats)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        
        loss = 0.0
        
        embeddings = self.encode(x)
        reconstruction = self.decode(embeddings)
        
        # 1. reconstruction loss
        loss_mse = nn.functional.mse_loss(reconstruction, x)
        
        # 2. contrastive loss (vectorized; no Python loop)
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        labels = self._expand_labels(y, embeddings.size(0), embeddings.device)

        permutation = torch.randperm(embeddings.size(0), device=embeddings.device)
        targets = torch.where(labels == labels[permutation], 1.0, -1.0)
        loss_contrastive = self.cosine_embedding_loss(embeddings, embeddings[permutation], targets)
        
        self.log("train_loss_mse", loss_mse)
        self.log("train_loss_contrastive", loss_contrastive)

        loss = self.config.weight_mse * loss_mse + self.config.weight_contrastive * loss_contrastive
        self.train_loss += loss
        self.num_train_steps += 1
        return loss
        
    def val_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        
        loss = 0.0
        
        embeddings = self.encode(x)
        reconstruction = self.decode(embeddings)
        
        # 1. reconstruction loss
        loss_mse = nn.functional.mse_loss(reconstruction, x)
        
        # 2. contrastive loss (vectorized; no Python loop)
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        labels = self._expand_labels(y, embeddings.size(0), embeddings.device)

        permutation = torch.randperm(embeddings.size(0), device=embeddings.device)
        targets = torch.where(labels == labels[permutation], 1.0, -1.0)
        loss_contrastive = self.cosine_embedding_loss(embeddings, embeddings[permutation], targets)
        
        self.log("val_loss_mse", loss_mse)
        self.log("val_loss_contrastive", loss_contrastive)

        loss = self.config.weight_mse * loss_mse + self.config.weight_contrastive * loss_contrastive
        
        self.val_loss += loss
        self.num_val_steps += 1
        return loss
    
    def on_train_epoch_end(self):
        assert (self.train_loss is not None) and (self.num_train_steps is not None), "train_loss and num_train_steps must be set"
        train_loss = self.train_loss / self.num_train_steps
        self.log("train_loss", train_loss)
        # self.reset_losses()
    
    def on_val_epoch_end(self):
        # assert (self.train_loss is not None) and (self.num_train_steps is not None), "train_loss and num_train_steps must be set"
        assert (self.val_loss is not None) and (self.num_val_steps is not None), "val_loss and num_val_steps must be set"
        
        # train_loss = self.train_loss / self.num_train_steps
        val_loss = self.val_loss / self.num_val_steps
        
        # self.log("train_loss", train_loss)
        self.log("val_loss", val_loss)
        
        self.reset_losses()
        
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        return optimizer