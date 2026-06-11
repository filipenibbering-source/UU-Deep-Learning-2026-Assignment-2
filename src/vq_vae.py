import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from sklearn.cluster import KMeans

class VectorQuantizeEMA(nn.Module):
    """
    Exponential Moving Average (EMA) vector quantization for VQ-VAE model
    
    Module taken from https://github.com/AndrewBoessen/VQ-VAE/blob/main/vqvae.py#L6
    """

    def __init__(
        self, n_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5
    ):
        """
        initialize VQ class

        :param n_embeddings number: number of discrete embeddings
        :param embedding_dim number: dimension of embeddings
        :param commitment_cost number: commitment cost weight
        :param decay number: decay rate for EMA
        :param epsilon number: epsilon value for EMA
        """
        super(VectorQuantizeEMA, self).__init__()

        self._embedding_dim = embedding_dim  # Dimension of an embedding vector, D
        self._n_embeddings = n_embeddings  # Number of categories in distribution, K

        # Parameters
        self._embedding = nn.Embedding(
            self._n_embeddings, self._embedding_dim
        )  # Embedding table for categorical distribution
        self._embedding.weight.data.normal_()  # Randomly initialize embeddings
        self.register_buffer(
            "_ema_cluster_size", torch.zeros(n_embeddings)
        )  # Clusters for EMA
        self._ema_w = nn.Parameter(
            torch.Tensor(n_embeddings, self._embedding_dim)
        )  # EMA weights
        self._ema_w.data.normal_()

        # Loss / Training Parameters
        self._commitment_cost = commitment_cost
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, z_e):
        """
        Quantize embeddings

        :param z_e numpy.ndarray: Embeddings from encoder to quantize
        """
        # reshape from BCHW -> BHWC
        # z_e = z_e.permute(0, 2, 3, 1).contiguous()
        shape = z_e.shape

        # Flatten input embeddings
        flat_z_e = z_e.view(-1, self._embedding_dim)

        # Claculate Distances
        # ||z_e||^2 + ||e||^2 - 2 * z_q
        distances = (
            torch.sum(flat_z_e**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_z_e, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._n_embeddings, device=z_e.device
        )
        # Convert to shape of embeddings
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and Unflatten
        z_q = torch.matmul(encodings, self._embedding.weight).view(shape)

        # Update weights with EMA
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._n_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_z_e)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        # Loss
        e_latent_loss = F.mse_loss(
            z_q.detach(), z_e
        )  # distance from encoder output and quantized embeddings
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Loss
        z_q = z_e + (z_q - z_e).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Convert shape back to BCHW
        # return loss, z_q.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
        return loss, z_q, perplexity, encodings


# class VQVAE(nn.Module):
class VQVAE(L.LightningModule):
    def __init__(self, input_size: int, num_layers: int, hidden_dim: int, num_pretrain_steps: int = 2_500):
        super(VQVAE, self).__init__()
        
        assert input_size == 248, "Each slice of the data has 248 sensors..., expected 248 as input size"
        
        # test if a variational autoencoder performs better than regular autoencoder
        downscaling_factor = 1
        encoder_layers: list[nn.Module] = []
        decoder_layers: list[nn.Module] = []
        for _ in range(num_layers-1):
            encoder_layers.append(
                nn.Linear(input_size // downscaling_factor, input_size // (downscaling_factor * 2))
            )
            encoder_layers.append(nn.ReLU())
            
            decoder_layers.append(
                nn.Linear(input_size // (downscaling_factor * 2), input_size // downscaling_factor)
            )
            decoder_layers.append(nn.ReLU())
            downscaling_factor *= 2
            
        # self.mean = nn.Linear(input_size // downscaling_factor, hidden_dim)
        # self.variance = nn.Linear(input_size // downscaling_factor, hidden_dim)
        
        encoder_layers.append(nn.Linear(input_size // downscaling_factor, hidden_dim))
        decoder_layers.append(nn.Linear(hidden_dim, input_size // downscaling_factor))
        
        
        self._encoder = nn.Sequential(*encoder_layers)
        self._decoder = nn.Sequential(*decoder_layers[::-1])
        
        self._vq = VectorQuantizeEMA(
            n_embeddings=64, 
            embedding_dim=hidden_dim,
            commitment_cost=0.25,
            decay=0.99
        )
        
        self.num_pretrain_steps = num_pretrain_steps
        self.train_step_counter = 0
        
        self.num_embeddings = 64
        self.hidden_dim = hidden_dim
        
    def encode(self, x):
        """
        Encode image

        :param x numpy.ndarray: Input image
        """
        z = self._encoder(x)
        return z
    
    def vq(self, z):
        loss, z_q, perplexity, _ = self._vq(z)
        return loss, z_q, perplexity
    
    def pretrain(self, x):
        """
        Bypass vector quantize step for pretraining

        :param x numpy.ndarray: Input image
        """
        z = self._encoder(x)
        # z = self._pre_vq_conv(z)
        x_recon = self._decoder(z)
        return x_recon
    
    
    def forward(self, x):
        """
        Encode and reconstruct image

        :param x numpy.ndarray: Input image
        """
        z = self._encoder(x)  # encode image to latent
        # z = self._pre_vq_conv(z)
        # quantize encoding to dicrete space
        loss, z_q, perplexity, _ = self._vq(z)
        x_recon = self._decoder(z_q)  # reconstruction of input from decoder
        return loss, x_recon, perplexity
        
    
    def set_embeddings(self, new_embeddings):
        """
        Set discrete embeddings codebook params

        :param new_embeddings numpy.ndarray: Embedding codebook
        """
        with torch.no_grad():
            self._vq._embedding.weight.copy_(new_embeddings)

        
    def training_step(self, batch, batch_idx):
        
        X = batch
        
        # print(X.shape)
        

        if self.train_step_counter == self.num_pretrain_steps:
            # use k-means clustering to initialize the embeddings
            # K Means Cluster to initialize discrete embeddings
            with torch.no_grad():
                embeddings = self.encode(X)

            print('embeddings.shape:', embeddings.shape) # [35624, 64]
            # embeddings = (
            #     embeddings.permute(0, 2, 3, 1)
            #     .contiguous()
            #     .reshape(-1, self.hidden_dim)
            # )

            np_e = embeddings.cpu().detach().numpy()

            n_clusters = self.num_embeddings
            kmeans = KMeans(n_clusters)
            kmeans.fit(np_e)

            cluster_centers = torch.from_numpy(kmeans.cluster_centers_).to(X.device)

            self.set_embeddings(cluster_centers)

        if self.train_step_counter < self.num_pretrain_steps:
            data_recon = self.pretrain(X)
            loss = nn.functional.mse_loss(data_recon, X)
            recon_error = loss
            vq_loss = torch.Tensor([0.0])
            perplexity = torch.Tensor([0.0])

        else:
            vq_loss, data_recon, perplexity = self.forward(X)
            recon_error = nn.functional.mse_loss(data_recon, X)
            loss = recon_error + vq_loss
            
            
        self.train_step_counter += 1
        
        self.log("train_step_loss", loss)
        self.log("train_step_recon_error", recon_error)
        self.log("train_step_vq_loss", vq_loss)
        self.log("train_step_perplexity", perplexity)
        
        
        
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        X = batch
        
        # print(X.shape)

        if self.train_step_counter < self.num_pretrain_steps:
            data_recon = self.pretrain(X)
            loss = nn.functional.mse_loss(data_recon, X)
            recon_error = loss
            vq_loss = torch.Tensor([0.0])
            perplexity = torch.Tensor([0.0])

        else:
            vq_loss, data_recon, perplexity = self.forward(X)
            recon_error = nn.functional.mse_loss(data_recon, X)
            loss = recon_error + vq_loss

        self.log("val_loss", loss)
        self.log("val_recon_error", recon_error)
        self.log("val_vq_loss", vq_loss)
        self.log("val_perplexity", perplexity)
        
        return loss
            