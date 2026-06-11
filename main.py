from src.data import DataSet, DataSetType
# from src.vq_vae import VQVAE
from src.train import train_vae

from src.config import Config
from src.auto_encoder import AutoEncoder
from src.models.cnn import ResNet1DEncoder, ResNet1DDecoder


def main():
    # model = VQVAE(input_size=248, num_layers=3, hidden_dim=64, num_pretrain_steps=20)
    config = Config(
        
    )

    print('Config:', config)
    encoder = ResNet1DEncoder(input_features=config.num_features, base_channels=config.base_channels, latent_dim=config.latent_dim)
    decoder = ResNet1DDecoder(output_features=config.num_features, base_channels=config.base_channels, latent_dim=config.latent_dim)
    model = AutoEncoder(encoder=encoder, decoder=decoder, config=config)

    train_vae(model, config)

if __name__ == "__main__":
    main()
