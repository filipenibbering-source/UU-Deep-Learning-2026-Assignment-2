from src.data import DataSet, DataSetType
# from src.vq_vae import VQVAE
from src.train import train_vae

from src.config import Config
from src.auto_encoder import AutoEncoder
from src.models.cnn import ResNet1DEncoder, ResNet1DDecoder
from src.models.lstm import LSTMEncoder, LSTMDecoder, Encoder, Decoder
from fire import Fire

from typing import Optional
import yaml

def extend_from_config(config: Config, config_file: str):
    
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Key {key} not found in config")

def main(config_file: Optional[str] = None):
    config = Config()
    if config_file is not None:
        extend_from_config(config, config_file)

    print('Config:', config)
    encoder = Encoder(input_size=config.num_features, hidden_size=config.hidden_size, num_layers=config.num_layers, dropout=config.dropout, bidirectional=config.bidirectional)
    decoder = Decoder(output_size=config.num_features, hidden_size=config.hidden_size, num_layers=config.num_layers, dropout=config.dropout, bidirectional=config.bidirectional)
    
    model = AutoEncoder(encoder=encoder, decoder=decoder, config=config)

    train_vae(model, config, run_name=config_file.split('/')[-1].split('.')[0] if config_file is not None else '')

if __name__ == "__main__":
    Fire(main)
