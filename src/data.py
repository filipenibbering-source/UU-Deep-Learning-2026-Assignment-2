from enum import Enum
import h5py
from glob import glob
import os

import re
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import re
from typing import Literal
import numpy as np
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import yaml
from src.config import Config
from src.models.lstm import Encoder, Decoder
from src.auto_encoder import AutoEncoder

def get_dataset_name(filenamewithdir):
    # print('Filename with directory:', filenamewithdir)
    return re.findall(r'/([a-zA-Z0-9_]+)_[0-9]*.h5', filenamewithdir)[0]

def extend_from_config(config: Config, config_file: str):
    print('Extending config from', config_file)
    
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Key {key} not found in config")


class DataSetType(Enum):
    CROSS = "Cross"
    INTRA = "Intra"
    
TASK_TO_LABEL = {
    "rest": 0,
    "task_motor": 1,
    "task_story_math": 2,
    "task_working_memory": 3,
}

def get_task_label(file_name: str) -> int:
    for task in TASK_TO_LABEL.keys():
        if task in file_name:
            return TASK_TO_LABEL[task]
    raise ValueError(f"Invalid task in file name: {file_name}")


class DataSet(Dataset):
    def __init__(self, dataset_type: DataSetType, split: str = 'train', data_transform: Literal['raw', 'fft', 'mel'] = 'raw', prefix: str = '', *args, **kwargs):
        super().__init__()
        self.dataset_type = dataset_type
        self.split = split
        self.data_transform = data_transform
        
        self.prefix = prefix
        
        self.mean = 0.0
        self.std = 1.0
        self.max_values = np.ones(248)
        
        
        self.args = args
        self.kwargs = kwargs
        
        
    def load(self):
        
        self.dataset_base = ""
        match self.dataset_type:
            case DataSetType.CROSS:
                self.dataset_base = "Cross"
            case DataSetType.INTRA:
                self.dataset_base = "Intra"
            case _:
                raise ValueError(f"Invalid dataset type: {self.dataset_type}")
                
        if self.prefix.strip() != '':
            filenamepath = f"{self.prefix}/data/{self.dataset_base}/{self.split}"
        else:
            filenamepath = f"data/{self.dataset_base}/{self.split}"
        
        all_files = glob(os.path.join(filenamepath, "*.h5"))
        print('Found', len(all_files), 'files in folder', filenamepath)
        file_path = all_files[0]
        print('Opening file:', file_path)

        with h5py.File(file_path, 'r') as f:
            datasetname = get_dataset_name(file_path.replace('data/', ''))
            print('Dataset name:', datasetname)
            
            print(f.keys())
            matrix = f.get(datasetname)[()]
            print(type(matrix))
            print(matrix.shape)
            
    def __len__(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def __getitem__(self, index): 
        raise NotImplementedError("Subclasses must implement this method")
            
            
class VQVAE_DataSet(DataSet):
    def __init__(self, dataset_type: DataSetType, split: str = 'train'):
        super().__init__(dataset_type, split)
        
        # self.mean = 0.0
        # self.std = None
        
        self.files = []
        
    def load(self):
        
        self.dataset_base = ""
        match self.dataset_type:
            case DataSetType.CROSS:
                self.dataset_base = "Cross"
            case DataSetType.INTRA:
                self.dataset_base = "Intra"
            case _:
                raise ValueError(f"Invalid dataset type: {self.dataset_type}")
                
        
        if self.prefix.strip() != '':
            filenamepath = f"{self.prefix}/data/{self.dataset_base}/{self.split}"
        else:
            filenamepath = f"data/{self.dataset_base}/{self.split}"
        
        
        all_files = glob(os.path.join(filenamepath, "*.h5"))
        # print('Found', len(all_files), 'files in folder', filenamepath)
        # file_path = all_files[0]
        # print('Opening file:', file_path)
        
        for file_path in tqdm(all_files, desc='initializing dataset'):
            with h5py.File(file_path, 'r') as f:
                datasetname = get_dataset_name(file_path.replace('data/', ''))
                # print('Dataset name:', datasetname)
                
                matrix = f.get(datasetname)[()]
                
                if self.mean is None:
                    self.mean = matrix.mean(axis=0)
                else:
                    self.mean += matrix.mean(axis=0)
                    
                if self.std is None:
                    self.std = matrix.std(axis=0)
                else:
                    self.std += matrix.std(axis=0)
                    
        assert self.mean is not None and self.std is not None, "Mean and std must be initialized"
        self.mean /= len(all_files)
        self.std /= len(all_files)
        
        self.files = all_files
        
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        file_path = self.files[index]
        with h5py.File(file_path, 'r') as f:
            datasetname = get_dataset_name(file_path)
            matrix = f.get(datasetname)[()]
            
            y = get_task_label(file_path)
            x = torch.from_numpy((matrix - self.mean) / self.std).float().T#.reshape(-1, 248)
            
            return x, y


class SampleDataSet(DataSet):
    def __init__(self, dataset_type: DataSetType, split: str = 'train', data_transform: Literal['raw', 'fft', 'mel'] = 'raw', *args, **kwargs):
        super().__init__(dataset_type, split, data_transform, *args, **kwargs)
        
        self.mean = 0.0
        self.std = 1.0
        
        self.files = []
        
        self.load()
        
    def load(self):
        
        self.dataset_base = ""
        match self.dataset_type:
            case DataSetType.CROSS:
                self.dataset_base = "Cross"
            case DataSetType.INTRA:
                self.dataset_base = "Intra"
            case _:
                raise ValueError(f"Invalid dataset type: {self.dataset_type}")
                
        
        if self.prefix.strip() != '':
            filenamepath = f"{self.prefix}/data/{self.dataset_base}/{self.split}"
        else:
            filenamepath = f"data/{self.dataset_base}/{self.split}"
            
        all_files = glob(os.path.join(filenamepath, "*.h5"))
        # print('Found', len(all_files), 'files in folder', filenamepath)
        # file_path = all_files[0]
        # print('Opening file:', file_path)
        
        if self.split != 'train':
            print('WARNING: There is data leakage between train and test set. Please pass the mean and std from the train dataset to this dataset...')
        
        max_values = np.zeros(248)
        
        for file_path in tqdm(all_files, desc='initializing dataset'):
            with h5py.File(file_path, 'r') as f:
                datasetname = get_dataset_name(file_path.replace('data/', ''))
                # print('Dataset name:', datasetname)
                
                matrix = f.get(datasetname)[()]
                
                
                # print(matrix.shape)
                # print(matrix.mean(axis=1).shape)
                # exit()
                
                if self.mean is None:
                    self.mean = matrix.mean(axis=1)
                else:
                    self.mean += matrix.mean(axis=1)
                    
                if self.std is None:
                    self.std = matrix.std(axis=1)
                else:
                    self.std += matrix.std(axis=1)
                    
                max_values = np.maximum(max_values, matrix.max(axis=1)[0])
                
        self.max_values = max_values
                
        assert self.mean is not None and self.std is not None, "Mean and std must be initialized"
        self.mean /= len(all_files)
        self.std /= len(all_files)
        
        self.std = 1.0
        
        self.files = all_files
        
    def get_mean_and_std(self, dataset: DataSet) -> None:
        self.mean = dataset.mean
        self.std = dataset.std
        self.max_values = dataset.max_values
        
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        file_path = self.files[index]
        with h5py.File(file_path, 'r') as f:
            datasetname = get_dataset_name(file_path)
            matrix = f.get(datasetname)[()]
            
            matrix = matrix.T
            
            y = get_task_label(file_path)
            x = torch.from_numpy(((matrix - self.mean) / self.max_values)).float()#.reshape(-1, 248)
            
            return x, y
        
        
class MelSpectrogramDataSet(DataSet):
    
    def __init__(self, dataset_type: DataSetType, split: str = 'train', data_transform: Literal['raw', 'fft', 'mel'] = 'raw', *args, **kwargs):
        super().__init__(dataset_type, split, data_transform, *args, **kwargs)
        
        self.files = []
        
        self.std = 1.0
        self.mean = 0.0
        
        self.return_labels = False
        
        self.load()
        
    def load(self):
        self.dataset_base = ""
        match self.dataset_type:
            case DataSetType.CROSS:
                self.dataset_base = "cross"
            case DataSetType.INTRA:
                self.dataset_base = "intra"
            case _:
                raise ValueError(f"Invalid dataset type: {self.dataset_type}")
                
        if self.prefix.strip() != '':
            filenamepath = f"{self.prefix}/data/mel_spectograms_cache/{self.dataset_base}/{self.split}"
        else:
            filenamepath = f"data/mel_spectograms_cache/{self.dataset_base}/{self.split}"
            
        all_files = glob(os.path.join(filenamepath, "*.pt"))
        
        
        # if we are using the cross dataset, use the data from ALL datasets
        if self.dataset_type == DataSetType.CROSS and self.split == 'train':
            if self.prefix != '':
                all_files_intra = glob(os.path.join(self.prefix, 'data', 'mel_spectograms_cache', 'intra', 'train', '*.pt'))
            else:
                all_files_intra = glob(os.path.join('data', 'mel_spectograms_cache', 'intra', 'train', '*.pt'))
                
            all_files = all_files + all_files_intra  
            
        # if self.prefix.strip() != '':
        #     filenamepath = f"{self.prefix}/data/mel_spectograms_cache/{self.dataset_base}/{self.split}"
        # else:
        #     filenamepath = f"data/mel_spectograms_cache/{self.dataset_base}/{self.split}"
            
        # all_files = glob(os.path.join(filenamepath, "*.pt"))
        
        if self.split == 'train':
            for file_path in tqdm(all_files, desc='Computing mean and std'):
            # with h5py.File(file_path, 'r') as f:
                # datasetname = get_dataset_name(file_path.replace('data/', ''))
                # print('Dataset name:', datasetname)
                
                # matrix = f.get(datasetname)[()]
                
                
                # print(matrix.shape)
                # print(matrix.mean(axis=1).shape)
                # exit()
                
                file_path = file_path
                matrix = torch.load(file_path)
        
                if self.mean is None:
                    self.mean = matrix.mean()
                else:
                    self.mean += matrix.mean()
                    
                if self.std is None:
                    self.std = matrix.std()
                else:
                    self.std += matrix.std()
                    
            self.mean /= len(all_files)
            self.std /= len(all_files)
        
        print('Found', len(all_files), 'files in folder', filenamepath)
        
        self.files = all_files
        
    def __len__(self) -> int:
        return len(self.files)
    
    def get_mean_and_std(self, dataset: DataSet) -> None:
        self.mean = dataset.mean
        self.std = dataset.std
        
    
    def __getitem__(self, index) -> torch.Tensor:
        file_path = self.files[index]
        x = torch.load(file_path)
        # y = get_task_label(file_path)
        
        x = (x - self.mean) / self.std
        
        if self.return_labels:
            # print(file_path)
            file_index = int(re.findall(r'(\d+)\_\d+', file_path.split('/')[-1])[0])
            sensor_index = int(re.findall(r'\d+\_(\d+)', file_path.split('/')[-1])[0])
            
            return x, file_index, sensor_index # type: ignore
            
        return x
    
    

class ExtractedFeaturesDataSet(DataSet):
    def __init__(self, dataset_type: DataSetType, split: str = 'train', data_transform: Literal['raw', 'fft', 'mel'] = 'raw', *args, **kwargs):
        super().__init__(dataset_type, split, data_transform, *args, **kwargs)
        
        self.mean = 0.0
        self.std = 1.0
        self.mel_mean = 0.0
        self.mel_std = 1.0
        
        self.max_values = np.ones(64)
        
        self.files = []
        
        
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
        
        
        self.load()
        
    def load(self):
        
        self.dataset_base = ""
        match self.dataset_type:
            case DataSetType.CROSS:
                self.dataset_base = "Cross"
            case DataSetType.INTRA:
                self.dataset_base = "Intra"
            case _:
                raise ValueError(f"Invalid dataset type: {self.dataset_type}")
                
        
        if self.prefix.strip() != '':
            filenamepath = f"{self.prefix}/data/{self.dataset_base}/{self.split}"
        else:
            filenamepath = f"data/{self.dataset_base}/{self.split}"
            
        all_files = glob(os.path.join(filenamepath, "*.h5"))
        
        
        print('WARNING: There is data leakage between train and test set. Please pass the mean and std from the train dataset to this dataset...')
        
        self.files = all_files
        
        # load the encoder
        config = Config()
        if self.dataset_base == "Cross":
            # extend_from_config(config, '/Users/felix/UU-Deep-Learning-2026-Assignment-2/configs/Cross/lstm/lstm_cross_128_true_1.yaml')
            extend_from_config(config, '/Users/felix/UU-Deep-Learning-2026-Assignment-2/configs/Cross/lstm/lstm_cross_8_false_1.yaml')
        else:
            extend_from_config(config, '/Users/felix/UU-Deep-Learning-2026-Assignment-2/configs/Intra/lstm/lstm_128_true_1.yaml')

        encoder = Encoder(input_size=config.num_features, hidden_size=config.hidden_size, num_layers=config.num_layers, dropout=config.dropout, bidirectional=config.bidirectional)
        decoder = Decoder(output_size=config.num_features, hidden_size=config.hidden_size, num_layers=config.num_layers, dropout=config.dropout, bidirectional=config.bidirectional)
            
        model = AutoEncoder(encoder=encoder, decoder=decoder, config=config)
        # load the checkpoint
        if self.dataset_base == "Cross":
            # model.load_state_dict(torch.load(f'/Users/felix/UU-Deep-Learning-2026-Assignment-2/logs/lstm_cross_128_true_1/version_0/checkpoints/last.ckpt')['state_dict'])
            model.load_state_dict(torch.load(f'/Users/felix/UU-Deep-Learning-2026-Assignment-2/logs/lstm_cross_8_false_1/version_0/checkpoints/last.ckpt')['state_dict'])
        else:
            model.load_state_dict(torch.load(f'/Users/felix/UU-Deep-Learning-2026-Assignment-2/logs/lstm_128_true_1/version_0/checkpoints/last.ckpt')['state_dict'])
        
        self.model = model.eval()
        self.encoder = model.encoder.eval()
        
    def get_mean_and_std(self, dataset: DataSet) -> None:
        self.mean = dataset.mean
        self.std = dataset.std
        # self.max_values = dataset.max_values
        
    def get_max_values(self, dataset: DataSet) -> None:
        self.max_values = dataset.max_values
        
    def from_sample_dataset(self, dataset: SampleDataSet):
        self.max_values = dataset.max_values
        self.mean = dataset.mean
        
    def from_same_dataset(self, dataset: "ExtractedFeaturesDataSet") -> None:
        self.mean = dataset.mean
        self.std = dataset.std
        self.max_values = dataset.max_values
        self.mel_mean = dataset.mel_mean
        self.mel_std = dataset.mel_std
        
    def get_mel_mean_and_std(self, dataset: DataSet) -> None:
        self.mel_mean = dataset.mean
        self.mel_std = dataset.std
        
        
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        file_path = self.files[index]
        
        with h5py.File(file_path, 'r') as f:
            datasetname = get_dataset_name(file_path)
            matrix = f.get(datasetname)[()]
            
            matrix = matrix.T
            
            # print('1. matrix shape:', matrix.shape)
            
            y = get_task_label(file_path)
            # print(type(matrix), type(self.mean), type(self.max_values))
            matrix = (((matrix - self.mean) / self.max_values))#.float()#.reshape(-1, 248)
            
            # print('2. x shape:', matrix.shape)
            
            embeddings = []
            for i in range(matrix.shape[1]):
                inp = matrix[:, i].reshape(-1)
                inp = torch.from_numpy(inp).float()
                # print('2,5. inp shape:', inp.shape)
                # print(type(self.mel_transform))
                mel_spectrogram = self.mel_transform(inp)
                # print('3. mel_spectrogram shape:', mel_spectrogram.shape)
                mel_spectrogram = self.db_transform(mel_spectrogram).T
                # print('4. mel_spectrogram shape:', mel_spectrogram.shape)
                
                # normalize the mel spectrogram
                mel_spectrogram = (mel_spectrogram - self.mel_mean) / self.mel_std
                
                with torch.no_grad():
                    embedding = self.model.encoder(mel_spectrogram.unsqueeze(0))
                
                # print(embedding.shape)
                # exit()
                
                embeddings.append(embedding[:, -1, :])
                # print(embeddings[-1].shape)
                
            embeddings = torch.cat(embeddings, dim=0)
            embeddings = embeddings.reshape(-1)
            # print(embeddings.shape)
            
            
            return embeddings, y
        