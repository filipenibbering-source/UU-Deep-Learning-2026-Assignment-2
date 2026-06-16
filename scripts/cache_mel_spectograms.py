import sys
sys.path.append('..')

from src.auto_encoder import AutoEncoder
from src.config import Config
from src.data import DataSet, DataSetType, SampleDataSet, DataSetType
from pathlib import Path

from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from tqdm import tqdm
import torch


def cache_split(dataset: SampleDataSet, d_type: DataSetType, split: str, mel_transform: MelSpectrogram, db_transform: AmplitudeToDB, path_save: Path):
    
    match(d_type):
        case DataSetType.INTRA:
            
            save_path = path_save / 'intra' / split
            save_path.mkdir(parents=True, exist_ok=True)
            
        case DataSetType.CROSS:
            
            
            save_path = path_save / 'cross' / split
            save_path.mkdir(parents=True, exist_ok=True)
            
        case _:
            raise ValueError(f"Invalid dataset type: {d_type}")

    print('Caching TRAIN dataset...')
    # dataset = SampleDataSet(dataset_type=d_type, split=spli, data_transform='raw', prefix='..')
    
    # path_save_train = path_save / 'train'
    # path_save_train.mkdir(parents=True, exist_ok=True)
    
    for i in tqdm(range(len(dataset)), desc=f'Caching {split.upper()} dataset', total=len(dataset)):
        x, y = dataset[i]
        
        # print(x)
        for j in range(x.shape[1]):
            x_transformed = mel_transform(dataset[i][0][:, j])
            x_transformed = db_transform(x_transformed)
            
            torch.save(x_transformed, save_path / f'{i}_{j}.pt')
            
    
def main(save_path: str):
    
    
    mel_transform = MelSpectrogram(
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

    db_transform = AmplitudeToDB(stype='power', top_db=80.0)
    
    path_save = Path(save_path)
    path_save.mkdir(parents=True, exist_ok=True)
    
    print("######### Caching INTRA dataset... #########")
    
    print('Caching TRAIN dataset...')
    dataset_train = SampleDataSet(dataset_type=DataSetType.INTRA, split='train', data_transform='raw', prefix='..')
    
    cache_split(dataset_train, DataSetType.INTRA, 'train', mel_transform, db_transform, path_save)
    
    print('Caching TEST dataset...')
    dataset_test = SampleDataSet(dataset_type=DataSetType.INTRA, split='test', data_transform='raw', prefix='..')
    dataset_test.get_mean_and_std(dataset_train)
    
    cache_split(dataset_test, DataSetType.INTRA, 'test', mel_transform, db_transform, path_save)
    
    print('######### Caching CROSS dataset... #########')
    
    print('Caching TRAIN dataset...')
    dataset_train = SampleDataSet(dataset_type=DataSetType.CROSS, split='train', data_transform='raw', prefix='..')
    
    cache_split(dataset_train, DataSetType.CROSS, 'train', mel_transform, db_transform, path_save)
    
    print('Caching TEST1 dataset...')
    dataset_test = SampleDataSet(dataset_type=DataSetType.CROSS, split='test1', data_transform='raw', prefix='..')
    dataset_test.get_mean_and_std(dataset_train)
    
    cache_split(dataset_train, DataSetType.CROSS, 'test1', mel_transform, db_transform, path_save)
    
    print('Caching TEST2 dataset...')
    dataset_test = SampleDataSet(dataset_type=DataSetType.CROSS, split='test2', data_transform='raw', prefix='..')
    dataset_test.get_mean_and_std(dataset_train)
    
    cache_split(dataset_train, DataSetType.CROSS, 'test2', mel_transform, db_transform, path_save)
    
    print('Caching TEST3 dataset...')
    dataset_test = SampleDataSet(dataset_type=DataSetType.CROSS, split='test3', data_transform='raw', prefix='..')
    dataset_test.get_mean_and_std(dataset_train)
    
    cache_split(dataset_train, DataSetType.CROSS, 'test3', mel_transform, db_transform, path_save)
    
    
    
    # path_save_train = path_save / 'train'
    # path_save_train.mkdir(parents=True, exist_ok=True)
    
    # for i in tqdm(range(len(dataset)), desc='Caching TRAIN dataset', total=len(dataset)):
    #     x, y = dataset[i]
        
    #     # print(x)
    #     for j in range(x.shape[1]):
            
    #         x_transformed = mel_transform(dataset[0][0][:, j])
    #         x_transformed = db_transform(x_transformed)
            
    #         torch.save(x_transformed, path_save_train / f'{i}_{j}.pt')
            
    # print('Caching TEST dataset...')
    # dataset_test = SampleDataSet(dataset_type=DataSetType.INTRA, split='test', data_transform='raw', prefix='..')
    
            
            
if __name__ == '__main__':
    main(save_path='../data/mel_spectograms_cache')