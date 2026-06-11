from enum import Enum
import h5py
from glob import glob
import os

import re
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import re

def get_dataset_name(filenamewithdir):
    # print('Filename with directory:', filenamewithdir)
    return re.findall(r'/([a-zA-Z0-9_]+)_[0-9]*.h5', filenamewithdir)[0]



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
    def __init__(self, dataset_type: DataSetType, split: str = 'train'):
        super().__init__()
        self.dataset_type = dataset_type
        self.split = split
        
    def load(self):
        
        self.dataset_base = ""
        match self.dataset_type:
            case DataSetType.CROSS:
                self.dataset_base = "Cross"
            case DataSetType.INTRA:
                self.dataset_base = "Intra"
            case _:
                raise ValueError(f"Invalid dataset type: {self.dataset_type}")
                
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
        
        self.mean = None
        self.std = None
        
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
            x = torch.from_numpy((matrix - self.mean) / self.std).float().reshape(-1, 248)
            
            return x, y