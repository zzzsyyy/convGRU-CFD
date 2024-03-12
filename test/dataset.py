# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

def load_naca(dir):
    path = os.path.join(dir, 'data_')
    # u = np.load(f'{path}u.npy')
    # v = np.load(f'{path}v.npy')
    # print(u.shape)
    # print(v.shape)
    # return(u, v)
    ru = np.load(f'{path}u.npy')
    print("ru shape:", ru.shape)
    return ru

def split_data(data, is_train):
    if is_train:
        train_data = data[:160,:,:]
        return train_data
    else:
        valid_data = data[160:234,:,:]
        return valid_data

class CustomDataset(Dataset):
    def __init__(self, dir, is_train, n_frames_input, n_frames_output):
        super().__init__()
        self.datas = load_naca(dir)
        self.num_frames_input = n_frames_input
        self.num_frames_output = n_frames_output
        self.num_frames = n_frames_input + n_frames_output
        self.datas = split_data(self.datas, is_train)
        print("self: ", self.datas.shape)
        print('Loaded {} samples ({})'.format(self.__len__(), 'train' if is_train else 'valid'))
    def __getitem__(self, idx):
        # return super().__getitem__(idx)
        data = self.datas[idx*self.num_frames:(idx+1)*self.num_frames]
        inputs = data[:self.num_frames_input]
        targets = data[self.num_frames_input:]
        inputs = inputs[..., np.newaxis]
        targets = targets[..., np.newaxis]
        inputs[inputs < 0] = 0.0
        targets[targets < 0] = 0.0
        inputs = torch.from_numpy(inputs).permute(0, 3, 1, 2).float().contiguous()
        targets = torch.from_numpy(targets).permute(0, 3, 1, 2).float().contiguous()
        print(idx, targets.shape, inputs.shape)
        return idx, targets, inputs
    def __len__(self):
        return self.datas.shape[0] // self.num_frames

def create_dataloader():
    dataset = CustomDataset(is_train=True,
                        dir='../',
                        n_frames_input=5,
                        n_frames_output=5)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
    return dataloader
