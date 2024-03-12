import re

def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

from torch import nn
from collections import OrderedDict
import numpy as np
import os
import torch
import torch.utils.data as data


def make_layers(block):
    """
    Making layers using parameters from NetParams.py
    :param block: OrderedDict
    :return: layers
    """
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0],
                                                 out_channels=v[1],
                                                 kernel_size=v[2],
                                                 stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0],
                               out_channels=v[1],
                               kernel_size=v[2],
                               stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError

    return nn.Sequential(OrderedDict(layers))

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
    
class NACA0012(data.Dataset):
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


""" class SstSeq(data.Dataset):
    def __init__(self, root, is_train, cond_len, pred_len, transform=None):
        super(SstSeq, self).__init__()

        self.SST_dataset = load_data(root)
        self.is_train = is_train
        self.cond_len = cond_len
        self.pred_len = pred_len
        self.transform = transform
        self.train_len = self.SST_dataset['X_train'].shape[0] - self.cond_len - self.pred_len  # 13188
        self.test_len = self.SST_dataset['X_test'].shape[0] - self.cond_len - self.pred_len  # 2412
        self.length = self.train_len  # size of each epoch

    def __getitem__(self, idx):
        if self.is_train:
            # random training
            start_point = np.random.randint(0, self.train_len)
            inputs = self.SST_dataset['X_train'][start_point:start_point + self.cond_len, ...]
            outputs = self.SST_dataset['Y_train'][
                      start_point + self.cond_len:start_point + self.cond_len + self.pred_len, ...]
        else:
            # sequentially testing
            start_point = idx
            inputs = self.SST_dataset['X_test'][start_point:start_point + self.cond_len, ...]
            outputs = self.SST_dataset['Y_test'][
                      start_point + self.cond_len:start_point + self.cond_len + self.pred_len, ...]

        inputs = inputs[:, np.newaxis, :, :]
        outputs = outputs[:, np.newaxis, :, :]
        outputs = torch.from_numpy(outputs).contiguous().float()
        inputs = torch.from_numpy(inputs).contiguous().float()
        out = [idx, outputs, inputs, start_point]

        return out

    def __len__(self):
        return self.length """


class RecordHist:
    def __init__(self, verbose=False):
        """
        Args:
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.verbose = verbose
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, epoch, save_path):
        self.save_checkpoint(val_loss, model, epoch, save_path)

    def save_checkpoint(self, val_loss, model, epoch, save_path):
        """
        Saves model.
        """
        if self.verbose:
            print(
                f'Validation loss from ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
            )
        torch.save(
            model, save_path + "/" +
            "checkpoint_{}_{:.6f}.pth.tar".format(epoch, val_loss))
        self.val_loss_min = val_loss