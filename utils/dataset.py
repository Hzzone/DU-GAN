import torch
import torch.utils.data as tordata
import os.path as osp
import numpy as np
from functools import partial


class CTPatchDataset(tordata.Dataset):
    def __init__(self, npy_root, hu_range, transforms=None):
        self.transforms = transforms
        hu_min, hu_max = hu_range
        data = torch.from_numpy(np.load(npy_root).astype(np.float32) - 1024)
        # normalize to [0, 1]
        data = (torch.clamp(data, hu_min, hu_max) - hu_min) / (hu_max - hu_min)
        self.low_doses, self.full_doses = data[0], data[1]

    def __getitem__(self, index):
        low_dose, full_dose = self.low_doses[index], self.full_doses[index]
        if self.transforms is not None:
            low_dose = self.transforms(low_dose)
            full_dose = self.transforms(full_dose)
        return low_dose, full_dose

    def __len__(self):
        return len(self.low_doses)


data_root = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'dataset')
dataset_dict = {
    'cmayo_train_64': partial(CTPatchDataset, npy_root=osp.join(data_root, 'cmayo/train_64.npy')),
    'cmayo_test_512': partial(CTPatchDataset, npy_root=osp.join(data_root, 'cmayo/test_512.npy')),
}
