import os
import os.path as osp
import numpy as np
import tqdm
import pandas as pd
from pydicom import dcmread

# dataset_name = 'lmayo'
dataset_name = 'cmayo'
threshold = 0.85
# num_samples = {
#     'train': 256000,
#     'test': 64000,
# }
# stride = 32
# patch_size = 64
# stride = 32
# patch_size = 128
num_samples = {
    'train': 100000,
    'test': 64000,
}
stride = 64
patch_size = 256


def crop(ds):
    patches = []
    for left in range(0, ds.shape[0] - patch_size + 1, stride):
        for top in range(0, ds.shape[1] - patch_size + 1, stride):
            patches.append(ds[left: left + patch_size, top: top + patch_size])
    return patches


# C021, C120 ...
for phase in ['train', 'test']:
    print('generate {} data...'.format(phase))
    # for phase in ['test']:
    patches = []
    ct_list = pd.read_csv(osp.join(dataset_name, phase + '_id.csv'), index_col=None, header=None)
    for i in tqdm.trange(len(ct_list)):
        full_dose, low_dose = ct_list[0][i], ct_list[1][i]
        full_dose_list = sorted(os.listdir(osp.join(dataset_name, full_dose)))
        low_dose_list = sorted(os.listdir(osp.join(dataset_name, low_dose)))
        if len(full_dose_list) != len(low_dose_list):
            raise Exception('{} and {} should be equal..'.format(full_dose, low_dose))
        for j in tqdm.trange(len(full_dose_list)):
            a, b = full_dose_list[j], low_dose_list[j]
            if a != b:
                print(a, b)
                continue
            f_ps = crop(dcmread(osp.join(dataset_name, full_dose, a)).pixel_array.astype(np.float32))
            l_ps = crop(dcmread(osp.join(dataset_name, low_dose, b)).pixel_array.astype(np.float32))
            for k in range(len(f_ps)):
                black_percent = np.mean(np.clip(f_ps[k] - 1024, -500, 500) == -500)
                if black_percent < threshold:
                    patches.append(np.array([l_ps[k], f_ps[k]]))
    patches = np.array(patches).reshape((-1, 2, 1, patch_size, patch_size)).transpose((1, 0, 2, 3, 4))
    print(patches.shape)
    print('process {} patches...'.format(phase))
    np.random.seed(0)
    index_file_path = osp.join(dataset_name, phase + '_index.npy')
    if osp.exists(index_file_path):
        index = np.load(index_file_path)
    else:
        index = np.random.choice(patches.shape[1], num_samples[phase], replace=False)
        np.save(index_file_path, index)
    print('save {} patches...'.format(phase))
    np.save(osp.join(dataset_name, '{}'.format(phase)), patches[:, index, :, :, :].astype(np.uint16))
    print('complete save {} patches...'.format(phase))
