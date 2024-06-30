# 在这里改成是BUSI对应的数据
import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from einops import repeat
from icecream import ic
import itertools
from torch.utils.data.sampler import Sampler
from PIL import Image

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label_h, label_w = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
        label = torch.from_numpy(label.astype(np.float32))
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long()}
        return sample


class ACDC_dataset(Dataset):
    def __init__(self, base_dir=None, split='train', num1=None, num2=None, transform=None, fold = 0):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.fold = fold
        if self.split == "train":
            with open(self._base_dir + "/train_{}.list".format(self.fold), "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "").split(".")[0] for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/val_{}.list".format(self.fold), "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "").split(".")[0] for item in self.sample_list]
        if num1 is not None and self.split == "train":
            if num1 < 306:
                start2_list = [306, 306, 307, 307, 307] # 现在这个实际上并没有随机划分
                start2 = start2_list[self.fold]
                self.sample_list = self.sample_list[:num1] + self.sample_list[start2:num2 + start2]
            else:
                self.sample_list = self.sample_list[:num1]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        image_path = '/research/d4/gds/jzmiao22/ultrasound/nnUNet_raw/Dataset999_BUSI0/imagesTr/' + case + '_0000.png'
        image = Image.open(image_path)
        image = np.array(image)
        # mn = image.mean()
        # std = image.std()
        # # print(data[c].shape, data[c].dtype, mn, std)
        # image = (image - mn) / (std + 1e-8)
        image = (image - image.min()) / (image.max() - image.min())
        label_path = '/research/d4/gds/jzmiao22/ultrasound/nnUNet_raw/Dataset999_BUSI0/labelsTr/' + case + '.png'
        label = Image.open(label_path)
        label = np.array(label)
        sample = {"image": image, "label": label}
        
        if self.split == "train":
            sample = self.transform(sample)

        sample['case_name'] = self.sample_list[idx].strip('\n')

        return sample
    

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)