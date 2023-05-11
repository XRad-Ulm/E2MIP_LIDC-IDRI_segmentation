import numpy as np
import os
import torch
from torch.utils.data import Dataset
from loading_helper import generateMyTrainingData


class RandomCrop3D():
    def __init__(self, crop_sz):
        self.crop_sz = tuple(crop_sz)

    def __call__(self, img, seg):
        img_sz = img.shape
        slice_hwd = [self._get_slice(i, k) for i, k in zip(img_sz, self.crop_sz)]
        return self._crop(img, seg, *slice_hwd)
    @staticmethod
    def _get_slice(sz, crop_sz):
        try:
            lower_bound = torch.randint(sz - crop_sz, (1,)).item()
            return lower_bound, lower_bound + crop_sz
        except:
            print("could not crop out patch size")
            return (None, None)
    #
    @staticmethod
    def _crop(img, seg, slice_h, slice_w, slice_d):
        return img[slice_h[0]:slice_h[1], slice_w[0]:slice_w[1], slice_d[0]:slice_d[1]], seg[slice_h[0]:slice_h[1], slice_w[0]:slice_w[1], slice_d[0]:slice_d[1]]


def load_training_data(args):
    if not os.path.isdir("my_training_data"):
        os.makedirs("my_training_data")
        print("Create my training dataset")
        generateMyTrainingData(args)

    train_idx = []
    for root, dirs, files in os.walk("my_training_data"):
        for file in files:
            if file.startswith('in_'):
                num = int(file.split("_")[1].split(".")[0])
                train_idx.append(num)
    np.random.shuffle(train_idx)
    val_train = 0.1
    val_idx = train_idx[len(train_idx) - int(val_train * len(train_idx)):]
    train_idx = train_idx[:len(train_idx) - int(val_train * len(train_idx))]


    fin_train_dataset = LIDCDataset("my_training_data", train_idx, RandomCrop3D(args.patch_size))
    fin_val_dataset = LIDCDataset("my_training_data", val_idx, RandomCrop3D(args.patch_size))

    train_loader = torch.utils.data.DataLoader(fin_train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(fin_val_dataset, batch_size=args.batch_size, shuffle=True)

    return train_loader, val_loader


class LIDCDataset(Dataset):
    def __init__(self, dir, allidx, transform):
        self.dir = dir
        self.allidx = allidx
        self.transform = transform

    def __len__(self):
        return len(self.allidx)

    def __getitem__(self, idx):
        volume = np.load(self.dir + "/in_" + str(self.allidx[idx]) + ".npy")
        segmentation = np.load(self.dir + "/seg_" + str(self.allidx[idx]) + ".npy")
        if self.transform:
            volume, segmentation = self.transform(volume,segmentation)
        volume = np.expand_dims(volume,axis=0)
        segmentation = np.expand_dims(segmentation,axis=0)
        return volume, segmentation
