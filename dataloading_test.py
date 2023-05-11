import numpy as np
import os
import torch
from torch.utils.data import Dataset
from loading_helper import generateMyTestingData

def my_collate(batch):
    data = []
    scanIDX = []
    orig_size = []
    for batch_idx in range(len(batch)):
        for patch_idx in range(len(batch[batch_idx])):
            data.append(batch[batch_idx][patch_idx][0])
            scanIDX.append([int(batch[batch_idx][patch_idx][1].split("_")[1])])
            orig_size.append([int(batch[batch_idx][patch_idx][1].split("_")[2].split("(")[-1].split(")")[0].split(",")[0]),
                              int(batch[batch_idx][patch_idx][1].split("_")[2].split("(")[-1].split(")")[0].split(",")[1]),
                              int(batch[batch_idx][patch_idx][1].split("_")[2].split("(")[-1].split(")")[0].split(",")[2])])
    return torch.tensor(np.array(data)), torch.tensor(np.array(scanIDX)), torch.tensor(np.array(orig_size))
def load_testing_data(args):

    if not os.path.isdir("my_testing_data"):
        os.makedirs("my_testing_data")
        print("Create my testing dataset")
        generateMyTestingData(args)

    test_idx = []
    for root, dirs, files in os.walk("my_testing_data"):
        for dir in dirs:
            test_idx.append(dir)

    fin_test_dataset = scan_LIDCDataset("my_testing_data/", test_idx)
    test_loader = torch.utils.data.DataLoader(fin_test_dataset, batch_size=1, shuffle=False,
                                              collate_fn=my_collate, pin_memory=True)

    return test_loader
class scan_LIDCDataset(Dataset):
    def __init__(self, dir, allidx):
        self.dir = dir
        self.allidx = allidx
    def __len__(self):
        return len(self.allidx)
    def __getitem__(self, idx):
        all_patches_of_this_scan = []
        for root, dirs, files in os.walk(self.dir+str(self.allidx[idx])+"/"):
            numpatches = len(files)
            for patch_idx in range(numpatches):
                input = np.expand_dims(np.load(self.dir +str(self.allidx[idx])+"/in_"+str(patch_idx)+".npy"), axis=0)
                all_patches_of_this_scan.append([input, self.allidx[idx]])
        return all_patches_of_this_scan
