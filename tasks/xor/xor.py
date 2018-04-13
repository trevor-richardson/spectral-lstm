
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# np.random.seed(2)

class XOR(Dataset):

    MODE_TRAIN = 0
    MODE_VAL = 1
    MODE_TEST = 2

    def __init__(self, mode=MODE_TRAIN):
        self.mode = mode
        self.full_data = torch.load(os.path.join(os.path.dirname(__file__), 'traindata.pt'))

        n_elements = self.full_data.shape[0]
        self.n_train = int(0.8 * n_elements)
        self.n_val = int(0.1 * n_elements)
        self.n_test = n_elements - self.n_train - self.n_val

        # n seq x tsteps x dim
        self.train_inputs = torch.from_numpy(self.full_data[:self.n_train, :, :2]).float()
        self.train_targets = torch.from_numpy(self.full_data[:self.n_train, :, 2:]).float()

        self.val_inputs = torch.from_numpy(self.full_data[self.n_train:self.n_train+self.n_val, :, :2]).float()
        self.val_targets = torch.from_numpy(self.full_data[self.n_train:self.n_train+self.n_val, :, 2:]).float()

        self.test_inputs = torch.from_numpy(self.full_data[self.n_train+self.n_val:, :, :2]).float()
        self.test_targets = torch.from_numpy(self.full_data[self.n_train+self.n_val:, :, 2:]).float()

        self.input_dimension = self.train_inputs.size(2)
        self.output_dimension = self.train_targets.size(2)

    def train(self):
        self.mode = XOR.MODE_TRAIN

    def val(self):
        self.mode = XOR.MODE_VAL

    def test(self):
        self.mode = XOR.MODE_TEST

    def __len__(self):
        if self.mode == XOR.MODE_TRAIN:
            return self.n_train
        elif self.mode == XOR.MODE_VAL:
            return self.n_val
        else:
            return self.n_test

    def __getitem__(self, i):
        if self.mode == XOR.MODE_TRAIN:
            return self.train_inputs[i], self.train_targets[i]
        elif self.mode == XOR.MODE_VAL:
            return self.val_inputs[i], self.val_targets[i]
        else:
            return self.test_inputs[i], self.test_targets[i]


if __name__ == '__main__':
    # generate data
    L = 50 # sequence length
    N = 100000 # n samples

    # samples x length, (2 x in, 1 x out)
    d = np.zeros((N, L, 3)).astype(np.float32)

    for i in range(N):
        d[i, :, 0] = np.random.randint(2, size=L)
        # generate the two random ind
        ind1 = np.random.randint(1, L//10)
        d[i, ind1, 1] = 1
        ind2 = np.random.randint(L//10, L//2)
        d[i, ind2, 1] = 1
        # set final ind as xor
        d[i, -1, 2] = (d[i, ind1, 0] and not d[i, ind2, 0]) or (not d[i, ind1, 0] and d[i, ind2, 0])
    torch.save(d, open('traindata.pt', 'wb'))
