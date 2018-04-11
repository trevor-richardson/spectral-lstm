
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class NoiselessMemorization(Dataset):

    MODE_TRAIN = 0
    MODE_VAL = 1
    MODE_TEST = 2

    def __init__(self, sequence_len=50, mode=MODE_TRAIN):
        assert sequence_len in [50, 100, 200, 500, 1000]
        self.mode = mode
        self.sequence_len = sequence_len
        self.full_data = torch.load(os.path.join(os.path.dirname(__file__), 'traindata'+str(self.sequence_len)+'.pt'))

        n_elements = self.full_data.shape[0]
        self.n_train = int(0.8 * n_elements)
        self.n_val = int(0.1 * n_elements)
        self.n_test = n_elements - self.n_train - self.n_val

        # n seq x tsteps x dim
        split = 3
        self.train_inputs = torch.from_numpy(self.full_data[:self.n_train, :, :split]).float()
        self.train_targets = torch.from_numpy(self.full_data[:self.n_train, :, split:]).float()

        self.val_inputs = torch.from_numpy(self.full_data[self.n_train:self.n_train+self.n_val, :, :split]).float()
        self.val_targets = torch.from_numpy(self.full_data[self.n_train:self.n_train+self.n_val, :, split:]).float()

        self.test_inputs = torch.from_numpy(self.full_data[self.n_train+self.n_val:, :, :split]).float()
        self.test_targets = torch.from_numpy(self.full_data[self.n_train+self.n_val:, :, split:]).float()

        self.input_dimension = self.train_inputs.size(2)
        self.output_dimension = 3 #self.train_targets.size(2)

    def train(self):
        self.mode = NoiselessMemorization.MODE_TRAIN

    def val(self):
        self.mode = NoiselessMemorization.MODE_VAL

    def test(self):
        self.mode = NoiselessMemorization.MODE_TEST

    def __len__(self):
        if self.mode == NoiselessMemorization.MODE_TRAIN:
            return self.n_train
        elif self.mode == NoiselessMemorization.MODE_VAL:
            return self.n_val
        else:
            return self.n_test

    def __getitem__(self, i):
        if self.mode == NoiselessMemorization.MODE_TRAIN:
            return self.train_inputs[i], self.train_targets[i]
        elif self.mode == NoiselessMemorization.MODE_VAL:
            return self.val_inputs[i], self.val_targets[i]
        else:
            return self.test_inputs[i], self.test_targets[i]



if __name__ == '__main__':
    # generate data
    for l in [50, 100, 200, 500, 1000]:
        L = l + 10 # sequence length
        N = 10000 # n samples

        # samples x length, (3 x in, 1 x out)
        # essentially classification
        d = np.zeros((N, L, 4)).astype(np.float32)

        # set input 0:5 input symbols
        for i in range(N):
            for j in range(5):
                symbol = np.random.randint(0, 2)
                # 5 random input bits
                x = np.zeros((3,))
                x[symbol] = 1
                # set input
                d[i, j, 0:3] = x
                # set final 5 targets
                # this is classification 0, 1, train with CrossEntropyLoss
                d[i, -5+j, 3] = symbol

        # set all intermediary inputs
        d[:, 5:L-5, 0:3] = np.array([0, 0, 0])

        # set all intermediary outputs
        d[:, 5:L-5, 3] = 2 #np.array([0, 0, 1])

        # set flag to signify time to output
        d[:, L-6, 0:3] = np.array([0, 0, 1])

        torch.save(d, open('traindata'+str(l)+'.pt', 'wb'))
