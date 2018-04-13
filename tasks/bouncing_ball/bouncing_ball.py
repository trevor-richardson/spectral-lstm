
import os

import numpy as np
import torch

from torch.utils.data import Dataset

# np.random.seed(2)

class BouncingBall(Dataset):

    MODE_TRAIN = 0
    MODE_VAL = 1
    MODE_TEST = 2

    def __init__(self, mode=MODE_TRAIN, vectorize=True):
        self.mode = mode
        self.vectorize = vectorize
        self.full_data = torch.load(os.path.join(os.path.dirname(__file__), 'traindata.pt'))
        self.full_data /= 255.0

        n_elements = self.full_data.shape[0]
        self.n_train = int(0.8 * n_elements)
        self.n_val = int(0.1 * n_elements)
        self.n_test = n_elements - self.n_train - self.n_val

        # n seq x tsteps x dim
        self.train_inputs = torch.from_numpy(self.full_data[:self.n_train, :-1, :, :]).float()
        self.train_targets = torch.from_numpy(self.full_data[:self.n_train, 1:, :, :]).float()

        self.val_inputs = torch.from_numpy(self.full_data[self.n_train:self.n_train+self.n_val, :-1, :, :]).float()
        self.val_targets = torch.from_numpy(self.full_data[self.n_train:self.n_train+self.n_val, 1:, :, :]).float()

        self.test_inputs = torch.from_numpy(self.full_data[self.n_train+self.n_val:, :-1, :, :]).float()
        self.test_targets = torch.from_numpy(self.full_data[self.n_train+self.n_val:, 1:, :, :]).float()

        if self.vectorize:
            self.train_inputs = self.train_inputs.contiguous().view(*self.train_inputs.size()[0:-2], -1)
            self.train_targets = self.train_targets.contiguous().view(*self.train_targets.size()[0:-2], -1)

            self.val_inputs = self.val_inputs.contiguous().view(*self.val_inputs.size()[0:-2], -1)
            self.val_targets = self.val_targets.contiguous().view(*self.val_targets.size()[0:-2], -1)

            self.test_inputs = self.test_inputs.contiguous().view(*self.test_inputs.size()[0:-2], -1)
            self.test_targets = self.test_targets.contiguous().view(*self.test_targets.size()[0:-2], -1)

        if self.vectorize:
            self.input_dimension = self.train_inputs.size(2)
            self.output_dimension = self.train_targets.size(2)
        else:
            self.input_dimension = self.train_inputs.size()[2:]
            self.output_dimension = self.train_targets.size()[2:]

    def train(self):
        self.mode = BouncingBall.MODE_TRAIN

    def val(self):
        self.mode = BouncingBall.MODE_VAL

    def test(self):
        self.mode = BouncingBall.MODE_TEST

    def __len__(self):
        if self.mode == BouncingBall.MODE_TRAIN:
            return self.n_train
        elif self.mode == BouncingBall.MODE_VAL:
            return self.n_val
        else:
            return self.n_test

    def __getitem__(self, i):
        if self.mode == BouncingBall.MODE_TRAIN:
            return self.train_inputs[i], self.train_targets[i]
        elif self.mode == BouncingBall.MODE_VAL:
            return self.val_inputs[i], self.val_targets[i]
        else:
            return self.test_inputs[i], self.test_targets[i]
