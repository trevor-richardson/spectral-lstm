
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class SequentialMNIST(Dataset):

    MODE_TRAIN = 0
    MODE_VAL = 1
    MODE_TEST = 2

    def __init__(self, mode=MODE_TRAIN, pixel_wise=True):
        self.mode = mode
        self.pixel_wise = pixel_wise

        self.all_data = datasets.MNIST(os.path.join(os.path.dirname(__file__)),
            train=True,
            download=True
        )

        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # shuffle just in case
        self.train_data = self.all_data.train_data
        self.train_labels = self.all_data.train_labels
        perm = torch.randperm(len(self.all_data))
        self.train_data = self.train_data[perm]
        self.train_labels = self.train_labels[perm]

        self.n_val = int(0.1 * len(self.all_data))
        self.n_train = len(self.all_data) - self.n_val

        self.data_train = self.train_data[0:self.n_train]
        self.data_train_labels = self.train_labels[0:self.n_train]

        self.data_val = self.train_data[self.n_train:]
        self.data_val_labels = self.train_labels[self.n_train:]

        self.data_test = datasets.MNIST(os.path.join(os.path.dirname(__file__)),
            train=False,
            download=True
        )
        self.n_test = len(self.data_test)

        # data in form (n x time x data dim)
        # in this case (60000 x 784 x 1)
        # print (self.data[0])

        self.input_dimension = 1
        self.output_dimension = 10

    def train(self):
        self.mode = SequentialMNIST.MODE_TRAIN

    def val(self):
        self.mode = SequentialMNIST.MODE_VAL

    def test(self):
        self.mode = SequentialMNIST.MODE_TEST

    def __len__(self):
        if self.mode == SequentialMNIST.MODE_TRAIN:
            return self.n_train
        elif self.mode == SequentialMNIST.MODE_VAL:
            return self.n_val
        else:
            return self.n_test

    def __getitem__(self, i):
        if self.mode == SequentialMNIST.MODE_TRAIN:
            inp = self.data_train[i]
            out = self.data_train_labels[i]
            # inp = inp.view(-1, 1)
        elif self.mode == SequentialMNIST.MODE_VAL:
            inp = self.data_val[i]
            out = self.data_val_labels[i]
            # inp = inp.view(-1, 1)
        else:
            inp, out = self.data_test[i]
            inp = inp.view(-1, 1)
            return inp, out

        inp = Image.fromarray(inp.numpy(), mode='L')
        inp = self.trans(inp)
        inp = inp.view(-1, 1)
        # print (inp)
        # input("")
        return inp, out

if __name__ == "__main__":
    d = SequentialMNIST()
