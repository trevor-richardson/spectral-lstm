
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

################################################################################
# LSTM variants
################################################################################

class SpectralLSTMCell(nn.Module):

    def __init__(self,
                 input_size=1,
                 hidden_size=64,
                 output_size=1,
                 output_weighting=True,
                 orthogonal=False):
        super(SpectralLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_weighting = output_weighting
        self.orthogonal = orthogonal

        self.Wii = nn.Parameter(torch.Tensor(hidden_size, input_size))
        #self.Wiit = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Wiit = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.Whi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        #self.Whit = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.Whit = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.Wif = nn.Parameter(torch.Tensor(hidden_size, input_size))
        #self.Wift = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Wift = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.Whf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.Whft = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.Wig = nn.Parameter(torch.Tensor(hidden_size, input_size))
        #self.Wigt = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Wigt = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.Whc = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.Whct = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.Wio = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.Who = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bo = nn.Parameter(torch.Tensor(hidden_size, 1))

        self.Wcx = nn.Parameter(torch.Tensor(hidden_size, 1))

        if orthogonal:
            nn.init.orthogonal(self.Wii.data)
            nn.init.orthogonal(self.Wiit.data)
            nn.init.orthogonal(self.Whi.data)
            nn.init.orthogonal(self.Whit.data)
            nn.init.constant(self.bi.data, 0)

            nn.init.orthogonal(self.Wif.data)
            nn.init.orthogonal(self.Wift.data)
            nn.init.orthogonal(self.Whf.data)
            nn.init.orthogonal(self.Whft.data)
            nn.init.constant(self.bf.data, 0)

            nn.init.orthogonal(self.Wig.data)
            nn.init.orthogonal(self.Wigt.data)
            nn.init.orthogonal(self.Whc.data)
            nn.init.orthogonal(self.Whct.data)
            nn.init.constant(self.bg.data, 0)

            nn.init.orthogonal(self.Wio.data)
            nn.init.orthogonal(self.Who.data)
            nn.init.constant(self.bo.data, 0)

            nn.init.orthogonal(self.Wcx.data)
        else:
            nn.init.xavier_normal(self.Wii.data)
            nn.init.xavier_normal(self.Wiit.data)
            nn.init.xavier_normal(self.Whi.data)
            nn.init.xavier_normal(self.Whit.data)
            nn.init.constant(self.bi.data, 0)

            nn.init.xavier_normal(self.Wif.data)
            nn.init.xavier_normal(self.Wift.data)
            nn.init.xavier_normal(self.Whf.data)
            nn.init.xavier_normal(self.Whft.data)
            nn.init.constant(self.bf.data, 0)

            nn.init.xavier_normal(self.Wig.data)
            nn.init.xavier_normal(self.Wigt.data)
            nn.init.xavier_normal(self.Whc.data)
            nn.init.xavier_normal(self.Whct.data)
            nn.init.constant(self.bg.data, 0)

            nn.init.xavier_normal(self.Wio.data)
            nn.init.xavier_normal(self.Who.data)
            nn.init.constant(self.bo.data, 0)

            nn.init.xavier_normal(self.Wcx.data)

        self.states = None

    def reset(self, batch_size=1, cuda=False):
        if cuda:
            self.states = (Variable(torch.randn(batch_size, self.hidden_size)).cuda(), Variable(torch.randn(batch_size, self.hidden_size, self.hidden_size)).cuda())
        else:
            self.states = (Variable(torch.randn(batch_size, self.hidden_size)), Variable(torch.randn(batch_size, self.hidden_size, self.hidden_size)))

    def batch_diag(self, m):
        """
        https://discuss.pytorch.org/t/batch-of-diagonal-matrix/13560/2

        NOTE: This requires the source install of pytorch
        """
        d = []
        for vec in m:
           d.append(torch.diag(vec))
        d = torch.stack(d)
        if m.data.is_cuda and torch.cuda.is_available():
            d = d.cuda()
        return d


    def forward(self, x):
        """
        Block lstm forward

        Assume cx is of form (batch size, hidden_size, hidden_size) now a matrix
        """
        hx, cx = self.states

        # print (x.size())
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if hx.dim() == 1:
            hx = hx.unsqueeze(0)

        x_diag = self.batch_diag(x)
        hx_diag = self.batch_diag(hx)

        bs = x.size()[0]

        i = F.sigmoid(torch.matmul(torch.matmul(self.Wii, x_diag), self.Wiit.t()) + torch.matmul(torch.matmul(self.Whi, hx_diag), self.Whit.t()) + self.bi)
        f = F.sigmoid(torch.matmul(torch.matmul(self.Wif, x_diag), self.Wift.t()) + torch.matmul(torch.matmul(self.Whf, hx_diag), self.Whft.t()) + self.bf)

        g = F.tanh(torch.matmul(torch.matmul(self.Wig, x_diag), self.Wigt.t()) + torch.matmul(torch.matmul(self.Whc, hx_diag), self.Whct.t()) + self.bg)


        x_vec = x.unsqueeze(2)
        hx_vec = hx.unsqueeze(2)

        o = F.sigmoid(torch.matmul(self.Wio, x_vec) + torch.matmul(self.Who, hx_vec) + self.bo)



        cx = torch.mul(f, cx) + torch.mul(i, g)

        if self.output_weighting:
            hx = torch.matmul(torch.transpose(F.tanh(cx), 1, 2), o).squeeze(2)
        else:

            cx_proj = torch.matmul(cx, self.Wcx)
            hx = torch.mul(o, F.tanh(cx_proj)).squeeze(2)



        self.states = (hx, cx)
        return hx


class SpectralLSTM(nn.Module):

    def __init__(self,
                 input_size=1,
                 hidden_size=64,
                 output_size=1,
                 output_weighting=True,
                 layers=1,
                 orthogonal=False):
        super(SpectralLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_weighting = output_weighting
        self.layers = layers
        self.orthogonal = orthogonal

        self.lstms = nn.ModuleList()
        self.lstms.append(SpectralLSTMCell(input_size=input_size, hidden_size=hidden_size, output_weighting=output_weighting, orthogonal=orthogonal))
        for i in range(self.layers-1):
            self.lstms.append(SpectralLSTMCell(input_size=hidden_size, hidden_size=hidden_size, output_weighting=output_weighting, orthogonal=orthogonal))
        self.fc1 = nn.Linear(hidden_size, output_size)

        nn.init.xavier_normal(self.fc1.weight.data)
        nn.init.constant(self.fc1.bias.data, 0)

    def reset(self, batch_size=1, cuda=False):
        for i in range(len(self.lstms)):
            self.lstms[i].reset(batch_size=batch_size, cuda=cuda)

    def forward(self, x):
        """
        Block lstm forward

        Assume cx is of form (batch size, hidden_size, hidden_size) now a matrix
        """
        for i in range(len(self.lstms)):
            x = self.lstms[i](x)
        o = self.fc1(x)
        return o
