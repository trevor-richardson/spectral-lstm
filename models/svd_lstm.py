
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

################################################################################
# LSTM variants
################################################################################

class SvdLSTMCell(nn.Module):

    def __init__(self,
                 input_size=1,
                 hidden_size=64,
                 output_size=1):
        super(SvdLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        #input weight matrices
        self.Wi = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.Wit = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.bi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.Wf = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.Wft = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.bf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.Wc = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.Wct = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.bc = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        #These are recurrent weights where there is no projection
        self.Wo = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.Who = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bo = nn.Parameter(torch.Tensor(hidden_size, 1))

        nn.init.xavier_normal(self.Wi.data)
        nn.init.xavier_normal(self.Wit.data)
        nn.init.constant(self.bi.data, 0)

        nn.init.xavier_normal(self.Wf.data)
        nn.init.xavier_normal(self.Wft.data)
        nn.init.constant(self.bf.data, 0)

        nn.init.xavier_normal(self.Wo.data)
        nn.init.orthogonal(self.Who.data)
        nn.init.constant(self.bo.data, 0)

        nn.init.xavier_normal(self.Wc.data)
        nn.init.xavier_normal(self.Wct.data)
        nn.init.constant(self.bc.data, 0)

        self.states = None
        self.U = None
        self.V = None

    def reset(self, batch_size=1, cuda=False):
        if cuda:
            self.states = (Variable(torch.randn(batch_size, self.hidden_size)).cuda(), Variable(torch.randn(batch_size, self.hidden_size, self.hidden_size)).cuda())
        else:
            self.states = (Variable(torch.randn(batch_size, self.hidden_size)), Variable(torch.randn(batch_size, self.hidden_size, self.hidden_size)))

        self.U, S, self.V = self.batch_svd(self.states[1])

    def batch_svd(self, inp):
        Us = []
        Ss = []
        Vs = []
        for element in inp:
            U, S, V = torch.svd(element.double())
            Us.append(U.float())
            Ss.append(S.float())
            Vs.append(V.t().float())

        return torch.stack(Us).cuda(), torch.stack(Ss).cuda(), torch.stack(Vs).cuda()

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
        Assume cx is of form (batch size, hidden_size, hidden_size) now a matrix
        """
        hx, cx = self.states

        if x.dim() == 1:
            x = x.unsqueeze(0)
        if hx.dim() == 1:
            hx = hx.unsqueeze(0)

        x_diag = self.batch_diag(x)
        hx_diag = self.batch_diag(hx)

        bs = x.size()[0]

        i = F.sigmoid(torch.matmul(torch.matmul(self.Wi, x_diag), self.Wit.t()) + torch.matmul(torch.matmul(self.U, hx_diag), self.V) + self.bi)
        f = F.sigmoid(torch.matmul(torch.matmul(self.Wf, x_diag), self.Wft.t()) + torch.matmul(torch.matmul(self.U, hx_diag), self.V) + self.bf)
        g = F.tanh(torch.matmul(torch.matmul(self.Wc, x_diag), self.Wct.t()) + torch.matmul(torch.matmul(self.U, hx_diag), self.V) + self.bc)

        x_vec = x.unsqueeze(2)
        hx_vec = hx.unsqueeze(2)

        o = F.sigmoid(torch.matmul(self.Wo, x_vec) + torch.matmul(self.Who, hx_vec) + self.bo)

        cx = torch.mul(f, cx) + torch.mul(i, g)

        self.U, S, self.V = self.batch_svd(cx)
        #need to extract row in the proper gradient optimal way
        o = o.squeeze()
        hx = torch.mul(o, F.sigmoid(S))

        self.states = (hx, cx)
        return hx


class SvdLSTM(nn.Module):

    def __init__(self,
                 input_size=1,
                 hidden_size=64,
                 output_size=1,
                 output_weighting=True,
                 layers=1):
        super(SvdLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = layers

        self.lstms = nn.ModuleList()
        self.lstms.append(SvdLSTMCell(input_size=input_size, hidden_size=hidden_size))
        for i in range(self.layers-1):
            self.lstms.append(SvdLSTMCell(input_size=hidden_size, hidden_size=hidden_size))
        self.fc1 = nn.Linear(hidden_size, output_size)

        nn.init.xavier_normal(self.fc1.weight.data)
        nn.init.constant(self.fc1.bias.data, 0)

    def reset(self, batch_size=1, cuda=False):
        for i in range(len(self.lstms)):
            self.lstms[i].reset(batch_size=batch_size, cuda=cuda)

    def forward(self, x):
        """
        Assume cx is of form (batch size, hidden_size, hidden_size) now a matrix
        """
        for i in range(len(self.lstms)):
            x = self.lstms[i](x)
        o = self.fc1(x)
        return o
