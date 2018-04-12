import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class PeepholeLSTMCell(nn.Module):
    def __init__(self, input_size,
                    hidden_size,
                    weight_init=None,
                    reccurent_weight_init=None,
                    drop=None,
                    rec_drop=None):
        super(PeepholeLSTMCell, self).__init__()

        print("Initializing PeepholeLSTMCell")
        self.hidden_size = hidden_size
        if(weight_init==None):
            self.W_f = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_f = nn.init.xavier_normal(self.W_f)
            self.W_i = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_i = nn.init.xavier_normal(self.W_i)
            self.W_o = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_o = nn.init.xavier_normal(self.W_o)
            self.W_c = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_c = nn.init.xavier_normal(self.W_c)
        else:
            self.W_f = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_f = weight_init(self.W_f)
            self.W_i = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_i = weight_init(self.W_i)
            self.W_o = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_o = weight_init(self.W_o)
            self.W_c = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_c = weight_init(self.W_c)

        if(reccurent_weight_init == None):
            self.U_f = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_f = nn.init.orthogonal(self.U_f)
            self.U_i = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_i = nn.init.orthogonal(self.U_i)
            self.U_o = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_o = nn.init.orthogonal(self.U_o)
            self.U_c = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_c = nn.init.orthogonal(self.U_c)
        else:
            self.U_f = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_f = recurrent_weight_initializer(self.U_f)
            self.U_i = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_i = recurrent_weight_initializer(self.U_i)
            self.U_o = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_o = recurrent_weight_initializer(self.U_o)
            self.U_c = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.U_c = recurrent_weight_initializer(self.U_c)

        self.b_f = nn.Parameter(torch.zeros(hidden_size))
        self.b_i = nn.Parameter(torch.zeros(hidden_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))
        self.b_c = nn.Parameter(torch.zeros(hidden_size))

        if(drop==None):
            self.keep_prob = False
        else:
            self.keep_prob = True
            self.dropout = nn.Dropout(drop)
        if(rec_drop == None):
            self.rec_keep_prob = False
        else:
            self.rec_keep_prob = True
            self.rec_dropout = nn.Dropout(rec_drop)

        self.states = None
        if((self.W_f.data.size()[0], self.W_f.data.size()[1]) != (input_size, hidden_size) or (self.U_f.data.size()[0], self.U_f.data.size()[1]) != (hidden_size, hidden_size)
            or self.b_f.data.size()[0] != (hidden_size)):
            print("Dimensions for weight_init return should be (input_dimension, hidden_size)\nDimensions for reccurent_weight_init shoudl be (hidden_size, hidden_size)")
            print((self.W_f.data.size()[0], self.W_f.data.size()[1]), "Current weight_init shape           ---- The shape should be ", (input_size, hidden_size))
            print((self.U_f.data.size()[0], self.U_f.data.size()[1]), "Current reccurent_weight_init shape ---- The shape should be ", (hidden_size, hidden_size))
            print(self.b_f.data.size()[0], "Current bias shape                  ---- The shape should be ", (hidden_size,))
            sys.exit()

    def reset(self, batch_size=1, cuda=True):
        if cuda:
            self.states = (Variable(torch.randn(batch_size, self.hidden_size)).cuda(), Variable(torch.randn(batch_size, self.hidden_size)).cuda())
        else:
            self.states = (Variable(torch.randn(batch_size, self.hidden_size)), Variable(torch.randn(batch_size, self.hidden_size)))

    def forward(self, X_t):
        h_t_previous, c_t_previous = self.states

        if self.keep_prob:
            X_t = self.dropout(X_t)
        if self.rec_keep_prob:
            h_t_previous = self.rec_dropout(h_t_previous)
            c_t_previous = self.rec_dropout(c_t_previous)


        f_t = F.sigmoid(
            torch.mm(X_t, self.W_f) + torch.mm(c_t_previous, self.U_f) + self.b_f #w_f needs to be the previous input shape by the number of hidden neurons
        )


        i_t = F.sigmoid(
            torch.mm(X_t, self.W_i) + torch.mm(c_t_previous, self.U_i) + self.b_i
        )


        o_t = F.sigmoid(
            torch.mm(X_t, self.W_o) + torch.mm(c_t_previous, self.U_o) + self.b_o
        )


        c_hat_t = F.tanh(
            torch.mm(X_t, self.W_c) + torch.mm(c_t_previous, self.U_c) + self.b_c
        )

        c_t = (f_t * c_t_previous) + (i_t * c_hat_t)

        h_t = o_t * F.tanh(c_t)

        self.states = (h_t, c_t)
        return h_t

class PeepholeLSTM(nn.Module):

    def __init__(self,
                 input_size=1,
                 hidden_size=64,
                 output_size=1,
                 layers=1):
        super(PeepholeLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = layers

        self.lstms = nn.ModuleList()
        self.lstms.append(PeepholeLSTMCell(input_size=input_size, hidden_size=hidden_size))
        for i in range(self.layers-1):
            self.lstms.append(PeepholeLSTMCell(input_size=hidden_size, hidden_size=hidden_size))
        self.fc1 = nn.Linear(hidden_size, output_size)

        nn.init.xavier_normal(self.fc1.weight.data)
        nn.init.constant(self.fc1.bias.data, 0)

    def reset(self, batch_size=1, cuda=True):
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
