import time
import argparse
import os
import random
import shutil
import joblib
import csv
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.autograd import Variable
from torch.utils.data import DataLoader

import configparser

config = configparser.ConfigParser()
config.read('./config.ini')

base_dir = config['DEFAULT']['BASE_DIR']
sys.path.append(base_dir + '/models/')

sys.path.append(base_dir + '/tasks/addition/')
sys.path.append(base_dir + '/tasks/bouncing_ball/')
sys.path.append(base_dir + '/tasks/multiplication/')
sys.path.append(base_dir + '/tasks/xor/')
sys.path.append(base_dir + '/tasks/sequential_mnist/')
sys.path.append(base_dir + '/tasks/noiseless_memorization/')

'''

To Do's

Make sure new two google brain tasks work
Add batchnorm as intermediate layer for network to battle covariate shift
Write visualization classes

Update Readme

'''

#get the datasets of iterest
from addition import Addition
from bouncing_ball import BouncingBall
from multiplication import Multiplication
from noiseless_memorization import NoiselessMemorization
from sequential_mnist import SequentialMNIST
from xor import XOR

from simple_lstm import SimpleLSTM, SimpleRNN
from spectral_lstm import SpectralLSTM
from svd_lstm import SvdLSTM
from gru import VanillaGRU

parser = argparse.ArgumentParser(description='SpectralRNN')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 1e-4)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='RMSprop optimizer momentum (default: 0.9)')
parser.add_argument('--alpha', type=float, default=0.95,
                    help='RMSprop alpha (default: 0.95)')
parser.add_argument('--epochs', type=int, default=500,
                    help='num training epochs (default: 500)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--hx', type=int, default=100,
                    help='hidden vec size for lstm models (default: 100)')
parser.add_argument('--layers', type=int, default=1,
                    help='num recurrent layers (default: 1)')
parser.add_argument('--batch-size', type=int, default=64,
                    help='training batch size (default: 1)')
parser.add_argument('--log-dir', default=base_dir + '/results/',
                    help='directory to save agent logs (default: base_dir/results/)')
parser.add_argument('--model-type', type=str, default='lstm',
                    help='use rnn, lstm, gru, phole, slstm, svdlstm')
parser.add_argument('--task', type=str, default='mem',
                    help='use add, mul, mem, xor, bball, seqmnist, strokemnist')
parser.add_argument('--sequence-len', type=int, default=50,
                    help='mem seq len (default: 50)')
parser.add_argument('--vis', action='store_true', default=False,
                    help='enables visdom visualization')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use gpu')

args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    pass

# save args
joblib.dump(args, os.path.join(args.log_dir, 'args_snapshot.pkl'))

fields = ['epoch', 'loss', 'acc']
train_csvfile = open(os.path.join(args.log_dir, 'train.csv'), 'w')
train_csvwriter = csv.DictWriter(train_csvfile, fieldnames=fields)
train_csvwriter.writeheader()
train_csvfile.flush()

fields_val = ['epoch', 'loss', 'acc']
val_csvfile = open(os.path.join(args.log_dir, 'val.csv'), 'w')
val_csvwriter = csv.DictWriter(val_csvfile, fieldnames=fields_val)
val_csvwriter.writeheader()
val_csvfile.flush()

def log_sigmoid(x):
    return torch.log(F.sigmoid(x))

def create_criterion():
    """
    Determine activation / loss fn for each task
    """
    if args.task == 'add':
        activation = nn.Sigmoid()
        criterion = nn.MSELoss()
    elif args.task == 'mul':
        activation = nn.Sigmoid()
        criterion = nn.MSELoss()
    elif args.task == 'mem':
        activation = nn.LogSoftmax(dim=1)
        criterion = nn.BCELoss()
    elif args.task == 'xor':
        activation = nn.Sigmoid()
        criterion = nn.MSELoss()
    elif args.task == 'bball':
        # NOTE: want sigmoid because each pixel output could be a prob
        activation = log_sigmoid
        criterion = nn.KLDivLoss()
    elif args.task == 'seqmnist':
        activation = nn.LogSoftmax(dim=1)
        criterion = nn.CrossEntropyLoss(size_average=False, reduce=False)
    else:
        raise Exception
    return activation, criterion

def create_dset():
    if args.task == 'add':
        dset = Addition()
    elif args.task == 'mul':
        dset = Multiplication()
    elif args.task == 'mem':
        dset = NoiselessMemorization(sequence_len=args.sequence_len)
    elif args.task == 'xor':
        dset = XOR()
    elif args.task == 'bball':
        dset = BouncingBall(vectorize=True)
    elif args.task == 'seqmnist':
        dset = SequentialMNIST()
    else:
        raise Exception
    return dset

dset = create_dset()
data_loader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, num_workers=2, shuffle=True)
activation, criterion = create_criterion()

def create_model():
    if args.model_type == 'rnn':
        return SimpleRNN(input_size=dset.input_dimension,
                                     hidden_size=args.hx,
                                     output_size=dset.output_dimension,
                                     layers=args.layers)
    elif args.model_type == 'lstm':
        return SimpleLSTM(input_size=dset.input_dimension,
                                      hidden_size=args.hx,
                                      output_size=dset.output_dimension,
                                      layers=args.layers)
    if args.model_type == 'gru':
        return VanillaGRU(input_size=dset.input_dimension,
                                     hidden_size=args.hx,
                                     output_size=dset.output_dimension,
                                     layers=args.layers)
    elif args.model_type == 'slstm':
        return SpectralLSTM(input_size=dset.input_dimension,
                                          hidden_size=args.hx,
                                          output_size=dset.output_dimension,
                                          output_weighting=args.output_weighting,
                                          layers=args.layers)
    elif args.model_type == 'svdlstm':
        return SvdLSTM(input_size=dset.input_dimension,
                                          hidden_size=args.hx,
                                          output_size=dset.output_dimension,
                                          layers=args.layers)
    else:
        raise Exception

#create model
model = create_model()
params = 0
for p in list(model.parameters()):
    params += p.numel()
print ("Num params: ", params)
print (model)
if args.cuda:
    model.cuda()

#initalize optimizer and learning rate decay scheduler
optimizer = optim.RMSprop(model.parameters(), lr=args.lr) #, momentum=args.momentum, alpha=args.alpha)
scheduler = ExponentialLR(optimizer, 0.95)


def run_sequence(seq, target):
    outputs = []
    targets = []
    model.reset(batch_size=seq.size(0), cuda=args.cuda)
    for i, input_t in enumerate(seq.chunk(seq.size(1), dim=1)):
        input_t = input_t.squeeze(1)
        p = activation(model(input_t))

        outputs.append(p)
        targets.append(target)

    return outputs, targets


def train(epoch):
    model.train()
    dset.train()

    total_loss = 0.0
    steps = 0
    n_correct = 0
    n_possible = 0

    for batch_idx, (data, target) in enumerate(data_loader):
        #prepare data
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        predicted_list, y_list = run_sequence(data, target)

        if args.task == 'seqmnist':
            pred = torch.cat(predicted_list)
            y_ = torch.cat(y_list)
            prediction = pred.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            n_correct += prediction.eq(y_.data.view_as(prediction)).sum()
            n_possible += int(prediction.shape[0])
            loss = F.nll_loss(pred, y_)

        elif args.task == 'mem':
            pred = predicted_list[-5:]
            pred = torch.stack(pred).transpose(1, 0).contiguous().view(-1, 3)
            target = target[:,-5:].contiguous().long().view(-1)
            prediction = pred.data.max(1, keepdim=True)[1].contiguous() # get the index of the max log-probability
            n_correct += prediction.eq(target.data.view_as(prediction)).sum()
            n_possible += int(prediction.shape[0] * prediction.shape[1])
            loss = F.nll_loss(pred, target)
        else:
            pred = torch.stack(predicted_list, 1)
            y_ = torch.stack(y_list, 1)
            loss = criterion(pred, target)

        loss.backward()
        optimizer.step()
        steps += 1
        total_loss += loss.cpu().data.numpy()[0]


    print("Train loss ", total_loss/steps)
    if args.task == 'seqmnist' or args.task == 'mem':
        print("Train Acc ", (n_correct/ n_possible))
        train_csvwriter.writerow(dict(epoch=str(epoch), loss=str(total_loss/steps), acc=str(n_correct/n_possible)))
    else:
        train_csvwriter.writerow(dict(epoch=str(epoch), loss=str(total_loss/steps), acc=str(-1)))
    train_csvfile.flush()


def validate(epoch):
    dset.val()
    model.eval()

    total_loss = 0.0
    n_correct = 0
    n_possible = 0
    steps = 0

    for batch_idx, (data, target) in enumerate(data_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data, volatile=True), Variable(target, volatile=True)

        predicted_list, y_list = run_sequence(data, target)

        if args.task == 'seqmnist':
            pred = predicted_list[-1]
            y_ = y_list[-1]
            prediction = pred.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            n_correct += prediction.eq(y_.data.view_as(prediction)).sum()
            n_possible += int(prediction.shape[0])
            loss = F.cross_entropy(pred, y_)

        elif args.task == 'mem':
            pred = predicted_list[-5:]
            pred = torch.cat(pred)
            target = target[:,-5:].contiguous().long().view(-1)
            prediction = pred.data.max(1, keepdim=True)[1].contiguous() # get the index of the max log-probability
            n_correct += prediction.eq(target.data.view_as(prediction)).sum()
            n_possible += int(prediction.shape[0] * prediction.shape[1])
            loss = F.nll_loss(pred, target)
        else:
            pred = torch.stack(predicted_list, 1)
            y_ = torch.stack(y_list, 1)
            loss = criterion(pred, target)

        steps += 1
        total_loss += loss.cpu().data.numpy()[0]

    if args.task == 'seqmnist' or args.task == 'mem':
        print("Validation Acc ", n_correct/n_possible)
        val_csvwriter.writerow(dict(epoch=str(epoch), loss=str(total_loss/steps), acc=str(n_correct/n_possible)))
    else:
        val_csvwriter.writerow(dict(epoch=str(epoch), loss=str(total_loss/steps), acc='-1'))
    val_csvfile.flush()

    return total_loss / steps

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, os.path.join(args.log_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(args.log_dir, filename), os.path.join(args.log_dir,'model_best.pth'))


def run():
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        print("\n\n*******************************************************\n\n")
        tim = time.time()
        train(epoch)
        trtim = time.time()
        val_loss = validate(epoch)
        scheduler.step()
        print ("Val Loss (epoch", epoch, "): ", val_loss)
        print("Time to train: ", trtim - tim, " val time: ", time.time() - trtim)
        # is_best = val_loss < best_val_loss
        # best_val_loss = min(val_loss, best_val_loss)
        # save_checkpoint({
        #     'epoch': epoch,
        #     'model': args.model_type,
        #     'state_dict': model.state_dict(),
        #     'best_val_loss': best_val_loss,
        #     'optimizer' : optimizer.state_dict(),
        # }, is_best, filename='checkpoint_'+str(epoch)+'.pth')

if __name__ == "__main__":
    run()
