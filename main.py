
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
config.read('../config.ini')

base_dir = config['DEFAULT']['BASE_DIR']
sys.path.append(base_dir + '/models/')

sys.path.append(base_dir + '/tasks/addition/')
sys.path.append(base_dir + '/tasks/bouncing_ball/')
sys.path.append(base_dir + '/tasks/multiplication/')
sys.path.append(base_dir + '/tasks/xor/')
sys.path.append(base_dir + '/tasks/sequential_mnist/')
sys.path.append(base_dir + '/tasks/noiseless_memorization/')


sys.path.append(base_dir + '/tasks/copy_task/')

'''

To Do's
Make sure all tasks work as expected -- with accuracy
Make sure new two google brain tasks work

Write visualization classes
Update Readme

'''
#get the datasets of iterest
from addition import Addition
from noiseless_memorization import NoiselessMemorization
from bouncing_ball import BouncingBall
from multiplication import Multiplication
from noiseless_memorization import NoiselessMemorization
from sequential_mnist import SequentialMNIST
from xor import XOR

from simple_lstm import SimpleLSTM, SimpleRNN
from spectral_lstm import SpectralLSTM
from svd_lstm import SvdLSTM

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
                    help='use rnn, lstm, slstm, svdlstm')
parser.add_argument('--task', type=str, default='mem',
                    help='use add, mul, mem, xor, bball, seqmnist, strokemnist')
parser.add_argument('--output-weighting', action='store_true', default=False,
                    help='use row wise weighting for lstm output')
parser.add_argument('--orthogonal-init', action='store_true', default=False,
                    help='use initialize weights orthogonally')
parser.add_argument('--sequence-len', type=int, default=5,
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

fields = ['epoch', 'loss', 'avg_loss']
train_csvfile = open(os.path.join(args.log_dir, 'train.csv'), 'w')
train_csvwriter = csv.DictWriter(train_csvfile, fieldnames=fields)
train_csvwriter.writeheader()
train_csvfile.flush()

fields_val = ['epoch', 'loss', 'avg_loss', 'acc']
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
        activation = nn.Sigmoid()
        criterion = nn.BCELoss()
    elif args.task == 'xor':
        activation = nn.Sigmoid()
        criterion = nn.MSELoss()
    elif args.task == 'bball':
        # NOTE: want sigmoid because each pixel output could be a prob
        activation = log_sigmoid
        criterion = nn.KLDivLoss()
    elif args.task == 'seqmnist' or args.task == 'strokemnist':
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
        dset = CopyTask(sequence_len=args.sequence_len)
    elif args.task == 'xor':
        dset = XOR()
    elif args.task == 'bball':
        dset = BouncingBall(vectorize=True)
    elif args.task == 'seqmnist':
        dset = SequentialMNIST()
    elif args.task == 'strokemnist':
        dset = StrokeMNIST()
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
                                     layers=args.layers,
                                     orthogonal=args.orthogonal_init)
    elif args.model_type == 'lstm':
        return SimpleLSTM(input_size=dset.input_dimension,
                                      hidden_size=args.hx,
                                      output_size=dset.output_dimension,
                                      layers=args.layers,
                                      orthogonal=args.orthogonal_init)
    elif args.model_type == 'slstm':
        return SpectralLSTM(input_size=dset.input_dimension,
                                          hidden_size=args.hx,
                                          output_size=dset.output_dimension,
                                          output_weighting=args.output_weighting,
                                          layers=args.layers,
                                          orthogonal=args.orthogonal_init)
    elif args.model_type == 'svdlstm':
        return SvdLSTM(input_size=dset.input_dimension,
                                          hidden_size=args.hx,
                                          output_size=dset.output_dimension,
                                          output_weighting=args.output_weighting,
                                          layers=args.layers,
                                          use_learned_decomposition=True,
                                          orthogonal=args.orthogonal_init)
    else:
        raise Exception


model = create_model()
params = 0
for p in list(model.parameters()):
    params += p.numel()
print ("Num params: ", params)
print (model)
if args.cuda:
    model.cuda()
optimizer = optim.RMSprop(model.parameters(), lr=args.lr) #, momentum=args.momentum, alpha=args.alpha)
scheduler = ExponentialLR(optimizer, 0.95)


def run_sequence(seq):
    outputs = []
    model.reset(batch_size=seq.size(0), cuda=args.cuda)
    for i, input_t in enumerate(seq.chunk(seq.size(1), dim=1)):
        input_t = input_t.squeeze(1)
        p = activation(model(input_t))

        outputs.append(p)
    outputs = outputs[5:]
    outputs = torch.stack(outputs, 1)

    return outputs


def train(epoch):
    model.train()
    dset.train()

    total_loss = 0.0
    steps = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        target = target[:,5:]
        if args.task == 'strokemnist':
            mask = data[:, :, 4] # batch, timesteps, dim
            data = data[:, :, 0:4] # batch, timesteps, dim
        optimizer.zero_grad()
        output = run_sequence(data)

        # if args.task == 'strokemnist':
        if args.task == 'seqmnist' or args.task == 'strokemnist':
            target = target.expand(output.size()[0], output.size()[1])
            # print (target.size())
            # input("")
            # target = mask * target
            # input("target masked")
            # mask_output = mask.unsqueeze(2)
            # mask_output = mask_output.expand(*output.size())
            # output = mask_output * output

        if isinstance(criterion, nn.CrossEntropyLoss):
            # Reshape for cross entropy of sequence
            output = output.contiguous().view(-1) #, output.size(2))
            target = target.contiguous().view(-1).long()

        loss = criterion(output, target)

        if args.task == 'strokemnist':
            loss = loss.view(data.size()[0], data.size()[1])
            loss = loss * mask

        if args.task == 'strokemnist':
            # now get new mask for exp loss
            loss = loss.view(data.size()[0], data.size()[1])
            loss_mask = Variable(torch.zeros(loss.size()))
            for i in range(data.size()[0]):
                t = int(mask[i].sum())
                loss_mask[i, :t] = Variable(torch.exp(-(t-1-torch.arange(t))))
            loss = loss * loss_mask
            loss = loss.mean()

        if args.task == 'seqmnist' or args.task == 'strokemnist':
            loss = loss.mean()

        loss.backward()
        # print ("Loss: ", loss.cpu().data.numpy()[0])
        optimizer.step()
        steps += 1
        total_loss += loss.cpu().data.numpy()[0]

        train_csvwriter.writerow(dict(epoch=str(epoch), loss=str(loss.cpu().data.numpy()[0]), avg_loss="-1"))
        train_csvfile.flush()

        # attempt to preserve memory
        del data, target, output, loss
    print("Train loss ", total_loss/steps)

    train_csvwriter.writerow(dict(epoch=str(epoch), loss="-1", avg_loss=str(total_loss/steps)))
    train_csvfile.flush()


def validate(epoch):
    dset.val()
    model.eval()

    total_loss = 0.0
    n_correct = 0
    n_total = 0
    steps = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # with torch.no_grad():
        target = target[:,5:]
        data, target = Variable(data, volatile=True), Variable(target, volatile=True)
        if args.task == 'seqmnist' or args.task == 'strokemnist':
            mask = data[:, :, 4] # batch, timesteps, dim
            data = data[:, :, 0:4] # batch, timesteps, dim
        output = run_sequence(data)

        # # print (output[:,-1,:].data.size(), target.size())
        # pred = output[:,-1,:].data.max(1, keepdim=True)[1].long() # get the index of the max log-probability
        # # print (pred.size())
        # n_total += data.size()[0] # batch size
        # batch_correct = pred.eq(target.data.view_as(pred).long()).cpu().sum()
        # n_correct += batch_correct
        # batch_accuracy = batch_correct / data.size()[0]

        if args.task == 'seqmnist' or args.task == 'strokemnist':
            # expand classification...
            target = target.expand(output.size()[0], output.size()[1]).contiguous()
            target = mask * target
            # input("target masked")
            mask_output = mask.unsqueeze(2)
            mask_output = mask_output.expand(*output.size())
            output = mask_output * output

        if isinstance(criterion, nn.CrossEntropyLoss):
            # Reshape for cross entropy of sequence
            output = output.view(-1, output.size(2))
            target = target.view(-1).long()

        loss = criterion(output, target)

        if args.task == 'strokemnist':
            loss = loss.view(data.size()[0], data.size()[1])
            loss_mask = Variable(torch.zeros(loss.size()))
            for i in range(data.size()[0]):
                t = int(mask[i].sum())
                loss_mask[i, :t] = Variable(torch.exp(-(t-1-torch.arange(t))))
            loss = loss * loss_mask

        if args.task == 'seqmnist' or args.task == 'strokemnist':
            loss = loss.mean()

        total_loss += loss.cpu().data.numpy()[0]
        steps += 1

        val_csvwriter.writerow(dict(epoch=str(epoch), loss=str(loss.cpu().data.numpy()[0]), avg_loss="-1", acc='-1')) #str(batch_accuracy)))
        val_csvfile.flush()

        # attempt to preserve memory
        del data, target, output, loss

    # print ("Acc: ", n_correct / n_total)
    val_csvwriter.writerow(dict(epoch=str(epoch), loss="-1", avg_loss=str(total_loss/steps), acc='-1')) #str(n_correct/n_total)))
    val_csvfile.flush()

    return total_loss / steps


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, os.path.join(args.log_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(args.log_dir, filename), os.path.join(args.log_dir,'model_best.pth'))

def run():
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        train(epoch)
        val_loss = validate(epoch)
        scheduler.step()
        print ("Val Loss (epoch", epoch, "): ", val_loss)
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        save_checkpoint({
            'epoch': epoch,
            'model': args.model_type,
            'state_dict': model.state_dict(),
            'best_val_loss': best_val_loss,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename='checkpoint_'+str(epoch)+'.pth')

if __name__ == "__main__":
    run()
