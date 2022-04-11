import argparse
import operator
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import get_batch
from meta_optimizer import MetaModel, MetaOptimizer, FastMetaOptimizer
from model import Model
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--optimizer_steps', type=int, default=100, metavar='N',
                    help='number of meta optimizer steps (default: 100)')
parser.add_argument('--truncated_bptt_step', type=int, default=20, metavar='N',
                    help='step at which it truncates bptt (default: 20)')
parser.add_argument('--updates_per_epoch', type=int, default=10, metavar='N',
                    help='updates per epoch (default: 100)')
parser.add_argument('--max_epoch', type=int, default=10000, metavar='N',
                    help='number of epoch (default: 10000)')
parser.add_argument('--hidden_size', type=int, default=10, metavar='N',
                    help='hidden size of the meta optimizer (default: 10)')
parser.add_argument('--num_layers', type=int, default=2, metavar='N',
                    help='number of LSTM layers (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

assert args.optimizer_steps % args.truncated_bptt_step == 0

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, drop_last=False, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, drop_last=False, **kwargs)

def main():
    # Create a meta optimizer that wraps a model into a meta model
    # to keep track of the meta updates.
    meta_model = Model()
    if args.cuda:
        meta_model.to(device)

    meta_optimizer = FastMetaOptimizer(MetaModel(meta_model), args.num_layers, args.hidden_size)
    if args.cuda:
        meta_optimizer.to(device)

    optimizer = optim.Adam(meta_optimizer.parameters(), lr=1e-3)

    # 可视化
    final_loss_list = []

    args.max_epoch = 2 # debug
    for epoch in range(args.max_epoch):
        decrease_in_loss = 0.0
        final_loss = 0.0
        train_iter = iter(train_loader)

        for i in range(args.updates_per_epoch):

            # Sample a new model
            model = Model()
            if args.cuda:
                model.to(device)

            x, y = next(train_iter)
            if args.cuda:
                x, y = x.to(device), y.to(device)
            x, y = Variable(x), Variable(y)
            # print('x.shape: ', x.shape) # x.shape:  torch.Size([32(batch_size), 1(RGB通道数), 28, 28])

            # Compute initial loss of the model
            f_x = model(x)
            initial_loss = F.nll_loss(f_x, y)

            for k in range(args.optimizer_steps // args.truncated_bptt_step):
                # Keep states for truncated BPTT
                meta_optimizer.reset_lstm(
                    keep_states=k > 0, model=model, use_cuda=args.cuda)

                loss_sum = 0
                prev_loss = torch.zeros(1)
                if args.cuda:
                    prev_loss = prev_loss.to(device)
                for j in range(args.truncated_bptt_step):
                    try:
                        x, y = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        x, y = next(train_iter)

                    if args.cuda:
                        x, y = x.to(device), y.to(device)
                    x, y = Variable(x), Variable(y)

                    # First we need to compute the gradients of the model
                    f_x = model(x)
                    # print(f_x.shape) # torch.Size([32, 10])
                    loss = F.nll_loss(f_x, y)
                    # print(loss)
                    model.zero_grad()
                    loss.backward()

                    # Perfom a meta update using gradients from model
                    # and return the current meta model saved in the optimizer
                    meta_model = meta_optimizer.meta_update(model, loss.data)

                    # Compute a loss for a step the meta optimizer
                    f_x = meta_model(x)
                    loss = F.nll_loss(f_x, y)

                    loss_sum += (loss - Variable(prev_loss))

                    prev_loss = loss.data

                # Update the parameters of the meta optimizer
                meta_optimizer.zero_grad()
                loss_sum.backward()
                for param in meta_optimizer.parameters():
                    param.grad.data.clamp_(-1, 1)  # clamp函数把值固定在-1到1之间，超过最大值和最小值直接按照最大值最小值处理
                optimizer.step()

            # Compute relative decrease in the loss function w.r.t initial
            # value

            decrease_in_loss += loss.item() / initial_loss.item()
            final_loss += loss.item()

        final_loss_list.append(final_loss / args.updates_per_epoch)
        # print(len(final_loss_list))
        print("Epoch: {}, final loss {}, average final/initial loss ratio: {}".format(epoch, final_loss / args.updates_per_epoch,
                                                                       decrease_in_loss / args.updates_per_epoch))
    final_loss_set = torch.tensor(final_loss_list, dtype=torch.float32)
    plt.plot(torch.arange(0, args.max_epoch), final_loss_set)
    plt.show()

if __name__ == "__main__":
    main()
