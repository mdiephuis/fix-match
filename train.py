import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import numpy as np

from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
import time

from models import *
from utils import *
from data import *

parser = argparse.ArgumentParser(description='Fixmatch')

parser.add_argument('--uid', type=str, default='Fixmatch',
                    help='Staging identifier (default: Fixmatch)')
parser.add_argument('--dataset-name', type=str, default='CIFAR10C',
                    help='Name of dataset (default: CIFAR10C')
parser.add_argument('--data-dir', type=str, default='data',
                    help='Path to dataset (default: data')
parser.add_argument('--mu', type=int, default=7,
                    help='Fraction of unlabeled data (default: 2')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input training batch-size')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of training epochs (default: 150)')
parser.add_argument('--lr', type=float, default=0.03,
                    help='learning rate (default: 0.03')
parser.add_argument('--decay-lr', type=float, default=1e-3,
                    help='learning rate (default: 1e-3')
parser.add_argument('--gamma', type=float, default=1.0,
                    help='gamma loss balance (default: 1.0')
parser.add_argument('--log-dir', type=str, default='runs',
                    help='logging directory (default: runs)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables cuda (default: False')
parser.add_argument('--load-model', type=str, default=None,
                    help='Load model to resume training for (default None)')

args = parser.parse_args()

# Notes:
# - SGD with momentum of 0.9 works the best. lr : 0.03
# - the Nesterov variant of momentum [42] is not required for achieving an error below 5%
# - use a cosine lr decay
# - for the cosine learning rate decay, picking a proper decaying rate is important. Finally, using no decay results in worse accuracy
# - We find that tuning the weight decay is exceptionally important : CIFAR10 -> 0.0005

# Set cuda
use_cuda = not args.no_cuda and torch.cuda.is_available()

if use_cuda:
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda")
    torch.cuda.set_device(1)
    print('GPU')
else:
    dtype = torch.FloatTensor
    device = torch.device("cpu")

# Setup tensorboard
use_tb = args.log_dir is not None
log_dir = args.log_dir

# Setup asset directories
if not os.path.exists('models'):
    os.makedirs('models')

if not os.path.exists('runs'):
    os.makedirs('runs')

# Logger
if use_tb:
    logger = SummaryWriter(comment='_' + args.uid + '_' + args.dataset_name)

# data
loader = LoaderCIFAR(args.data_dir, True, args.batch_size, args.mu, use_cuda)


# train validate
def train(model, loader, optimizer, schedular, epoch, use_cuda):

    # TODO TWO LOSS FUNCTIONS
    loss_func = nn.CrossEntropyLoss()

    data_loader = zip(loader.train_labeled, loader.train_unlabeled)

    model.train()

    total_loss = 0.0

    tqdm_bar = tqdm(data_loader, total=len(loader.train_labeled))
    for batch_idx, (data_s, data_u) in enumerate(tqdm_bar):

        # labeled data
        x_i_s, y_s = data_s

        # unlabled_data
        x_i_u, x_j_u, _ = data_u

        # to cuda()
        x_i_s = x_i_s.cuda() if use_cuda else x_i_s
        x_i_u = x_i_u.cuda() if use_cuda else x_i_u
        x_j_u = x_j_u.cuda() if use_cuda else x_j_u

        y_s = y_s.cuda() if use_cuda else y_s

        # model forward supervised
        y_s_hat = model(x_i_s)

        # model forward non supervised, weak data
        y_i_u_hat = model(x_i_u).detach()

        # supervised loss
        loss_supervised = loss_func(y_s_hat, y_s)

        # Check confidence in prediction. If above threshold, use the strongly augmented pair
        with torch.no_grad():
            predictions = torch.softmax(y_i_u_hat, dim=1)
            score, labels = torch.max(predictions, dim=1)
            valid = score > 0.95

        if sum(valid) > 0:
            # Create pseudo labels for valid entries and select the matching correct strongly
            # augmented images
            y_pseudo = labels[valid]
            x_j_u = x_j_u[valid]

            # model forward for pseudo labeled strong augmented pairs
            y_j_hat = model(x_j_u)

            loss_unsupervised = loss_func(y_j_hat, y_pseudo)
        else:
            loss_unsupervised = 0

        loss = loss_supervised + (args.gamma * loss_unsupervised)

        schedular.step(7 * np.pi * (batch_idx) / (16 * len(loader.train_labeled)))

        model.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        tqdm_bar.set_description('Train: Epoch: [{}] Loss: {:.4f}'.format(epoch, loss.item()))

    return total_loss / (len(loader.train_labeled))


def validation(model, loader, optimizer, epoch, use_cuda):

    loss_func = nn.CrossEntropyLoss()

    data_loader = loader.test

    model.eval()

    total_loss = 0.0
    total_acc = 0.0

    tqdm_bar = tqdm(data_loader, total=len(loader.test))
    for batch_idx, (x, y) in enumerate(tqdm_bar):

        x = x.cuda() if use_cuda else x

        y = y.cuda() if use_cuda else y

        # model forward supervised
        y_hat = model(x)

        loss = loss_func(y_hat, y)

        total_loss += loss.item()

        # accuracy
        pred = y_hat.max(dim=1)[1]
        correct = pred.eq(y).sum().item()
        correct /= y.size(0)
        batch_acc = (correct * 100)
        total_acc += batch_acc

        tqdm_bar.set_description('Validation: Epoch: [{}] Loss: {:.4f} Acc: {:.4f}'.format(epoch, loss.item(), batch_acc))

    return total_loss / (len(loader.test)), total_acc / (len(loader.test))


def execute_graph(model, loader, optimizer, schedular, epoch, use_cuda):

    t_loss = train(model, loader, optimizer, schedular, epoch, use_cuda)
    v_loss, v_acc = validation(model, loader, optimizer, epoch, use_cuda)

    if use_tb:
        logger.add_scalar(log_dir + '/train-loss', t_loss, epoch)
        logger.add_scalar(log_dir + '/valid-loss', v_loss, epoch)

        logger.add_scalar(log_dir + '/valid-acc', v_acc, epoch)

    # print('Epoch: {} Train loss {}'.format(epoch, t_loss))
    # print('Epoch: {} Valid loss {}'.format(epoch, v_loss))

    return v_loss


model = resnet50_cifar().type(dtype)
init_weights(model)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
schedular = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=len(loader.train_labeled))


# Main training loop
best_loss = np.inf

# Resume training
if args.load_model is not None:
    if os.path.isfile(args.load_model):
        checkpoint = torch.load(args.load_model)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        schedular.load_state_dict(checkpoint['schedular'])
        best_loss = checkpoint['val_loss']
        epoch = checkpoint['epoch']
        print('Loading model: {}. Resuming from epoch: {}'.format(args.load_model, epoch))
    else:
        print('Model: {} not found'.format(args.load_model))

for epoch in range(args.epochs):
    v_loss = execute_graph(model, loader, optimizer, schedular, epoch, use_cuda)

    if v_loss < best_loss:
        best_loss = v_loss
        print('Writing model checkpoint')
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'schedular': schedular.state_dict(),
            'val_loss': v_loss
        }
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M', t)
        file_name = 'models/{}_{}_{}_{:04.4f}.pt'.format(timestamp, args.uid, epoch, v_loss)

        torch.save(state, file_name)


# TensorboardX logger
logger.close()

# save model / restart trainin
