#!/usr/bin/env python
# coding: utf-8

#	Source:
#	Tiling code inspired from
#	https://github.com/facebookresearch/moco
#	which is Copyright (c) Facebook, Inc. and its affiliates.
#	The code has been modified to analyze interstitial pneumonia by Wataru Uegami, MD

from datetime import datetime
from functools import partial
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.models import resnet
from tqdm import tqdm
import argparse
import json
import math
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from PIL import ImageFilter

# original util library
from mocotools import mocoutil

parser = argparse.ArgumentParser(description='Train MoCo on JF cases')

parser.add_argument('-a', '--arch', default='resnet18')

# lr: 0.06 for batch 512 (or 0.03 for batch 256)
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', 
                    help='initial learning rate', dest='lr')
parser.add_argument('--epochs', default=200, type=int, metavar='N', 
                    help='number of total epochs to run')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, 
                    help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
parser.add_argument('--cos', action='store_true', 
                    help='use cosine lr schedule')

parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N', 
                    help='mini-batch size')
parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int, 
                    help='feature dimension')
parser.add_argument('--moco-k', default=4096, type=int, 
                    help='queue size; number of negative keys') # original deafult 65536
parser.add_argument('--moco-m', default=0.99, type=float, 
                    help='moco momentum of updating key encoder')
parser.add_argument('--moco-t', default=0.1, type=float, 
                    help='softmax temperature')

parser.add_argument('--bn-splits', default=8, type=int, 
                    help='simulate multi-gpu behavior of BatchNorm in one gpu; 1 is SyncBatchNorm in multi-gpu')

parser.add_argument('--symmetric', action='store_true', 
                    help='use a symmetric loss function that backprops to both crops')

# utils
parser.add_argument('-d', '--data', default='', type=str, metavar='PATH',
                    help='path to data folder')
parser.add_argument('--resume', default='', type=str, metavar='PATH', 
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--results-dir', default='', type=str, metavar='PATH', 
                    help='path to cache (default: none)')


args = parser.parse_args()  # running in command line

if args.results_dir == '':
    args.results_dir = './cache-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-moco")

# these value need to be set for each dataset
normalize = transforms.Normalize(mean=[0.85, 0.7, 0.78],
                                  std=[0.15, 0.24, 0.2])

augmentation = [
     transforms.RandomRotation(90),  # rotate shoud be first
     transforms.CenterCrop(200),
     transforms.RandomApply([
        transforms.ColorJitter(brightness=0.07, # 0.4
                               contrast=0.15, # 0.4
                               saturation=0.6,
                               hue=0.03)  # not strengthened  # 0.1
    ], p=1),# 0.8
     transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip(),
     transforms.ToTensor(),
     normalize
]

train_dataset = datasets.ImageFolder(args.data,
    mocoutil.TwoCropsTransform(transforms.Compose(augmentation)))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=8, pin_memory=True, drop_last=True)

# create model
model = mocoutil.ModelMoCo(
        dim=args.moco_dim,
        K=args.moco_k,
        m=args.moco_m,
        T=args.moco_t,
        arch=args.arch,
        bn_splits=args.bn_splits,
        symmetric=args.symmetric,
    ).cuda()


# train for one epoch
def train(net, data_loader, train_optimizer, epoch, args):
    net.train()
    adjust_learning_rate(optimizer, epoch, args)

    total_loss = 0.0
    total_num = 0
    train_bar = tqdm(data_loader)


    for tb in train_bar:
        images, _ = tb
        im_1, im_2 = images[0].cuda(non_blocking=True), images[1].cuda(non_blocking=True)

        loss = net(im_1, im_2)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, args.epochs, optimizer.param_groups[0]['lr'], total_loss / total_num))

    return total_loss / total_num

# lr scheduler for training
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# define optimizer
#optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# load model if resume
epoch_start = 1
if args.resume is not '':
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch_start = checkpoint['epoch'] + 1
    print('Loaded from: {}'.format(args.resume))

# logging
results = {'train_loss': [], 'test_acc@1': []}
if not os.path.exists(args.results_dir):
    os.mkdir(args.results_dir)
# dump args
with open(args.results_dir + '/args.json', 'w') as fid:
    json.dump(args.__dict__, fid, indent=2)

# training loop
for epoch in range(epoch_start, args.epochs + 1):
    train_loss = train(model, train_loader, optimizer, epoch, args)
    if (epoch == 1) | (epoch % 10 == 0): 
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, '{}/epoch{}.pth'.format(args.results_dir, str(epoch)))
