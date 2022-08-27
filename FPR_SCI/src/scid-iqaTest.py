#!/usr/bin/env python3
"""
Main Script
"""

import sys
import os

import shutil
import argparse

import torch
import torch.backends.cudnn as cudnn
from torch import nn

from model import IQANet
from dataset import TID2013Dataset, WaterlooDataset,IQADataset,SCIDDataset
from utils import AverageMeter, SROCC, PLCC, RMSE
from utils import SimpleProgressBar as ProgressBar
from utils import MMD_loss
from VIDLoss import VIDLoss


def test(test_data_loader, model):
    srocc = SROCC()
    plcc = PLCC()
    rmse = RMSE()
    len_test = len(test_data_loader)
    pb = ProgressBar(len_test, show_step=True)

    print("Testing")

    model.eval()
    with torch.no_grad():
        for i, ((img, ref), score) in enumerate(test_data_loader):
            img = img.cuda()
            ref = ref.cuda()
            output = model(img, img).cpu().data.numpy()
            score = score.data.numpy()

            srocc.update(score, output)
            plcc.update(score, output)
            rmse.update(score, output)

            pb.show(i, "Test: [{0:5d}/{1:5d}]\t"
                    "Score: {2:.4f}\t"
                    "Label: {3:.4f}"
                    .format(i+1, len_test, float(output), float(score)))

    print("\n\nSROCC: {0:.4f}\n"
            "PLCC: {1:.4f}\n"
            "RMSE: {2:.4f}"
            .format(srocc.compute(), plcc.compute(), rmse.compute())
    )



def test_iqa(args):
    batch_size = 1
    pro = args.pro
    num_workers = args.workers
    subset = args.subset
    data_dir = args.data_dir
    list_dir = args.list_dir
    resume = args.resume

    for k, v in args.__dict__.items():
        print(k, ':', v)

    model = IQANet(args.weighted)

    test_loader = torch.utils.data.DataLoader(
        Dataset(data_dir, phase='test_'+str(pro), list_dir=list_dir, 
        n_ptchs=args.n_ptchs_per_img,
        subset=subset), 
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    cudnn.benchmark = True

    # Resume from a checkpoint
    if resume:
        resume = resume.split('t.')[0]+'t_'+str(pro)+'.pkl'
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    test(test_loader, model.cuda())


def adjust_learning_rate(args, optimizer, epoch):
    """
    Sets the learning rate
    """
    if args.lr_mode == 'step':
        lr = args.lr * (0.5 ** (epoch // args.step))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.epochs) ** 1.1
    elif args.lr_mode == 'const':
        lr = args.lr
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_checkpoint(state, is_best, filename='checkpoint.pkl'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '../models/model_best.pkl')

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-cmd', type=str,default='test')
    parser.add_argument('-d', '--data-dir', default='../../../datasets/tid2013/')
    parser.add_argument('-l', '--list-dir', default='../sci_scripts/scid-scripts-6-2-2/',
                        help='List dir to look for train_images.txt etc. '
                             'It is the same with --data-dir if not set.')
    parser.add_argument('-n', '--n-ptchs-per-img', type=int, default=1024, metavar='N', 
                        help='number of patches for each image (default: 32)')
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32, metavar='B',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='NE',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--lr-mode', type=str, default='const')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--resume', default='../models/scid/model_best.pkl', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('--pro', type=int, default=2)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--subset', default='test')
    parser.add_argument('--evaluate', dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--weighted',default=True, dest='weighted')
    parser.add_argument('--dump_per', type=int, default=50, 
                        help='the number of epochs to make a checkpoint')
    parser.add_argument('--dataset', type=str, default='SCID')
    parser.add_argument('--anew', action='store_true')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    # Choose dataset
    global Dataset
    Dataset = globals().get(args.dataset+'Dataset', None)
    if args.cmd == 'train':
        train_iqa(args)
    elif args.cmd == 'test':
        test_iqa(args)


if __name__ == '__main__':
    main()
