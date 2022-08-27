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
from dataset import  IQADataset,SCIDDataset
from utils import AverageMeter, SROCC, PLCC, RMSE
from utils import SimpleProgressBar as ProgressBar
from utils import MMD_loss
from VIDLoss import VIDLoss
#f=open("log.txt","a")
#ftmp=sys.stdout
#sys.stdout=f

def validate(val_loader, model, criterion, show_step=False):
    losses = AverageMeter()
    srocc = SROCC()
    len_val = len(val_loader)
    pb = ProgressBar(len_val, show_step=show_step)

    print("Validation")

    # Switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, ((img,ref), score)in enumerate(val_loader):
            img, ref, score = img.cuda(), ref.cuda(), score.squeeze().cuda()

            # Compute output
            _,_,output,_,_,_,_ = model(img, img)
            
            loss = criterion(output, score)
            losses.update(loss, img.shape[0])

            output = output.cpu()
            score = score.cpu()
            srocc.update(score.numpy(), output.numpy())

            pb.show(i, "[{0:d}/{1:d}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Output {out:.4f}\t"
                    "Target {tar:.4f}\t"
                    .format(i+1, len_val, loss=losses, 
                    out=output, tar=score))


    return float(1.0-srocc.compute())  # losses.avg
    

def train(train_loader, model, criterion, optimizer, epoch):
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    losses3 = AverageMeter()
    losses4 = AverageMeter()
    losses5 = AverageMeter()
#    losses6 = AverageMeter()
    len_train = len(train_loader)
    pb = ProgressBar(len_train)

    print("Training")

    # Switch to train mode
    model.train()
    criterion.cuda()
#    vidloss = VIDLoss(128,256,128).cuda()
    trip_loss = nn.TripletMarginLoss(margin=0.5, p=2.0).cuda()
#    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2).cuda()
    for i, ((img,ref), score)in enumerate(train_loader):
        img, ref, score = img.cuda(), ref.cuda(), score.cuda()
        # Compute output
        FS,NFake_FS,NS,f1,f2,fake_f1, fake_f2 = model(img, ref)      
        
        loss1 = criterion(FS, score) 
        loss2 = criterion(NFake_FS, score) 
        loss5 = criterion(NS, score)  
        loss3 = 20*trip_loss(f1,fake_f1,f2)
        loss4 = 20*trip_loss(f2,fake_f2,f1).sum()

     
        loss = loss1+loss2+loss3+loss4+loss5
        # Measure accuracy and record loss
        losses1.update(loss1.data, img.shape[0])
        losses2.update(loss2.data, img.shape[0])
        losses3.update(loss3.data, img.shape[0])
        losses4.update(loss4.data, img.shape[0])
        losses5.update(loss5.data, img.shape[0])

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pb.show(i, "[{0:d}/{1:d}]\t"
                "FR {loss1.val:.4f} ({loss1.avg:.4f})\t"
                "NR {loss2.val:.4f} ({loss2.avg:.4f})\t"
                "NS {loss5.val:.4f} ({loss5.avg:.4f})\t"
                "fco {loss3.val:.4f} ({loss3.avg:.4f})\t"
                "sf {loss4.val:.4f} ({loss4.avg:.4f})\t"

                .format(i+1, len_train, loss1=losses1, loss2=losses2, \
                        loss5=losses5,loss3=losses3,loss4=losses4))
                


def train_iqa(args):
    pro = args.pro
    batch_size = args.batch_size
    num_workers = args.workers
    data_dir = args.data_dir
    list_dir = args.list_dir
    resume = args.resume
    n_ptchs = args.n_ptchs_per_img

    print(' '.join(sys.argv))

    for k, v in args.__dict__.items():
        print(k, ':', v)

    model = IQANet(args.weighted,istrain=True)
    criterion = nn.L1Loss()

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        Dataset(data_dir, 'train_'+str(pro), list_dir=list_dir, 
        n_ptchs=n_ptchs),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        Dataset(data_dir, 'val_'+str(pro), list_dir=list_dir, 
        n_ptchs=n_ptchs, sample_once=True),
        batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True
    )

    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=args.lr, 
                                betas=(0.9, 0.999), 
                                weight_decay=args.weight_decay)
    
    cudnn.benchmark = True
    min_loss = 100.0
    start_epoch = 0

    # Resume from a checkpoint
    if resume:
        resume = resume.split('t.')[0]+'t_'+str(pro)+'.pkl'
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            if not args.anew:
                min_loss = checkpoint['min_loss']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    if args.evaluate:
        validate(val_loader, model.cuda(), criterion, show_step=True)
        return

    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args, optimizer, epoch)
        print("\nEpoch: [{0}]\tlr {1:.06f}".format(epoch, lr))
        # Train for one epoch
        train(train_loader, model.cuda(), criterion, optimizer, epoch)
        
        if epoch % 1 == 0:    
            # Evaluate on validation set
            loss = validate(val_loader, model.cuda(), criterion)
            
            is_best = loss < min_loss
            min_loss = min(loss, min_loss)
            print("Current: {:.6f}\tBest: {:.6f}\t".format(loss, min_loss))
            checkpoint_path = '../models/'+ resume.split('/')[2]+'/checkpoint_latest_'+str(pro)+'.pkl'
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'min_loss': min_loss,
            }, is_best, filename=checkpoint_path,pro =args.pro, res = resume)

            # if epoch % args.dump_per == 0:
            #     history_path = '../models/checkpoint_{:03d}_'.format(epoch+1)+str(pro)+'.pkl'
            #     shutil.copyfile(checkpoint_path, history_path)
            
            




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

def save_checkpoint(state, is_best, filename='checkpoint.pkl',pro = '0',res='./script'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '../models/'+ res.split('/')[2]+'/model_best_'+str(pro)+'.pkl')


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-cmd', type=str,default='train')
    parser.add_argument('-d', '--data-dir', default='../../../datasets/tid2013/')
    parser.add_argument('-l', '--list-dir', default='../sci_scripts/siqad-scripts-6-2-2/',
                        help='List dir to look for train_images.txt etc. '
                             'It is the same with --data-dir if not set.')
    parser.add_argument('-n', '--n-ptchs-per-img', type=int, default=8, metavar='N', 
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
    parser.add_argument('--resume', default='../models/siqad/checkpoint_latest.pkl', type=str, metavar='PATH',
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
    parser.add_argument('--dataset', type=str, default='IQA')
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
