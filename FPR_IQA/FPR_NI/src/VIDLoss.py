from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class VIDLoss(nn.Module):
    """Variational Information Distillation for Knowledge Transfer (CVPR 2019),
    code from author: https://github.com/ssahn0215/variational-information-distillation"""
    def __init__(self,
                 num_input_channels,
                 num_mid_channel,
                 num_target_channels,
                 init_pred_var=5.0,
                 eps=1e-5):
        super(VIDLoss, self).__init__()

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, padding=0,
                bias=False, stride=stride)

        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_target_channels),
        )
        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(init_pred_var-eps)-1.0) * torch.ones(num_target_channels)
            )
        self.eps = eps

    def forward(self, input, target):
        # pool for dimentsion match
        s_H, t_H = input.shape[2], target.shape[2]
        if s_H > t_H:
            input = F.adaptive_avg_pool2d(input, (t_H, t_H))
        elif s_H < t_H:
            target = F.adaptive_avg_pool2d(target, (s_H, s_H))
        else:
            pass
        pred_mean = self.regressor(input)
        pred_var = torch.log(1.0+torch.exp(self.log_scale))+self.eps
        pred_var = pred_var.view(1, -1, 1, 1)
        neg_log_prob = 0.5*(
            (pred_mean-target)**2/pred_var+torch.log(pred_var)
            )
        loss = torch.mean(neg_log_prob)
        return loss
    


    
def test():

    x1 = torch.randn(2, 128,4,4)
    x1 = Variable(x1.cuda())
    net = VIDLoss(128,256,128)
    net.cuda()
  
    
#    x2 = Variable(x2.cuda())
#    y1,y2,y3,y4,y5,y6 = net.forward(x1,x1,x1,x1)
#    print(y1.shape,y2.shape,y3.shape,y4.shape,y5.shape,y6.shape)
    
    y1 = net.forward(x1,x1)


    print(y1)

    
if __name__== '__main__':
    test()     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    