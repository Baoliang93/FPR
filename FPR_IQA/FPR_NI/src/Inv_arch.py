import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Subnet_constructor import DenseBlock,DenseBlock1X1
from torch.autograd import Variable

class InvBlockExp(nn.Module):
    def __init__(self, split_len1, split_len2, clamp=1.0, Use1x1=False):
        super(InvBlockExp, self).__init__()

        self.split_len1 = split_len1
        self.split_len2 = split_len2

        self.clamp = clamp

        if not Use1x1:
            self.F = DenseBlock(self.split_len2, self.split_len1)
            self.G = DenseBlock(self.split_len1, self.split_len2)
            self.H = DenseBlock(self.split_len1, self.split_len2)
        else:
            self.F = DenseBlock1X1(self.split_len2, self.split_len1)
            self.G = DenseBlock1X1(self.split_len1, self.split_len2)
            self.H = DenseBlock1X1(self.split_len1, self.split_len2)
            

    def forward(self, x1, x2,rev=False):
        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return y1, y2

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]



class InvRescaleNet(nn.Module):
    def __init__(self, split_len1=32, split_len2=32, block_num=3,Use1x1=False):
        super(InvRescaleNet, self).__init__()
        operations = []
        for j in range(block_num):
            b = InvBlockExp(split_len1, split_len2,Use1x1=Use1x1)
            operations.append(b)
        self.operations = nn.ModuleList(operations)
       

    def forward(self, x1,x2, rev=False, cal_jacobian=False):
        out1 = x1
        out2 = x2
        jacobian = 0

        if not rev:
            for op in self.operations:
                out1,out2 = op.forward(out1, out2,rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out1, rev)
        else:
            for op in reversed(self.operations):
                out1,out2 = op.forward(out1, out2,rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out1, rev)

        if cal_jacobian:
            return out1,out2, jacobian
        else:
            return out1,out2
        
        
        
        

def test():

    net = InvRescaleNet(split_len1=32, split_len2=32, block_num=3)
    net.cuda()
  
    x1 = torch.randn(2,32,32,32)
    x1 = Variable(x1.cuda())
    
    x2 = torch.randn(2,32,32,32)
    x2 = Variable(x2.cuda())
    
#    x2 = Variable(x2.cuda())
#    y1,y2,y3,y4,y5,y6 = net.forward(x1,x1,x1,x1)
#    print(y1.shape,y2.shape,y3.shape,y4.shape,y5.shape,y6.shape)
    
    y1,y2 = net.forward(x1,x2)


    print(y1.shape,y2.shape)

    
if __name__== '__main__':
    test()            
            