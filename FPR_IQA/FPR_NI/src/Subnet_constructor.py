import torch
import torch.nn as nn
import torch.nn.functional as F
import module_util as mutil
from torch.autograd import Variable
class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5

#class DenseBlock1X1(nn.Module):
#    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
#        super(DenseBlock1X1, self).__init__()
#        self.conv1 = nn.Conv2d(channel_in, gc, 1, 1, 0, bias=bias)
#        self.conv2 = nn.Conv2d(channel_in + gc, gc,1,1, 0, bias=bias)
#        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 1, 1, 0, bias=bias)
#        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 1, 1, 0, bias=bias)
#        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 1, 1, 0, bias=bias)
#        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#
#        if init == 'xavier':
#            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
#        else:
#            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
#        mutil.initialize_weights(self.conv5, 0)
#
#    def forward(self, x):
#        x1 = self.lrelu(self.conv1(x))
#        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
#        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
#        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
#        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
#
#        return x5
    
    

class DenseBlock1X1(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock1X1, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_in*2, 1, 1, 0, bias=bias)
        self.conv2 = nn.Conv2d(channel_in*2, channel_in*2,1,1, 0, bias=bias)
        self.conv3 = nn.Conv2d(channel_in*2, channel_in*2, 1, 1, 0, bias=bias)
        self.conv4 = nn.Conv2d(channel_in*2, channel_in, 1, 1, 0, bias=bias)
    
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
      

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(x2))
        x4 = self.conv4(x3)
   

        return x4
    
def test():

    net = DenseBlock1X1(32,32)
    net.cuda()
  
    x1 = torch.randn(1,32,9,9)
    x1 = Variable(x1.cuda())


    
    y1 = net.forward(x1)


    print(y1.shape)

    
if __name__== '__main__':
    test()            
                
    
    
    
    
    
    
    
    
    
    