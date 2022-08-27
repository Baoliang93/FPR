"""
The CNN Model for FR-IQA
-------------------------

KVASS Tastes good!
"""

import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from Inv_arch import InvRescaleNet

class Conv3x3(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Conv3x3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=(1,1), padding=(1,1), bias=True), 
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class MaxPool2x2(nn.Module):
    def __init__(self):
        super(MaxPool2x2, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=(2,2), padding=(0,0))
    
    def forward(self, x):
        return self.pool(x)

class DoubleConv(nn.Module):
    """
    Double convolution as a basic block for the net

    Actually this is from a VGG16 block
    """
    def __init__(self, in_dim, out_dim,ispool = True):
        super(DoubleConv, self).__init__()
        self.conv1 = Conv3x3(in_dim, out_dim)
        self.conv2 = Conv3x3(out_dim, out_dim)
        self.pool = MaxPool2x2()
        self.ispool = ispool

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.ispool:
            y = self.pool(y)
        return y

class SingleConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SingleConv, self).__init__()
        self.conv = Conv3x3(in_dim, out_dim)
        self.pool = MaxPool2x2()

    def forward(self, x):
        y = self.conv(x)
        y = self.pool(y)
        return y


class IQANet(nn.Module):
    """
    The CNN model for full-reference image quality assessment
    
    Implements a siamese network at first and then there is regression
    """
    def __init__(self, weighted=False,istrain=False,scale=4,\
                 block_num =3,channel_input=256):
        super(IQANet, self).__init__()

        self.weighted = weighted
        self.istrain = istrain
        self.scale = scale
        
        # Feature extraction layers
        self.fl1 = DoubleConv(3, 64)
        self.fl2 = DoubleConv(64, 128)
        self.fl3 = DoubleConv(128, 256)
     
        
        self.sfl1 = DoubleConv(3, 32*self.scale)
        self.sfl21 = DoubleConv(32*self.scale, 64*self.scale,ispool = False)
        self.sfl22 = DoubleConv(64*self.scale, 64*self.scale)
        self.sfl23 = DoubleConv(64*self.scale, 64*self.scale,ispool = False)
        self.sfl3 = DoubleConv(64*self.scale, 128*4)
        
        self.InvRescaleNet = InvRescaleNet(split_len1=channel_input, \
                                           split_len2=channel_input, \
                                           block_num=block_num,\
                                           Use1x1 = True)
        # Fusion layers
        self.cl1 = SingleConv(256*2, 128)
        self.cl2 = nn.Conv2d(128, 64, kernel_size=3)

        # Regression layers
        self.rl1 = nn.Linear(256, 32)
        self.rl2 = nn.Linear(32, 1)
        
        
        # Fusion layers
        self.scl1 = SingleConv(512, 128)
        self.scl2 = nn.Conv2d(128, 64, kernel_size=3)

        # Regression layers
        self.srl1 = nn.Linear(256, 32)
        self.srl2 = nn.Linear(32, 1)        
        
        self.gn=torch.nn.GroupNorm(num_channels=256,num_groups=64)

        if self.weighted:
            self.wl1 = nn.GRU(256, 32, batch_first=True)
            self.wl2 = nn.Linear(32, 1)
            
            self.swl1 = nn.GRU(256, 32, batch_first=True)
            self.swl2 = nn.Linear(32, 1)


        self._initialize_weights()
  
    def _get_initial_state(self, batch_size):
            h0 = torch.zeros(1, batch_size, 32,device=0)
            return h0  
        
    def extract_feature(self, x):
        """ Forward function for feature extraction of each branch of the siamese net """
        y = self.fl1(x)
        y = self.fl2(y)
        y = self.fl3(y)
#        y = self.gn(y)

        return y
   
    def NR_extract_feature(self, x):
        """ Forward function for feature extraction of each branch of the siamese net """
        y = self.sfl1(x)
        y = self.sfl21(y)
        y = self.sfl22(y)
        y = self.sfl23(y)
        y = self.sfl3(y)
        y1,y2 = torch.split(y, int(y.shape[1]/2), dim=1) 
  

        return y1,y2    
   
    def gaussian_batch(self, dims,scale=1):
        lenth = dims[0]*dims[1]*dims[2]*dims[3]
        inv = torch.normal(mean=0, std=0.5*torch.ones(lenth)).cuda()
        return inv.view_as(torch.Tensor(dims[0],dims[1],dims[2],dims[3]))
    

    def forward(self, x1, x2):
        """ x1 as distorted and x2 as reference """
        n_imgs, n_ptchs_per_img = x1.shape[0:2]
     
        
        # Reshape
        x1 = x1.view(-1,*x1.shape[-3:])
        x2 = x2.view(-1,*x2.shape[-3:])

        f1 = self.extract_feature(x1)
        f2 = self.extract_feature(x2)
        sf1,sf2 = self.NR_extract_feature(x1)
        fake_f1, fake_f2 = self.InvRescaleNet(sf1,sf2)
        
        ini_f_com = torch.cat([f2, f1], dim=1) 
        fake_f_com = torch.cat([fake_f2, fake_f1], dim=1)
        f_com = torch.cat([ini_f_com,fake_f_com], dim=0) 
        
        f_com = self.cl1(f_com)
        f_com = self.cl2(f_com)
        flatten = f_com.view(f_com.shape[0], -1)
        y = self.rl1(flatten)
        y = self.rl2(y)
        y1,y2 = torch.split(y, int(y.shape[0]/2), dim=0) 
        
        fake_sf1,fake_sf2 = self.InvRescaleNet(f1,f2, rev=True) 
        sf = torch.cat((sf1,sf2),1)
        
        
        NF_com = self.scl1(sf)
        NF_com = self.scl2(NF_com)
        Nflatten = NF_com.view(NF_com.shape[0], -1)
        Ny = self.srl1(Nflatten)
        Ny = self.srl2(Ny)
        
        if self.weighted:
            # print('use weighted')
            flatten = flatten.view(2*n_imgs, n_ptchs_per_img,-1)
            w,_ = self.wl1(flatten)
            w = self.wl2(w)
            w = torch.nn.functional.relu(w) + 1e-8
            # Weighted average            
            w1,w2 = torch.split(w, int(w.shape[0]/2), dim=0)
            
            y1_by_img = y1.view(n_imgs, n_ptchs_per_img)
            w1_by_img = w1.view(n_imgs, n_ptchs_per_img)
            FS = torch.sum(y1_by_img*w1_by_img, dim=1) / torch.sum(w1_by_img, dim=1)
            
            y2_by_img = y2.view(n_imgs, n_ptchs_per_img)
            w2_by_img = w2.view(n_imgs, n_ptchs_per_img)
            NFake_FS = torch.sum(y2_by_img*w2_by_img, dim=1) / torch.sum(w2_by_img, dim=1)
            
            
            
            Nflatten = Nflatten.view(n_imgs, n_ptchs_per_img,-1)
            sw,_ = self.swl1(Nflatten,self._get_initial_state(Nflatten.size(0)))
            sw = self.swl2(sw)
            sw = torch.nn.functional.relu(sw) + 1e-8
            Ny_by_img = Ny.view(n_imgs, n_ptchs_per_img)
            Nw_by_img = sw.view(n_imgs, n_ptchs_per_img)
            NS = torch.sum(Ny_by_img*Nw_by_img, dim=1) / torch.sum(Nw_by_img, dim=1)
            
            
        else:
            print('not use weighted')
            # Calculate average score for each image
            FS = torch.mean(y1.view(n_imgs, n_ptchs_per_img), dim=1)
            NFake_FS = torch.mean(y2.view(n_imgs, n_ptchs_per_img), dim=1)
            NS = torch.mean(Ny.view(n_imgs, n_ptchs_per_img), dim=1)

        if self.istrain:
           return FS.squeeze(),NFake_FS.squeeze(),NS.squeeze(),\
                 f1,f2,fake_f1, fake_f2             
    
        else:
           return NS.squeeze()        
        
        

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            else:
                pass
            
                        
def test():

    net = IQANet(weighted=True,istrain=True)
    net.cuda()
  
    x1 = torch.randn(2, 16,3,64,64)
    x1 = Variable(x1.cuda())
    
    y1,y2,y3,y4,y5,y6,y7= net.forward(x1,x1)


    print(y1.shape,y2.shape,y3.shape,y4.shape,y5.shape,y6.shape,y7.shape)

    
if __name__== '__main__':
    test()                 
            
            
            
