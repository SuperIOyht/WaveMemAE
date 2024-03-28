import torch.nn as nn
import torch
import torch.nn.modules
from memory import MemModule_ForLoss
import wavelet

# WaveletMemAE
class WaveletMemAE(nn.Module):
    def __init__(self,mem_dim=64,fea_dim=512*2*2,tempreture=0.1,shrink_thres=0.0025):
        super().__init__()
        # encoder
        self.conv1 = nn.Conv2d(3,32,(3,3),stride=2,padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.relu  = nn.LeakyReLU(0.2,inplace=True)
        self.conv2 = nn.Conv2d(32,64,3,stride=2,padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128,3,stride=2,padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,256,3,stride=2,padding=1)
        self.bn4   = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256,512,3,stride=2,padding=1)
        self.bn5   = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512,1024,3,stride=2,padding=1)
        self.bn6   = nn.BatchNorm2d(1024)
        
        
        #encoder_H
        self.conv_H1 = nn.Conv2d(3*3,32*3,(3,3),stride=2,padding=1)
        self.bn_H1   = nn.BatchNorm2d(32*3)
        self.conv_H2 = nn.Conv2d(32*3,64*3,3,stride=2,padding=1)
        self.bn_H2   = nn.BatchNorm2d(64*3)
        self.conv_H3 = nn.Conv2d(64*3,128*3,3,stride=2,padding=1)
        self.bn_H3   = nn.BatchNorm2d(128*3)
        self.conv_H4 = nn.Conv2d(128*3,256*3,3,stride=2,padding=1)
        self.bn_H4   = nn.BatchNorm2d(256*3)
        self.conv_H5 = nn.Conv2d(256*3,512*3,3,stride=2,padding=1)
        self.bn_H5   = nn.BatchNorm2d(512*3)
        self.conv_H6 = nn.Conv2d(512*3,1024*3,3,stride=2,padding=1)
        self.bn_H6   = nn.BatchNorm2d(1024*3)
        
        #memory
        self.mem_dim = mem_dim
        self.mem_rep = MemModule_ForLoss(
            mem_dim=mem_dim,
            fea_dim=fea_dim,
            tempreture = tempreture,
            shrink_thres=shrink_thres)

        # decoder 
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv1 = nn.Conv2d(1024,512,3,stride=1,padding=1)
        self.debn1 = nn.BatchNorm2d(512)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv2 = nn.Conv2d(512,256,3,stride=1,padding=1)
        self.debn2 = nn.BatchNorm2d(256)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv3 = nn.Conv2d(256,128,3,stride=1,padding=1)
        self.debn3 = nn.BatchNorm2d(128)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv4 = nn.Conv2d(128,64,3,stride=1,padding=1)
        self.debn4 = nn.BatchNorm2d(64)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv5 = nn.Conv2d(64,32,3,stride=1,padding=1)
        self.debn5 = nn.BatchNorm2d(32)
        self.up6 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv6 = nn.Conv2d(32,3,3,stride=1,padding=1)
        self.final = nn.Sigmoid()

        #wavelet
        self.waveletDecompose = wavelet.WavePool_Two(in_channels=3).cuda()
        self.waveletCompose = wavelet.WaveUnpool(1024, 'sum')
        
        #conv
        #self.conv_connection_1 = nn.Conv2d(2, 1, 1,stride=1, padding=0)
        self.sig = nn.Sigmoid()
    def freq_connection(self, freq_h, freq_l):
        x = torch.cat((freq_h,freq_l), dim=1)
        b, c, h, w = x.shape
        conv_connection = nn.Conv2d(c, 1, 1, stride=1, padding=0).cuda()
        x = conv_connection(x)
        x = self.sig(x)
        x = x * freq_l 
        x = x.repeat(1, 3, 1, 1)
        return x
    
    def encoder_H(self, freq_h, feat_1, feat_2, feat_3, feat_4, feat_5):
        x = freq_h
        x = self.conv_H1(x)
        x = self.relu(self.bn_H1(x))
        x = self.conv_H2(self.freq_connection(x, feat_1) + x)
        x = self.relu(self.bn_H2(x))
        x = self.conv_H3(self.freq_connection(x, feat_2) + x)
        x = self.relu(self.bn_H3(x))
        x = self.conv_H4(self.freq_connection(x, feat_3) + x)
        x = self.relu(self.bn_H4(x))
        x = self.conv_H5(self.freq_connection(x, feat_4) + x)
        x = self.relu(self.bn_H5(x))
        x = self.conv_H6(self.freq_connection(x, feat_5) + x)
        x = self.relu(self.bn_H6(x))
        return x
    
    def encoder_L(self, freq_l):
        x = freq_l
        x = self.conv1(x) #->[32, 32, 256, 256]
        x = self.relu(self.bn1(x))
        feat_1 = x
        x = self.conv2(x) #->[32, 64, 128, 128]
        x = self.relu(self.bn2(x))
        feat_2 = x
        x = self.conv3(x) #->[32, 128, 64, 64]
        x = self.relu(self.bn3(x))
        feat_3 = x
        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        feat_4 = x
        x = self.conv5(x)
        x = self.relu(self.bn5(x))
        feat_5 = x
        x = self.conv6(x)
        x = self.relu(self.bn6(x))
        return x, feat_1, feat_2, feat_3, feat_4, feat_5
    
    def decoder(self, x):
        x = self.up1(x) #[-1, 512, 32, 32]
        # [-1, 512, 32, 32] -> [-1, 256, 64, 64]
        x = self.relu(self.debn1(self.deconv1(x)))
        x = self.up2(x)
        # [-1, 256, 64, 64] -> [-1, 128, 128, 128]
        x = self.relu(self.debn2(self.deconv2(x))) 
        x = self.up3(x)
        # [-1, 128, 128, 128] -> [-1, 64, 256, 256] 
        x = self.relu(self.debn3(self.deconv3(x)))
        x = self.up4(x)
        x = self.relu(self.debn4(self.deconv4(x)))
        x = self.up5(x)
        x = self.relu(self.debn5(self.deconv5(x)))
        x = self.up6(x)
        x = self.deconv6(x)
        x = self.final(x)
        return x
    
    def forward(self,x):
        LL, LH, HL, HH = self.waveletDecompose(x)
        freq_h = torch.cat((LH,HL,HH),dim=1)
        freq_l = LL
        x_LL, feat_1, feat_2, feat_3, feat_4, feat_5 = self.encoder_L(freq_l)
        x_H = self.encoder_H(freq_h, feat_1, feat_2, feat_3, feat_4, feat_5)
        x_LH = x_H[:, 0:1024, :, :]
        x_HL = x_H[:, 1024:2048, :, :]
        x_HH = x_H[:, 2048:3072, :, :]
        encoder_com = self.waveletCompose(x_LL, x_LH, x_HL, x_HH)
        b,c,h,w = encoder_com.shape
        encoder_com = encoder_com.view(b, -1)
        res_mem = self.mem_rep(encoder_com)
        f = res_mem['output']
        f = f.view(b,c,h,w)
        decoder = self.decoder(f)
        return decoder, encoder_com.view(b,c,w,h), f, res_mem['att'], res_mem['membank']
    
