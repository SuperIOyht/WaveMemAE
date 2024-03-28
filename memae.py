from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
import torchvision as tv
import torch
import torch.nn.modules
import torch_dct

from .memory import MemModule, MemoryUnit, MemModule_ForLoss, MemModule_ForLoss_notemp
import wavelet

class AutoEncoderFCMem(nn.Module):
    def __init__(self,mem_dim=1024,shrink_thres=0.0025):
        super(AutoEncoderFCMem,self).__init__()
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

        self.mem_dim = mem_dim
        self.mem_rep = MemModule(
            mem_dim=mem_dim,
            fea_dim=512,
            shrink_thres=shrink_thres)

        # decoder 
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv1 = nn.Conv2d(512,256,3,stride=1,padding=1)
        self.debn1 = nn.BatchNorm2d(256)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv2 = nn.Conv2d(256,128,3,stride=1,padding=1)
        self.debn2 = nn.BatchNorm2d(128)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv3 = nn.Conv2d(128,64,3,stride=1,padding=1)
        self.debn3 = nn.BatchNorm2d(64)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv4 = nn.Conv2d(64,32,3,stride=1,padding=1)
        self.debn4 = nn.BatchNorm2d(32)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv5 = nn.Conv2d(32,1,3,stride=1,padding=1)
        self.final = nn.Tanh()
    def encoder(self,x):
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        x = self.conv3(x)
        x = self.relu(self.bn3(x))
        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        x = self.conv5(x)
        x = self.relu(self.bn5(x))
        return x
    def decoder(self,x):
        x = self.up1(x)
        x = self.relu(self.debn1(self.deconv1(x)))
        x = self.up2(x)
        x = self.relu(self.debn2(self.deconv2(x)))
        x = self.up3(x)
        x = self.relu(self.debn3(self.deconv3(x)))
        x = self.up4(x)
        x = self.relu(self.debn4(self.deconv4(x)))
        x = self.up5(x)
        x = self.deconv5(x)
        x = self.final(x)
        return x
    def forward(self,x):
        encoder = self.encoder(x)

        res_mem = self.mem_rep(encoder)
        f = res_mem['output']
        att = res_mem['att']
        decoder = self.decoder(f)
        return decoder,x,att

class AutoEncoderFCMemMoreFeature(nn.Module):
    def __init__(self,mem_dim=1024,shrink_thres=0.0025):
        super().__init__()
        # encoder
        self.conv1 = nn.Conv2d(1,32,(3,3),stride=2,padding=1)
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

        self.mem_dim = mem_dim
        self.mem_rep = MemModule(
            mem_dim=mem_dim,
            fea_dim=512*2*8,
            shrink_thres=shrink_thres)

        # decoder 
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv1 = nn.Conv2d(512,256,3,stride=1,padding=1)
        self.debn1 = nn.BatchNorm2d(256)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv2 = nn.Conv2d(256,128,3,stride=1,padding=1)
        self.debn2 = nn.BatchNorm2d(128)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv3 = nn.Conv2d(128,64,3,stride=1,padding=1)
        self.debn3 = nn.BatchNorm2d(64)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv4 = nn.Conv2d(64,32,3,stride=1,padding=1)
        self.debn4 = nn.BatchNorm2d(32)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv5 = nn.Conv2d(32,1,3,stride=1,padding=1)
        self.final = nn.Tanh()
    def encoder(self,x):
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        x = self.conv3(x)
        x = self.relu(self.bn3(x))
        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        x = self.conv5(x)
        x = self.relu(self.bn5(x))
        return x
    def decoder(self,x):
        x = self.up1(x)
        x = self.relu(self.debn1(self.deconv1(x)))
        x = self.up2(x)
        x = self.relu(self.debn2(self.deconv2(x)))
        x = self.up3(x)
        x = self.relu(self.debn3(self.deconv3(x)))
        x = self.up4(x)
        x = self.relu(self.debn4(self.deconv4(x)))
        x = self.up5(x)
        x = self.deconv5(x)
        x = self.final(x)
        return x
    def forward(self,x):
        
        encoder = self.encoder(x) # batch_size,512,2,8
        b,c,h,w = encoder.shape
        # hope : batch_size,(512,2,8)
        encoder = encoder.view(b,-1)
        res_mem = self.mem_rep(encoder)
        f = res_mem['output']
        # batch_size,(512,2,8)
        f = f.view(b,c,h,w)
        att = res_mem['att']
        # att = att.view(b,c,h,w)
        decoder = self.decoder(f)
        return decoder,x,att

class MemoryAutoEncoder(nn.Module):
    def __init__(self,mem_dim=64,shrink_thres=0.0025,fea_dim=512*2*2):
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

        self.mem_dim = mem_dim
        self.mem_rep = MemModule(
            mem_dim=mem_dim,
            fea_dim=fea_dim,
            shrink_thres=shrink_thres)

        # decoder 
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv1 = nn.Conv2d(512,256,3,stride=1,padding=1)
        self.debn1 = nn.BatchNorm2d(256)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv2 = nn.Conv2d(256,128,3,stride=1,padding=1)
        self.debn2 = nn.BatchNorm2d(128)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv3 = nn.Conv2d(128,64,3,stride=1,padding=1)
        self.debn3 = nn.BatchNorm2d(64)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv4 = nn.Conv2d(64,32,3,stride=1,padding=1)
        self.debn4 = nn.BatchNorm2d(32)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv5 = nn.Conv2d(32,3,3,stride=1,padding=1)
        self.final = nn.Tanh()
        
        self.waveletDecompose = wavelet.WavePool_Two(in_channels=3).cuda()
        
    def encoder(self,x):
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        x = self.conv3(x)
        x = self.relu(self.bn3(x))
        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        x = self.conv5(x)
        x = self.relu(self.bn5(x))
        return x
    def decoder(self,x):
        x = self.up1(x)
        x = self.relu(self.debn1(self.deconv1(x)))
        x = self.up2(x)
        x = self.relu(self.debn2(self.deconv2(x)))
        x = self.up3(x)
        x = self.relu(self.debn3(self.deconv3(x)))
        x = self.up4(x)
        x = self.relu(self.debn4(self.deconv4(x)))
        x = self.up5(x)
        x = self.deconv5(x)
        x = self.final(x)
        return x
    def forward(self,x):
        encoder = self.encoder(x) # batch_size,512,2,8
        b,c,h,w = encoder.shape
        encoder_com = encoder.view(b, -1)
        res_mem = self.mem_rep(encoder_com)
        f = res_mem['output']
        f = f.view(b,c,h,w)
        att = res_mem['att']
        decoder = self.decoder(f)
        return encoder, f, decoder, att

class AutoEncoderFCMemMoreFeatureArgMaxDoubleEncoder(nn.Module):
    def __init__(self,mem_dim=64,shrink_thres=0.0025):
        super().__init__()
        # encoder
        self.conv1 = nn.Conv2d(1,32,(3,3),stride=2,padding=1)
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

        self.mem_dim = mem_dim
        self.mem_rep = MemoryUnitWithArgMax(
            mem_dim=mem_dim,
            fea_dim=512*2*2,
            shrink_thres=shrink_thres)

        # decoder 
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv1 = nn.Conv2d(512,256,3,stride=1,padding=1)
        self.debn1 = nn.BatchNorm2d(256)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv2 = nn.Conv2d(256,128,3,stride=1,padding=1)
        self.debn2 = nn.BatchNorm2d(128)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv3 = nn.Conv2d(128,64,3,stride=1,padding=1)
        self.debn3 = nn.BatchNorm2d(64)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv4 = nn.Conv2d(64,32,3,stride=1,padding=1)
        self.debn4 = nn.BatchNorm2d(32)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv5 = nn.Conv2d(32,1,3,stride=1,padding=1)
        self.final = nn.Tanh()
    def encoder(self,x):
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        x = self.conv3(x)
        x = self.relu(self.bn3(x))
        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        x = self.conv5(x)
        x = self.relu(self.bn5(x))
        return x
    def decoder(self,x):
        x = self.up1(x)
        x = self.relu(self.debn1(self.deconv1(x)))
        x = self.up2(x)
        x = self.relu(self.debn2(self.deconv2(x)))
        x = self.up3(x)
        x = self.relu(self.debn3(self.deconv3(x)))
        x = self.up4(x)
        x = self.relu(self.debn4(self.deconv4(x)))
        x = self.up5(x)
        x = self.deconv5(x)
        x = self.final(x)
        return x
    def forward(self,x):
        
        encoder = self.encoder(x) # batch_size,512,2,8
        b,c,h,w = encoder.shape
        # hope : batch_size,(512,2,8)
        encoder = encoder.view(b,-1)
        res_mem = self.mem_rep(encoder)
        f = res_mem['output']
        # batch_size,(512,2,8)
        f = f.view(b,c,h,w)
        att = res_mem['att']
        # att = att.view(b,c,h,w)
        decoder = self.decoder(f)
        return decoder,x,att

class AutoEncoderFCMemMoreFeatureArgMaxWithShadow22(nn.Module):
    def __init__(self,mem_dim=64,shrink_thres=0.0025,fea_dim=512*2*8):
        super().__init__()
        # encoder
        self.conv1 = nn.Conv2d(1,32,(3,3),stride=2,padding=1)
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

        self.mem_dim = mem_dim
        self.mem_rep = MemoryUnitWithArgMax(
            mem_dim=mem_dim,
            fea_dim=fea_dim,
            shrink_thres=shrink_thres)
        self.shallow_mem = MemModuleArgmax(
            mem_dim=1024,
            fea_dim=32,
            shrink_thres=shrink_thres)
        # decoder 
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv1 = nn.Conv2d(512,256,3,stride=1,padding=1)
        self.debn1 = nn.BatchNorm2d(256)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv2 = nn.Conv2d(256,128,3,stride=1,padding=1)
        self.debn2 = nn.BatchNorm2d(128)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv3 = nn.Conv2d(128,64,3,stride=1,padding=1)
        self.debn3 = nn.BatchNorm2d(64)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv4 = nn.Conv2d(64,32,3,stride=1,padding=1)
        self.debn4 = nn.BatchNorm2d(32)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv5 = nn.Conv2d(32,1,3,stride=1,padding=1)
        # self.deconv6 = nn.Conv2d(16,1,3,)
        self.final = nn.Tanh()
    def encoder(self,x):
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        self.shallow = x
        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        x = self.conv3(x)
        x = self.relu(self.bn3(x))
        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        x = self.conv5(x)
        x = self.relu(self.bn5(x))
        return x
    
    def retrieval(self,x,memory):
        # b,c,h,w = x.shape
        e = x
        res_mem = memory(e)
        f = res_mem['output']
        # f = f.view(b,c,h,w)
        att = res_mem['att']
        return f,att

    def decoder(self,x):
        x = self.up1(x)
        x = self.relu(self.debn1(self.deconv1(x)))
        x = self.up2(x)
        x = self.relu(self.debn2(self.deconv2(x)))
        x = self.up3(x)
        x = self.relu(self.debn3(self.deconv3(x)))
        x = self.up4(x)
        x = self.relu(self.debn4(self.deconv4(x)))
        f,att = self.retrieval(self.shallow,self.shallow_mem)
        # x = 0.9*f + 0.1*x
        # x = torch.cat((x,f),dim=1)
        # x = (f + x) / 2
        x = self.up5(x)
        x = self.deconv5(x)
        x = self.final(x)
        return x
    def forward(self,x):
        encoder = self.encoder(x) # batch_size,512,2,8
        b,c,h,w = encoder.shape
        # hope : batch_size,(512,2,8)
        encoder = encoder.view(b,-1)
        res_mem = self.mem_rep(encoder)
        f = res_mem['output']
        # batch_size,(512,2,8)
        f = f.view(b,c,h,w)
        att = res_mem['att']
        # att = att.view(b,c,h,w)
        decoder = self.decoder(f)
        return decoder,x,att

class AutoEncoderFCMemMoreFeatureArgMaxWithShadow(nn.Module):
    def __init__(self,mem_dim=64,shrink_thres=0.0025,fea_dim=512*2*8):
        super().__init__()
        # encoder
        self.conv1 = nn.Conv2d(1,32,(3,3),stride=2,padding=1)
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

        self.mem_dim = mem_dim
        self.mem_rep = MemoryUnitWithArgMax(
            mem_dim=mem_dim,
            fea_dim=fea_dim,
            shrink_thres=shrink_thres)
        self.shallow_mem = MemoryUnitWithArgMax(
            mem_dim=mem_dim*4,
            fea_dim=fea_dim*16,
            shrink_thres=shrink_thres)
        # decoder 
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv1 = nn.Conv2d(512,256,3,stride=1,padding=1)
        self.debn1 = nn.BatchNorm2d(256)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv2 = nn.Conv2d(256,128,3,stride=1,padding=1)
        self.debn2 = nn.BatchNorm2d(128)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv3 = nn.Conv2d(128,64,3,stride=1,padding=1)
        self.debn3 = nn.BatchNorm2d(64)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv4 = nn.Conv2d(64,32,3,stride=1,padding=1)
        self.debn4 = nn.BatchNorm2d(32)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv5 = nn.Conv2d(32,1,3,stride=1,padding=1)
        self.deconv6 = nn.Conv2d(16,1,3,)
        self.final = nn.Tanh()
    def encoder(self,x):
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        self.shallow = x
        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        x = self.conv3(x)
        x = self.relu(self.bn3(x))
        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        x = self.conv5(x)
        x = self.relu(self.bn5(x))
        return x
    
    def retrieval(self,x,memory):
        b,c,h,w = x.shape
        e = x.view(b,-1)
        res_mem = memory(e)
        f = res_mem['output']
        f = f.view(b,c,h,w)
        att = res_mem['att']
        return f,att

    def decoder(self,x):
        x = self.up1(x)
        x = self.relu(self.debn1(self.deconv1(x)))
        x = self.up2(x)
        x = self.relu(self.debn2(self.deconv2(x)))
        x = self.up3(x)
        x = self.relu(self.debn3(self.deconv3(x)))
        x = self.up4(x)
        x = self.relu(self.debn4(self.deconv4(x)))
        # f,att = self.retrieval(self.shallow,self.shallow_mem)
        # x = 0.9*f + 0.1*x
        # x = torch.cat((x,f),dim=1)
        x = self.up5(x)
        x = self.deconv5(x)
        x = self.final(x)
        return x
    def forward(self,x):
        encoder = self.encoder(x) # batch_size,512,2,8
        b,c,h,w = encoder.shape
        # hope : batch_size,(512,2,8)
        encoder = encoder.view(b,-1)
        res_mem = self.mem_rep(encoder)
        f = res_mem['output']
        # batch_size,(512,2,8)
        f = f.view(b,c,h,w)
        att = res_mem['att']
        # att = att.view(b,c,h,w)
        decoder = self.decoder(f)
        return decoder,x,att

class AutoEncoderFCMemMoreFeatureArgMaxWithShadowWithoutDeep(nn.Module):
    def __init__(self,mem_dim=64,shrink_thres=0.0025,fea_dim=512*2*8):
        super().__init__()
        # encoder
        self.conv1 = nn.Conv2d(1,32,(3,3),stride=2,padding=1)
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

        self.mem_dim = mem_dim
        self.mem_rep = MemoryUnitWithArgMax(
            mem_dim=mem_dim,
            fea_dim=fea_dim,
            shrink_thres=shrink_thres)
        self.shallow_mem = MemoryUnitWithArgMax(
            mem_dim=mem_dim*4,
            fea_dim=fea_dim*16,
            shrink_thres=shrink_thres)
        # decoder 
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv1 = nn.Conv2d(512,256,3,stride=1,padding=1)
        self.debn1 = nn.BatchNorm2d(256)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv2 = nn.Conv2d(256,128,3,stride=1,padding=1)
        self.debn2 = nn.BatchNorm2d(128)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv3 = nn.Conv2d(128,64,3,stride=1,padding=1)
        self.debn3 = nn.BatchNorm2d(64)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv4 = nn.Conv2d(64,32,3,stride=1,padding=1)
        self.debn4 = nn.BatchNorm2d(32)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv5 = nn.Conv2d(64,1,3,stride=1,padding=1)
        self.deconv6 = nn.Conv2d(16,1,3,)
        self.final = nn.Tanh()
    def encoder(self,x):
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        self.shallow = x
        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        x = self.conv3(x)
        x = self.relu(self.bn3(x))
        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        x = self.conv5(x)
        x = self.relu(self.bn5(x))
        return x
    
    def retrieval(self,x,memory):
        b,c,h,w = x.shape
        e = x.view(b,-1)
        res_mem = memory(e)
        f = res_mem['output']
        f = f.view(b,c,h,w)
        att = res_mem['att']
        return f,att

    def decoder(self,x):
        x = self.up1(x)
        x = self.relu(self.debn1(self.deconv1(x)))
        x = self.up2(x)
        x = self.relu(self.debn2(self.deconv2(x)))
        x = self.up3(x)
        x = self.relu(self.debn3(self.deconv3(x)))
        x = self.up4(x)
        x = self.relu(self.debn4(self.deconv4(x)))
        f,att = self.retrieval(self.shallow,self.shallow_mem)
        # x = 0.9*f + 0.1*x
        x = torch.cat((x,f),dim=1)
        x = self.up5(x)
        x = self.deconv5(x)
        x = self.final(x)
        return x
    def forward(self,x):
        encoder = self.encoder(x) # batch_size,512,2,8
        decoder = self.decoder(encoder)
        att= torch.tensor([0.1,0.2,0.7])
        b,c,h,w = encoder.shape
        # # hope : batch_size,(512,2,8)
        encoder = encoder.view(b,-1)
        res_mem = self.mem_rep(encoder)
        f = res_mem['output']
        # # batch_size,(512,2,8)
        f = f.view(b,c,h,w)
        att = res_mem['att']
        # att = att.view(b,c,h,w)
        # decoder = self.decoder(f)
        return decoder,x,att

class AutoEncoderFCMemResidual(nn.Module):
    def __init__(self,mem_dim=1024,shrink_thres=0.0025,residual_rate=0.5):
        super(AutoEncoderFCMemResidual,self).__init__()
        # encoder
        self.residual_rate = residual_rate
        self.conv1 = nn.Conv2d(1,32,(3,3),stride=2,padding=1)
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

        self.mem_dim = mem_dim
        self.mem_rep = MemModule(
            mem_dim=mem_dim,
            fea_dim=512,
            shrink_thres=shrink_thres)

        # decoder 
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv1 = nn.Conv2d(512,256,3,stride=1,padding=1)
        self.debn1 = nn.BatchNorm2d(256)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv2 = nn.Conv2d(256,128,3,stride=1,padding=1)
        self.debn2 = nn.BatchNorm2d(128)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv3 = nn.Conv2d(128,64,3,stride=1,padding=1)
        self.debn3 = nn.BatchNorm2d(64)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv4 = nn.Conv2d(64,32,3,stride=1,padding=1)
        self.debn4 = nn.BatchNorm2d(32)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv5 = nn.Conv2d(32,1,3,stride=1,padding=1)
        self.final = nn.Tanh()
    def encoder(self,x):
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        x = self.conv3(x)
        x = self.relu(self.bn3(x))
        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        x = self.conv5(x)
        x = self.bn5(x)
        return x
    def decoder(self,x):
        x = self.up1(x)
        x = self.relu(self.debn1(self.deconv1(x)))
        x = self.up2(x)
        x = self.relu(self.debn2(self.deconv2(x)))
        x = self.up3(x)
        x = self.relu(self.debn3(self.deconv3(x)))
        x = self.up4(x)
        x = self.relu(self.debn4(self.deconv4(x)))
        x = self.up5(x)
        x = self.deconv5(x)
        x = self.final(x)
        return x
    def forward(self,x):
        encoder = self.encoder(x)

        res_mem = self.mem_rep(encoder)
        f = res_mem['output'] * (self.residual_rate) + (1 - self.residual_rate) * encoder
        att = res_mem['att']
        decoder = self.decoder(f)
        return decoder,x,att

def cifar10AELoss(x_h,x,att,reg_param=0):
    b = x.shape[0]
    recon_loss = (x-x_h).abs().div(0.1).exp().mean() # 链式调用
    return {"loss":recon_loss,"recon_loss":0,"regularizer":0}

def MemAELoss(x_h,x,att,reg_param=0.2):
    attention_weights = att
    # recon_loss = F.mse_loss(x,x_h)
    b = x.shape[0]
    recon_loss = (x-x_h).abs().view(b,-1).sum(dim=1).mean() # 链式调用
    regularizer = F.softmax(attention_weights,dim=1) * F.log_softmax(attention_weights,dim=1)
    regularizer = (-1) * regularizer.sum()
    loss = recon_loss + reg_param*regularizer
    return {"loss":loss,"recon_loss":recon_loss,"regularizer":regularizer}

def MyMemAELoss(x_h,x,att,reg_param=0):
    attention_weights = att
    # recon_loss = F.mse_loss(x,x_h)
    b = x.shape[0]
    # recon_loss = (x-x_h).abs().div(0.1).exp().mean() # 链式调用
    recon_loss = (x-x_h).abs().mean()

    regularizer = F.softmax(attention_weights,dim=1) * F.log_softmax(attention_weights,dim=1)
    regularizer = (-1) * regularizer.sum()
    loss = recon_loss + reg_param*regularizer
    return {"loss":loss,"recon_loss":recon_loss,"regularizer":regularizer}

if __name__ == '__main__':
    batch_size = 1024
    version = '1'
    in_col_dim = 2 # number of feature columns in the data
    mem_dim = 1024
    shrink_thres = 0.00025
    learning_rate = 0.9
    regularization_parameter = 0.0001
    epochs = 5

    test_data = torch.rand(8,1,64,256)
    # ## Init Model, loss and optimizer 
    model = AutoEncoderFCMem(mem_dim=mem_dim, shrink_thres=shrink_thres)

    out = model(test_data)
    print(len(out),out[0].shape,out[1].shape,out[2].shape)

class WaveletBlock(nn.Module):
    def __init__(self, in_channel=16, out_channel = 8, upscale = False):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upscale = upscale
        if(upscale):
            self.wavePool = wavelet.WavePool_Two_Upscale(in_channel)
        else:
            self.wavePool = wavelet.WavePool_Two(in_channel)
        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, 3, stride=1, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.bn = nn.BatchNorm2d(self.out_channel)
    def forward(self, x):
        LL, LH, HL, HH = self.wavePool(x)
        fre_H = LH + HL + HH
        fre_H = self.conv1(fre_H)
        fre_H = self.bn(fre_H)
        return fre_H, LL

#AutoEncoderFCMemMoreFeatureArgMax using wavelet
class AutoEncoderFCMemMoreFeatureArgMaxUsingWavelet(nn.Module):
    def __init__(self,mem_dim=64,shrink_thres=0.0025,fea_dim=512*2*2):
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

        self.mem_dim = mem_dim
        self.mem_rep = MemoryUnitWithArgMax(
            mem_dim=mem_dim,
            fea_dim=fea_dim,
            shrink_thres=shrink_thres)

        # decoder 
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv1 = nn.Conv2d(512,256,3,stride=1,padding=1)
        self.debn1 = nn.BatchNorm2d(256)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv2 = nn.Conv2d(256,128,3,stride=1,padding=1)
        self.debn2 = nn.BatchNorm2d(128)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv3 = nn.Conv2d(128,64,3,stride=1,padding=1)
        self.debn3 = nn.BatchNorm2d(64)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv4 = nn.Conv2d(64,32,3,stride=1,padding=1)
        self.debn4 = nn.BatchNorm2d(32)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv5 = nn.Conv2d(32,3,3,stride=1,padding=1)
        self.final = nn.Tanh()

        #wavelet
        self.waveletResBlock1 = WaveletBlock(3, 32)
        self.waveletResBlock2 = WaveletBlock(32, 64)
        self.waveletResBlock3 = WaveletBlock(64, 128)
        self.waveletResBlock4 = WaveletBlock(128, 256)
        self.waveletResBlock5 = WaveletBlock(256, 512)
        
        self.waveletResBlockDown1 = WaveletBlock(512, 256, upscale=True)
        self.waveletResBlockDown2 = WaveletBlock(256, 128, upscale=True)
        self.waveletResBlockDown3 = WaveletBlock(128, 64, upscale=True)
        self.waveletResBlockDown4 = WaveletBlock(64, 32, upscale=True)
        self.waveletResBlockDown5 = WaveletBlock(32, 3, upscale=True)
    def encoder(self,x):
        fre_feat, LL = self.waveletResBlock1(x)
        #x = LL
        x = self.conv1(x) #->[32, 32, 256, 256]
        x = self.relu(self.bn1(x) + fre_feat)
        fre_feat, LL = self.waveletResBlock2(x) #->[32, 64, 128, 128]
        x = self.conv2(x) #->[32, 64, 128, 128]
        x = self.relu(self.bn2(x) + fre_feat)
        fre_feat, LL = self.waveletResBlock3(x)
        x = self.conv3(x) #->[32, 128, 64, 64]
        x = self.relu(self.bn3(x) + fre_feat)
        fre_feat, LL = self.waveletResBlock4(x) #[32, 256, 32, 32]
        x = self.conv4(x)
        x = self.relu(self.bn4(x) + fre_feat)
        fre_feat, LL = self.waveletResBlock5(x)
        x = self.conv5(x)
        x = self.relu(self.bn5(x) + fre_feat)
        return x
    def decoder(self,x):
        x = self.up1(x) #[-1, 512, 32, 32]
        # [-1, 512, 32, 32] -> [-1, 256, 64, 64]
        fre_feat, LL = self.waveletResBlockDown1(x)
        x = self.relu(self.debn1(self.deconv1(x)) + fre_feat)
        x = self.up2(x)

        # [-1, 256, 64, 64] -> [-1, 128, 128, 128]
        fre_feat, LL = self.waveletResBlockDown2(x)
        x = self.relu(self.debn2(self.deconv2(x)) + fre_feat) 
        x = self.up3(x)
        fre_feat, LL = self.waveletResBlockDown3(x)

        # [-1, 128, 128, 128] -> [-1, 64, 256, 256] 
        x = self.relu(self.debn3(self.deconv3(x)) + fre_feat)
        x = self.up4(x)
        fre_feat, LL = self.waveletResBlockDown4(x)
        x = self.relu(self.debn4(self.deconv4(x)) + fre_feat)
        x = self.up5(x)
        fre_feat, LL = self.waveletResBlockDown5(x)
        x = self.deconv5(x)
        #x = self.final(x + fre_feat) 或者这个
        x = self.final(x + fre_feat)
        return x
    
    def forward(self,x):
        encoder = self.encoder(x) # batch_size,512,2,8
        b,c,h,w = encoder.shape
        # hope : batch_size,(512,2,8)
        encoder = encoder.view(b,-1)
        res_mem = self.mem_rep(encoder)
        f = res_mem['output']
        # batch_size,(512,2,8)
        f = f.view(b,c,h,w)
        att = res_mem['att']
        # att = att.view(b,c,h,w)
        decoder = self.decoder(f)

        return decoder,encoder.view(b,c,w,h),f
    
    #decoder过程中则么使用频率信息？现在的肯定存在问题。
    #Line1005，LL和Fre_H两路提取信息？

#AutoEncoderFCMemMore using wavelet two path
class AutoEncoderFCMemMoreUsingWaveletTwoPath(nn.Module):
    def __init__(self,mem_dim=64,shrink_thres=0.0025,fea_dim=512*2*2):
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

        self.mem_dim = mem_dim
        self.mem_rep = MemoryUnitWithArgMax(
            mem_dim=mem_dim,
            fea_dim=fea_dim,
            shrink_thres=shrink_thres)

        # decoder 
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv1 = nn.Conv2d(512,256,3,stride=1,padding=1)
        self.debn1 = nn.BatchNorm2d(256)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv2 = nn.Conv2d(256,128,3,stride=1,padding=1)
        self.debn2 = nn.BatchNorm2d(128)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv3 = nn.Conv2d(128,64,3,stride=1,padding=1)
        self.debn3 = nn.BatchNorm2d(64)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv4 = nn.Conv2d(64,32,3,stride=1,padding=1)
        self.debn4 = nn.BatchNorm2d(32)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv5 = nn.Conv2d(32,3,3,stride=1,padding=1)
        self.final = nn.Tanh()
        #wavelet
        self.wavePool = wavelet.WavePool_Two(3)

    def encoder(self,x):
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        x = self.conv3(x)
        x = self.relu(self.bn3(x))
        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        x = self.conv5(x)
        x = self.relu(self.bn5(x))
        return x
    def decoder(self,x):
        x = self.up1(x)
        x = self.relu(self.debn1(self.deconv1(x)))
        x = self.up2(x)
        x = self.relu(self.debn2(self.deconv2(x)))
        x = self.up3(x)
        x = self.relu(self.debn3(self.deconv3(x)))
        x = self.up4(x)
        x = self.relu(self.debn4(self.deconv4(x)))
        x = self.up5(x)
        x = self.deconv5(x)
        x = self.final(x)
        return x
    
    def forward(self,x):
        LL, LH, HL, HH = self.wavePool(x)
        fre_H = HL + LH + HH
        encoder_H = self.encoder(fre_H) # batch_size,512,2,8
        encoder_L = self.encoder(LL) # batch_size,512,2,8
        b,c,h,w = encoder_H.shape
        # hope : batch_size,(512,2,8)
        encoder_H = encoder_H.view(b,-1)
        encoder_L = encoder_L.view(b,-1)
        res_mem_H = self.mem_rep(encoder_H)
        res_mem_L = self.mem_rep(encoder_L)
        f_H = res_mem_H['output']
        f_L = res_mem_H['output']
        # batch_size,(512,2,8)
        f_H = f_H.view(b,c,h,w)
        f_L = f_L.view(b,c,h,w)
        att_H = res_mem_H['att']
        att_L = res_mem_L['att']
        # att = att.view(b,c,h,w)
        decoder_H = self.decoder(f_H)
        decoder_L = self.decoder(f_L)

        return decoder_H,decoder_L,encoder_H.view(b,c,w,h),encoder_L.view(b,c,w,h),f_H,f_L,fre_H,LL
    
    #decoder过程中则么使用频率信息？现在的肯定存在问题。
    #Line1005，LL和Fre_H两路提取信息？

#AutoEncoderFCMemMoreFeatureArgMax using wavelet and two memory
class AutoEncoderFCMemMoreFeatureArgMaxUsingWaveletTwoMem(nn.Module):
    def __init__(self,mem_dim=64,shrink_thres=0.0025,fea_dim=512*2*2):
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

        self.mem_dim = mem_dim
        self.mem_rep = MemoryUnitWithArgMax(
            mem_dim=mem_dim,
            fea_dim=fea_dim,
            shrink_thres=shrink_thres)

        self.mem_rep_freq = MemoryUnitWithArgMax(
            mem_dim=mem_dim,
            fea_dim=fea_dim,
            shrink_thres=shrink_thres
        )

        # decoder 
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv1 = nn.Conv2d(512,256,3,stride=1,padding=1)
        self.debn1 = nn.BatchNorm2d(256)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv2 = nn.Conv2d(256,128,3,stride=1,padding=1)
        self.debn2 = nn.BatchNorm2d(128)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv3 = nn.Conv2d(128,64,3,stride=1,padding=1)
        self.debn3 = nn.BatchNorm2d(64)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv4 = nn.Conv2d(64,32,3,stride=1,padding=1)
        self.debn4 = nn.BatchNorm2d(32)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv5 = nn.Conv2d(32,3,3,stride=1,padding=1)
        self.final = nn.Tanh()

        #wavelet
        self.waveletResBlock1 = WaveletBlock(3, 32)
        self.waveletResBlock2 = WaveletBlock(32, 64)
        self.waveletResBlock3 = WaveletBlock(64, 128)
        self.waveletResBlock4 = WaveletBlock(128, 256)
        self.waveletResBlock5 = WaveletBlock(256, 512)
        
        self.waveletResBlockDown1 = WaveletBlock(512, 256, upscale=True)
        self.waveletResBlockDown2 = WaveletBlock(256, 128, upscale=True)
        self.waveletResBlockDown3 = WaveletBlock(128, 64, upscale=True)
        self.waveletResBlockDown4 = WaveletBlock(64, 32, upscale=True)
        self.waveletResBlockDown5 = WaveletBlock(32, 3, upscale=True)
    def encoder(self,x):
        fre_feat, LL = self.waveletResBlock1(x)
        #x = LL
        x = self.conv1(x) #->[32, 32, 256, 256]
        x = self.relu(self.bn1(x) + fre_feat)
        fre_feat, LL = self.waveletResBlock2(x) #->[32, 64, 128, 128]
        x = self.conv2(x) #->[32, 64, 128, 128]
        x = self.relu(self.bn2(x) + fre_feat)
        fre_feat, LL = self.waveletResBlock3(x)
        x = self.conv3(x) #->[32, 128, 64, 64]
        x = self.relu(self.bn3(x) + fre_feat)
        fre_feat, LL = self.waveletResBlock4(x) #[32, 256, 32, 32]
        x = self.conv4(x)
        x = self.relu(self.bn4(x) + fre_feat)
        fre_feat, LL = self.waveletResBlock5(x)
        x = self.conv5(x)
        x = self.relu(self.bn5(x) + fre_feat)
        return x, fre_feat
    def decoder(self, x, fre_f):
        x = self.up1(x) #[-1, 512, 32, 32]
        # [-1, 512, 32, 32] -> [-1, 256, 64, 64]
        fre_feat, LL = self.waveletResBlockDown1(self.up1(fre_f))
        x = self.relu(self.debn1(self.deconv1(x)) + fre_feat)
        x = self.up2(x)

        # [-1, 256, 64, 64] -> [-1, 128, 128, 128]
        fre_feat, LL = self.waveletResBlockDown2(x)
        x = self.relu(self.debn2(self.deconv2(x)) + fre_feat) 
        x = self.up3(x)
        fre_feat, LL = self.waveletResBlockDown3(x)

        # [-1, 128, 128, 128] -> [-1, 64, 256, 256] 
        x = self.relu(self.debn3(self.deconv3(x)) + fre_feat)
        x = self.up4(x)
        fre_feat, LL = self.waveletResBlockDown4(x)
        x = self.relu(self.debn4(self.deconv4(x)) + fre_feat)
        x = self.up5(x)
        fre_feat, LL = self.waveletResBlockDown5(x)
        x = self.deconv5(x)
        #x = self.final(x + fre_feat) 或者这个
        x = self.final(x + fre_feat)
        return x
    
    def forward(self,x):
        encoder, fre_feat = self.encoder(x) # batch_size,512,2,8
        b,c,h,w = encoder.shape
        # hope : batch_size,(512,2,8)
        encoder = encoder.view(b,-1)
        b_f,c_f,h_f,w_f = fre_feat.shape
        fre_feat = fre_feat.view(b_f, -1)
        res_mem = self.mem_rep(encoder)
        #freq memory
        res_mem_freq = self.mem_rep_freq(fre_feat)
        f = res_mem['output']
        # batch_size,(512,2,8)
        f = f.view(b,c,h,w)
        fre_f = res_mem_freq['output']
        fre_f = fre_f.view(b_f,c_f,h_f,w_f)
        att = res_mem['att']
        # att = att.view(b,c,h,w)
        decoder = self.decoder(f, fre_f)

        return decoder, encoder.view(b,c,w,h), fre_feat.view(b_f,c_f,h_f,w_f), f, fre_f
    
    #AutoEncoderFCMemMoreFeatureArgMax using wavelet and two encoder

###best module####
class AutoEncoderFCMemMoreFeatureTwoEncoderWithWavelet(nn.Module):
    def __init__(self,mem_dim=64,shrink_thres=0.0025,fea_dim=512*2*2):
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
        
        #memory
        self.mem_dim = mem_dim
        self.mem_rep = MemoryUnitWithArgMax(
            mem_dim=mem_dim,
            fea_dim=fea_dim,
            shrink_thres=shrink_thres)
        # self.mem_rep = MyMemoryUnitWithArgMax(
        #     mem_dim=mem_dim,
        #     fea_dim=fea_dim)

        # decoder 
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv1 = nn.Conv2d(512,256,3,stride=1,padding=1)
        self.debn1 = nn.BatchNorm2d(256)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv2 = nn.Conv2d(256,128,3,stride=1,padding=1)
        self.debn2 = nn.BatchNorm2d(128)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv3 = nn.Conv2d(128,64,3,stride=1,padding=1)
        self.debn3 = nn.BatchNorm2d(64)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv4 = nn.Conv2d(64,32,3,stride=1,padding=1)
        self.debn4 = nn.BatchNorm2d(32)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv5 = nn.Conv2d(32,3,3,stride=1,padding=1)
        self.final = nn.Tanh()

        #wavelet
        self.waveletDecompose = wavelet.WavePool_Two(in_channels=3).cuda()
        self.waveletCompose = wavelet.WaveUnpool(512, 'sum')
        self.waveletResBlock1 = WaveletBlock(3, 32)
        self.waveletResBlock2 = WaveletBlock(32, 64)
        self.waveletResBlock3 = WaveletBlock(64, 128)
        self.waveletResBlock4 = WaveletBlock(128, 256)
        self.waveletResBlock5 = WaveletBlock(256, 512)
        
        self.waveletResBlockDown1 = WaveletBlock(512, 256, upscale=True)
        self.waveletResBlockDown2 = WaveletBlock(256, 128, upscale=True)
        self.waveletResBlockDown3 = WaveletBlock(128, 64, upscale=True)
        self.waveletResBlockDown4 = WaveletBlock(64, 32, upscale=True)
        self.waveletResBlockDown5 = WaveletBlock(32, 3, upscale=True)
        
        #conv
        self.conv_connection_1 = nn.Conv2d(2, 1, 1,stride=1, padding=0)
        self.sig = nn.Sigmoid()
    def freq_connection(self, freq_h, freq_l):
        x = torch.cat((freq_h,freq_l), dim=1)
        #h, w, c = x.shape
        x_max = torch.max(x, dim = 1).values.unsqueeze(1)
        x_sum = torch.sum(x, dim = 1).unsqueeze(1)
        x = torch.cat((x_max, x_sum), dim=1)
        x = self.conv_connection_1(x)
        x = self.sig(x)
        return x * freq_h
    
    def encoder_H(self, freq_h, feat_1, feat_2, feat_3, feat_4):
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
        return x, feat_1, feat_2, feat_3, feat_4
    
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
        x = self.deconv5(x)
        x = self.final(x)
        return x
    
    def forward(self,x):
        LL, LH, HL, HH = self.waveletDecompose(x)
        freq_h = torch.cat((LH,HL,HH),dim=1)
        freq_l = LL
        x_LL, feat_1, feat_2, feat_3, feat_4 = self.encoder_L(freq_l)
        x_H = self.encoder_H(freq_h, feat_1, feat_2, feat_3, feat_4)
        #torch.split(x_H, split_size_or_sections=3, dim=1)
        x_LH = x_H[:, 0:512, :, :]
        x_HL = x_H[:, 512:1024, :, :]
        x_HH = x_H[:, 1024:1536, :, :]
        encoder_com = self.waveletCompose(x_LL, x_LH, x_HL, x_HH)
        b,c,h,w = encoder_com.shape
        encoder_com = encoder_com.view(b, -1)
        res_mem = self.mem_rep(encoder_com)
        f = res_mem['output']
        f = f.view(b,c,h,w)
        decoder = self.decoder(f)
        return decoder, encoder_com.view(b,c,w,h), f, res_mem['att']
    
        #AutoEncoderFCMemMoreFeatureArgMax using wavelet and two encoder and different memory

class AutoEncoderTwoEncoderAndOneDecoderWithWaveletDiffMem(nn.Module):
    def __init__(self,mem_dim=64,shrink_thres=0.0025,fea_dim=512*2*2):
        super().__init__()
        #memory
        self.mem_dim = mem_dim
        self.mem_rep1 = MemoryUnitWithArgMax(
            mem_dim=mem_dim,
            fea_dim=fea_dim * 3,
            shrink_thres=shrink_thres
        )
        
        self.mem_rep2 = MemoryUnitWithArgMax(
            mem_dim=mem_dim,
            fea_dim=fea_dim,
            shrink_thres=shrink_thres
        )

        #wavelet
        self.waveletDecompose = wavelet.WavePool_Two(in_channels=3).cuda()
        self.waveletCompose = wavelet.WaveUnpool(512, 'sum')
        self.waveletResBlock1 = WaveletBlock(3, 32)
        self.waveletResBlock2 = WaveletBlock(32, 64)
        self.waveletResBlock3 = WaveletBlock(64, 128)
        self.waveletResBlock4 = WaveletBlock(128, 256)
        self.waveletResBlock5 = WaveletBlock(256, 512)
        
        self.waveletResBlockDown1 = WaveletBlock(512, 256, upscale=True)
        self.waveletResBlockDown2 = WaveletBlock(256, 128, upscale=True)
        self.waveletResBlockDown3 = WaveletBlock(128, 64, upscale=True)
        self.waveletResBlockDown4 = WaveletBlock(64, 32, upscale=True)
        self.waveletResBlockDown5 = WaveletBlock(32, 3, upscale=True)
        
        #conv
        self.conv_connection_1 = nn.Conv2d(2, 1, 1,stride=1, padding=0)
        self.sig = nn.Sigmoid()
        self.encoder_H = self.Encoder_H()
        self.encoder_L = self.Encoder_L()
        self.decoder = self.Decoder()
        
    def freq_connection(self, freq_h, freq_l):
        x = torch.cat((freq_h,freq_l), dim=1)
        #h, w, c = x.shape
        x_max = torch.max(x, dim = 1).values.unsqueeze(1)
        x_sum = torch.sum(x, dim = 1).unsqueeze(1)
        x = torch.cat((x_max, x_sum), dim=1)
        x = self.conv_connection_1(x)
        x = self.sig(x)
        return x * freq_h
    
    class Encoder_H(nn.Module):
        def __init__(self):
            super().__init__()
            #encoder_H
            self.relu  = nn.LeakyReLU(0.2,inplace=True)
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
            self.conv_connection_1 = nn.Conv2d(2, 1, 1,stride=1, padding=0)
            self.sig = nn.Sigmoid()
        
        def freq_connection(self, freq_h, freq_l):
            x = torch.cat((freq_h,freq_l), dim=1)
            #h, w, c = x.shape
            x_max = torch.max(x, dim = 1).values.unsqueeze(1)
            x_sum = torch.sum(x, dim = 1).unsqueeze(1)
            x = torch.cat((x_max, x_sum), dim=1)
            x = self.conv_connection_1(x)
            x = self.sig(x)
            return x * freq_h    
            
        def forward(self, freq_h, feat_1, feat_2, feat_3, feat_4):
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
            return x
            
    class Encoder_L(nn.Module):
        def __init__(self):
            super().__init__()
            #encoder
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
            
        def forward(self, freq_l):
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
            return x, feat_1, feat_2, feat_3, feat_4                        
    
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu  = nn.LeakyReLU(0.2,inplace=True)
            self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.deconv1 = nn.Conv2d(512,256,3,stride=1,padding=1)
            self.debn1 = nn.BatchNorm2d(256)
            self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.deconv2 = nn.Conv2d(256,128,3,stride=1,padding=1)
            self.debn2 = nn.BatchNorm2d(128)
            self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.deconv3 = nn.Conv2d(128,64,3,stride=1,padding=1)
            self.debn3 = nn.BatchNorm2d(64)
            self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.deconv4 = nn.Conv2d(64,32,3,stride=1,padding=1)
            self.debn4 = nn.BatchNorm2d(32)
            self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.deconv5 = nn.Conv2d(32,3,3,stride=1,padding=1)
            self.final = nn.Tanh()
            
        def forward(self, x):
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
            x = self.deconv5(x)
            x = self.final(x)
            return x
    
    def cat_freq(self, LL, LH, HL, HH):
        LL = torch.unsqueeze(LL, 0)
        LH = torch.unsqueeze(LH, 0)
        HL = torch.unsqueeze(HL, 0)
        HH = torch.unsqueeze(HH, 0)
        return torch.cat((LL,LH,HL,HH), dim = 0)
    
    def cat_freq2(self, LH, HL, HH):
        LH = torch.unsqueeze(LH, 0)
        HL = torch.unsqueeze(HL, 0)
        HH = torch.unsqueeze(HH, 0)
        return torch.cat((LH,HL,HH), dim = 0)
    
    def decat_freq(self, input):
        # x_LL = input[0:1, :, :, :, :]
        # x_LL = torch.squeeze(x_LL, dim = 0)
        x_LH = input[0:1, :, :, :, :]
        x_LH = torch.squeeze(x_LH, dim = 0)
        x_HL = input[1:2 ,:, :, :, :]
        x_HL = torch.squeeze(x_HL, dim = 0)
        x_HH = input[2:3, :, :, :, :]
        x_HH = torch.squeeze(x_HH, dim = 0)
        return x_LH, x_HL, x_HH
    
    def forward(self,x):
        LL, LH, HL, HH = self.waveletDecompose(x)
        freq_h = torch.cat((LH,HL,HH),dim=1)
        freq_l = LL
        x_LL, feat_1, feat_2, feat_3, feat_4 = self.encoder_L(freq_l)
        x_H = self.encoder_H(freq_h, feat_1, feat_2, feat_3, feat_4)
        x_LH = x_H[:, 0:512, :, :]
        x_HL = x_H[:, 512:1024, :, :]
        x_HH = x_H[:, 1024:1536, :, :]
        #encoder_com = self.waveletCompose(x_LL, x_LH, x_HL, x_HH)
        #将低频和高频在memory中分开
        encoder_com = self.cat_freq2(x_LH, x_HL, x_HH)
        n,b,c,h,w = encoder_com.shape   #3*batchsize*c*h*w
        encoder_com = encoder_com.permute(1,0,2,3,4).reshape(b, -1).squeeze()
        out_LL = x_LL.reshape(1, b, -1).squeeze()
        res_mem1 = self.mem_rep1(encoder_com)
        res_mem2 = self.mem_rep2(out_LL)
        f1 = res_mem1['output']
        f1 = f1.view(n,b,c,h,w)
        f2 = res_mem2['output']
        f2 = f2.view(1,b,c,h,w)
        out_LL = f2.squeeze()
        out_LH, out_HL, out_HH = self.decat_freq(f1)
        decoder_in = self.waveletCompose(out_LL, out_LH, out_HL, out_HH)
        decoder = self.decoder(decoder_in)
        return decoder, self.waveletCompose(x_LL, x_LH, x_HL, x_HH), decoder_in
    
#AutoEncoderFCMemMoreFeatureArgMax using wavelet and two encoder and different memory
class AutoEncoderTwoEncoderWithWaveletDiffMemAndMyConnect(nn.Module):
    def __init__(self,mem_dim=64,shrink_thres=0.0025,fea_dim=512*2*2):
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
        
        #memory
        self.mem_dim = mem_dim
        # self.mem_rep = MemoryUnitWithArgMax(
        #     mem_dim=mem_dim,
        #     fea_dim=fea_dim,
        #     shrink_thres=shrink_thres)
        # 有问题的三维memory
        # self.mem_rep = MyMemoryUnitWithArgMax(
        #     mem_dim=mem_dim,
        #     fea_dim=fea_dim,
        #     feat_channel=4)
        #self.mem_rep1 = MemoryUnitWithArgMax(
        self.mem_rep1 = MemModule( 
            mem_dim=mem_dim,
            fea_dim=fea_dim * 3,
            shrink_thres=shrink_thres
        )
        
        #self.mem_rep2 = MemoryUnitWithArgMax(
        self.mem_rep2 = MemModule(
            mem_dim=mem_dim,
            fea_dim=fea_dim,
            shrink_thres=shrink_thres
        )

        # decoder 
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv1 = nn.Conv2d(512,256,3,stride=1,padding=1)
        self.debn1 = nn.BatchNorm2d(256)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv2 = nn.Conv2d(256,128,3,stride=1,padding=1)
        self.debn2 = nn.BatchNorm2d(128)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv3 = nn.Conv2d(128,64,3,stride=1,padding=1)
        self.debn3 = nn.BatchNorm2d(64)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv4 = nn.Conv2d(64,32,3,stride=1,padding=1)
        self.debn4 = nn.BatchNorm2d(32)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv5 = nn.Conv2d(32,3,3,stride=1,padding=1)
        self.final = nn.Tanh()

        #wavelet
        self.waveletDecompose = wavelet.WavePool_Two(in_channels=3).cuda()
        self.waveletCompose = wavelet.WaveUnpool(512, 'sum')
        self.waveletResBlock1 = WaveletBlock(3, 32)
        self.waveletResBlock2 = WaveletBlock(32, 64)
        self.waveletResBlock3 = WaveletBlock(64, 128)
        self.waveletResBlock4 = WaveletBlock(128, 256)
        self.waveletResBlock5 = WaveletBlock(256, 512)
        
        self.waveletResBlockDown1 = WaveletBlock(512, 256, upscale=True)
        self.waveletResBlockDown2 = WaveletBlock(256, 128, upscale=True)
        self.waveletResBlockDown3 = WaveletBlock(128, 64, upscale=True)
        self.waveletResBlockDown4 = WaveletBlock(64, 32, upscale=True)
        self.waveletResBlockDown5 = WaveletBlock(32, 3, upscale=True)
        
    def freq_connection(self, freq_data, orig):
        b,c,h,w = freq_data.shape
        b,c_o,h,w = orig.shape
        connect_conv = nn.Conv2d(c, c_o, kernel_size=1, stride=1).cuda()
        connect_bn = nn.BatchNorm2d(c_o).cuda()
        connect_relu = nn.ReLU().cuda()
        out = connect_relu(connect_bn(connect_conv(freq_data)))
        return out 
    
    
    def encoder_H(self, freq_h, comp_H2, comp_H3, comp_H4, comp_H5):
        x = freq_h
        x = self.conv_H1(x)
        x = self.relu(self.bn_H1(x))
        x = self.conv_H2(self.freq_connection(comp_H2, x) + x)
        x = self.relu(self.bn_H2(x))
        x = self.conv_H3(self.freq_connection(comp_H3, x) + x)
        x = self.relu(self.bn_H3(x))
        x = self.conv_H4(self.freq_connection(comp_H4, x) + x)
        x = self.relu(self.bn_H4(x))
        x = self.conv_H5(self.freq_connection(comp_H5, x) + x)
        x = self.relu(self.bn_H5(x))
        return x
    
    def encoder_L(self, freq_l, LL_2, LL_3, LL_4, LL_5):
        x = freq_l
        x = self.conv1(x) #->[32, 32, 256, 256]
        x = self.relu(self.bn1(x))
        x = self.conv2(x + self.freq_connection(LL_2, x)) #->[32, 64, 128, 128]
        x = self.relu(self.bn2(x))
        x = self.conv3(x + self.freq_connection(LL_3, x)) #->[32, 128, 64, 64]
        x = self.relu(self.bn3(x))
        x = self.conv4(x + self.freq_connection(LL_4, x))
        x = self.relu(self.bn4(x))
        x = self.conv5(x + self.freq_connection(LL_5, x))
        x = self.relu(self.bn5(x))
        return x
    
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
        x = self.deconv5(x)
        x = self.final(x)
        return x
    
    def cat_freq(self, LL, LH, HL, HH):
        LL = torch.unsqueeze(LL, 0)
        LH = torch.unsqueeze(LH, 0)
        HL = torch.unsqueeze(HL, 0)
        HH = torch.unsqueeze(HH, 0)
        return torch.cat((LL,LH,HL,HH), dim = 0)
    
    def cat_freq2(self, LH, HL, HH):
        LH = torch.unsqueeze(LH, 0)
        HL = torch.unsqueeze(HL, 0)
        HH = torch.unsqueeze(HH, 0)
        return torch.cat((LH,HL,HH), dim = 0)
    
    def decat_freq(self, input):
        # x_LL = input[0:1, :, :, :, :]
        # x_LL = torch.squeeze(x_LL, dim = 0)
        x_LH = input[ :, 0:512, :, :]
        x_LH = torch.squeeze(x_LH, dim = 0)
        x_HL = input[: ,512:1024, :, :]
        x_HL = torch.squeeze(x_HL, dim = 0)
        x_HH = input[:, 1024:1536, :, :]
        x_HH = torch.squeeze(x_HH, dim = 0)
        return x_LH, x_HL, x_HH
    
    def forward(self,x):
        LL, LH, HL, HH = self.waveletDecompose(x)
        LL_2, LH_2, HL_2, HH_2 = self.waveletDecompose(LL)
        comp_H2 = torch.cat((LH_2, HL_2, HH_2), dim=1)
        LL_3, LH_3, HL_3, HH_3 = self.waveletDecompose(LL_2)
        comp_H3 = torch.cat((LH_3, HL_3, HH_3), dim=1)
        LL_4, LH_4, HL_4, HH_4 = self.waveletDecompose(LL_3)
        comp_H4 = torch.cat((LH_4, HL_4, HH_4), dim=1)
        LL_5, LH_5, HL_5, HH_5 = self.waveletDecompose(LL_4)
        comp_H5 = torch.cat((LH_5, HL_5, HH_5), dim=1)
        freq_h = torch.cat((LH, HL, HH),dim=1)
        freq_l = LL
        x_LL = self.encoder_L(freq_l, LL_2, LL_3, LL_4, LL_5)
        x_H = self.encoder_H(freq_h, comp_H2, comp_H3, comp_H4, comp_H5)
        x_LH = x_H[:, 0:512, :, :]
        x_HL = x_H[:, 512:1024, :, :]
        x_HH = x_H[:, 1024:1536, :, :]
        #encoder_com = self.waveletCompose(x_LL, x_LH, x_HL, x_HH)
        #将低频和高频在memory中分开
        encoder_com = x_H
        b,c,h,w = encoder_com.shape   #3*batchsize*c*h*w
        #encoder_com = encoder_com.reshape(b, -1)
        #out_LL = x_LL.reshape(b, -1)
        res_mem1 = self.mem_rep1(encoder_com)
        res_mem2 = self.mem_rep2(x_LL)
        f1 = res_mem1['output']
        b,c,h,w = x_LH.shape
        #f1 = f1.view(b,3*c,h,w)
        f2 = res_mem2['output']
        #f2 = f2.view(b,c,h,w)
        out_LH, out_HL, out_HH = self.decat_freq(f1)
        decoder_in = self.waveletCompose(f2, out_LH, out_HL, out_HH)
        decoder = self.decoder(decoder_in)
        return decoder, self.waveletCompose(x_LL, x_LH, x_HL, x_HH), decoder_in
    
    
    
    
    #AutoEncoderFCMemMoreFeatureArgMax using wavelet and pyramid structure

class AutoEncoderTwoEncoderWithWaveletPyramid(nn.Module):
    def __init__(self,mem_dim=64,shrink_thres=0.0025,fea_dim=512*2*2):
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
        
        #encoder_H
        self.conv_H1 = nn.Conv2d(3*3,32*3,3,stride=2,padding=1)
        self.bn_H1   = nn.BatchNorm2d(32*3)
        self.conv_H2 = nn.Conv2d(32*3,64*3,3,stride=2,padding=1)
        self.bn_H2   = nn.BatchNorm2d(64*3)
        self.conv_H3 = nn.Conv2d(64*3,128*3,3,stride=2,padding=1)
        self.bn_H3   = nn.BatchNorm2d(128*3)
        self.conv_H4 = nn.Conv2d(128*3,256*3,3,stride=2,padding=1)
        self.bn_H4   = nn.BatchNorm2d(256*3)
        self.conv_H5 = nn.Conv2d(256*3,512*3,3,stride=2,padding=1)
        self.bn_H5   = nn.BatchNorm2d(512*3)
        
        #memory
        self.mem_dim = mem_dim
        # self.mem_rep = MemoryUnitWithArgMax(
        #     mem_dim=mem_dim,
        #     fea_dim=fea_dim,
        #     shrink_thres=shrink_thres)
        # 有问题的三维memory
        # self.mem_rep = MyMemoryUnitWithArgMax(
        #     mem_dim=mem_dim,
        #     fea_dim=fea_dim,
        #     feat_channel=4)
        fea_dim_H1 = int(fea_dim * 3)
        fea_dim_H2 = int(fea_dim / 4 * 3)
        fea_dim_H3 = int(fea_dim / 4 / 4 * 3)
        fea_dim_L1 = int(fea_dim)
        fea_dim_L2 = int(fea_dim / 4)
        fea_dim_L3 = int(fea_dim / 4 / 4)
        self.mem_rep_H1 = MemoryUnitWithArgMax(
            mem_dim=mem_dim,
            fea_dim=fea_dim_H1,
            shrink_thres=shrink_thres
        )
        
        self.mem_rep_H2 = MemoryUnitWithArgMax(
            mem_dim=mem_dim,
            fea_dim=fea_dim_H2,
            shrink_thres=shrink_thres
        )
        
        self.mem_rep_H3 = MemoryUnitWithArgMax(
            mem_dim = mem_dim,
            fea_dim = fea_dim_H3,
            shrink_thres = shrink_thres
        )
        
        self.mem_rep_L1 = MemoryUnitWithArgMax(
            mem_dim=mem_dim,
            fea_dim=fea_dim_L1,
            shrink_thres=shrink_thres
        )
        
        self.mem_rep_L2 = MemoryUnitWithArgMax(
            mem_dim=mem_dim,
            fea_dim=fea_dim_L2,
            shrink_thres=shrink_thres
        )
        
        self.mem_rep_L3 = MemoryUnitWithArgMax(
            mem_dim = mem_dim,
            fea_dim = fea_dim_L3,
            shrink_thres = shrink_thres
        )

        # decoder 
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv1 = nn.Conv2d(512*4,256*4,3,stride=1,padding=1)
        self.debn1 = nn.BatchNorm2d(256*4)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv2 = nn.Conv2d(256*4,128*4,3,stride=1,padding=1)
        self.debn2 = nn.BatchNorm2d(128*4)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv3 = nn.Conv2d(128*4,64*4,3,stride=1,padding=1)
        self.debn3 = nn.BatchNorm2d(64*4)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv4 = nn.Conv2d(64*4,32*4,3,stride=1,padding=1)
        self.debn4 = nn.BatchNorm2d(32*4)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv5 = nn.Conv2d(32*4,3*4,3,stride=1,padding=1)
        self.final = nn.Tanh()

        #wavelet
        self.waveletDecompose = wavelet.WavePool_Two(in_channels=3).cuda()
        self.waveletCompose1 = wavelet.WaveUnpool(3, 'sum')
        self.waveletResBlock1 = WaveletBlock(3, 32)
        self.waveletResBlock2 = WaveletBlock(32, 64)
        self.waveletResBlock3 = WaveletBlock(64, 128)
        self.waveletResBlock4 = WaveletBlock(128, 256)
        self.waveletResBlock5 = WaveletBlock(256, 512)
        
        self.waveletResBlockDown1 = WaveletBlock(512, 256, upscale=True)
        self.waveletResBlockDown2 = WaveletBlock(256, 128, upscale=True)
        self.waveletResBlockDown3 = WaveletBlock(128, 64, upscale=True)
        self.waveletResBlockDown4 = WaveletBlock(64, 32, upscale=True)
        self.waveletResBlockDown5 = WaveletBlock(32, 3, upscale=True)
        
    def freq_connection(self, freq_data, orig):
        b,c,h,w = freq_data.shape
        b,c_o,h,w = orig.shape
        connect_conv = nn.Conv2d(c, c_o, kernel_size=1, stride=1).cuda()
        connect_bn = nn.BatchNorm2d(c_o).cuda()
        connect_relu = nn.ReLU().cuda()
        out = connect_relu(connect_bn(connect_conv(freq_data)))
        return out 
    
    
    def encoder_H(self, freq_h):
        x = freq_h
        x = self.conv_H1(x)
        x = self.relu(self.bn_H1(x))
        x = self.conv_H2(x)
        x = self.relu(self.bn_H2(x))
        x = self.conv_H3(x)
        x = self.relu(self.bn_H3(x))
        x = self.conv_H4(x)
        x = self.relu(self.bn_H4(x))
        x = self.conv_H5(x)
        x = self.relu(self.bn_H5(x))
        return x
    
    def encoder_L(self, freq_l):
        x = freq_l
        x = self.conv1(x) #->[32, 32, 256, 256]
        x = self.relu(self.bn1(x))
        x = self.conv2(x) #->[32, 64, 128, 128]
        x = self.relu(self.bn2(x))
        x = self.conv3(x) #->[32, 128, 64, 64]
        x = self.relu(self.bn3(x))
        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        x = self.conv5(x)
        x = self.relu(self.bn5(x))
        return x
    
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
        x = self.deconv5(x)
        x = self.final(x)
        return x
    
    def cat_freq(self, LL, LH, HL, HH):
        LL = torch.unsqueeze(LL, 0)
        LH = torch.unsqueeze(LH, 0)
        HL = torch.unsqueeze(HL, 0)
        HH = torch.unsqueeze(HH, 0)
        return torch.cat((LL,LH,HL,HH), dim = 0)
    
    def cat_freq2(self, LH, HL, HH):
        LH = torch.unsqueeze(LH, 0)
        HL = torch.unsqueeze(HL, 0)
        HH = torch.unsqueeze(HH, 0)
        return torch.cat((LH,HL,HH), dim = 0)
    
    def decat_freq(self, input):
        # x_LL = input[0:1, :, :, :, :]
        # x_LL = torch.squeeze(x_LL, dim = 0)
        b, c, h, w = input.shape
        step = int(c / 4)
        x_LL = input[ :, 0:step, :, :]
        x_LL = torch.squeeze(x_LL, dim = 0)
        x_LH = input[: ,step:step*2, :, :]
        x_LH = torch.squeeze(x_LH, dim = 0)
        x_HL = input[:, step*2:step*3, :, :]
        x_HL = torch.squeeze(x_HL, dim = 0)
        x_HH = input[:, step*3:c, :, :]
        x_HH = torch.squeeze(x_HH, dim = 0)
        return x_LL, x_LH, x_HL, x_HH
    
    def forward(self,x):
        LL, LH, HL, HH = self.waveletDecompose(x)
        LL_2, LH_2, HL_2, HH_2 = self.waveletDecompose(LL)
        comp_H2 = torch.cat((LH_2, HL_2, HH_2), dim=1)
        LL_3, LH_3, HL_3, HH_3 = self.waveletDecompose(LL_2)
        comp_H3 = torch.cat((LH_3, HL_3, HH_3), dim=1)
        comp_H1 = torch.cat((LH, HL, HH), dim=1)
        freq_L1 = LL
        freq_L2 = LL_2
        freq_L3 = LL_3
        x_L1 = self.encoder_L(freq_L1)
        x_H1 = self.encoder_H(comp_H1)
        b,c,w,h = x_L1.shape
        f1_H = self.mem_rep_H1(x_H1.view(b, -1))
        f1_H = f1_H['output'].view(b,c*3,w,h)
        f1_L = self.mem_rep_L1(x_L1.view(b, -1))
        f1_L = f1_L['output'].view(b,c,w,h) 
        
        x_L2 = self.encoder_L(freq_L2)
        x_H2 = self.encoder_H(comp_H2)
        b,c,w,h = x_L2.shape
        f2_H = self.mem_rep_H2(x_H2.view(b, -1))
        f2_H = f2_H['output'].view(b,c*3,w,h)
        f2_L = self.mem_rep_L2(x_L2.view(b, -1))
        f2_L = f2_L['output'].view(b,c,w,h)
        
        x_L3 = self.encoder_L(freq_L3)
        x_H3 = self.encoder_H(comp_H3)
        b,c,w,h = x_L3.shape
        f3_H = self.mem_rep_H3(x_H3.view(b, -1))
        f3_H = f3_H['output'].view(b,c*3,w,h)
        f3_L = self.mem_rep_L3(x_L3.view(b, -1))
        f3_L = f3_L['output'].view(b,c,w,h)
        
        # x_LH = x_H[:, 0:512, :, :]
        # x_HL = x_H[:, 512:1024, :, :]
        # x_HH = x_H[:, 1024:1536, :, :]
        
        toPyramid1 = torch.cat((x_L1, x_H1), dim=1)
        toPyramid2 = torch.cat((x_L2, x_H2), dim=1)
        toPyramid3 = torch.cat((x_L3, x_H3), dim=1) # LL, LH, HL, HH
        
        # b,c,w,h = toPyramid1.shape
        # res_mem1 = self.mem_rep1(toPyramid1.view(b,-1))
        # f1 = res_mem1['output'].view(b,c,w,h)
        # b,c,w,h = toPyramid2.shape
        # res_mem2 = self.mem_rep2(toPyramid2.view(b,-1))
        # f2 = res_mem2['output'].view(b,c,w,h)
        # b,c,w,h = toPyramid3.shape 
        # res_mem3 = self.mem_rep3(toPyramid3.view(b,-1))
        # f3 = res_mem3['output'].view(b,c,w,h)
        
        beta = 0.5
        
        f3 = torch.cat((f3_H, f3_L), dim=1)
        decoder_out3 = self.decoder(f3)
        feat_LL3, feat_LH3, feat_HL3, feat_HH3 = self.decat_freq(decoder_out3)
        decomp_out3 = self.waveletCompose1(feat_LL3, feat_LH3, feat_HL3, feat_HH3)
        
        f2 = torch.cat((f2_H, f2_L), dim=1)
        decoder_out2 = self.decoder(f2)
        feat_LL2, feat_LH2, feat_HL2, feat_HH2 = self.decat_freq(decoder_out2)
        decomp_out2 = self.waveletCompose1(feat_LL2 * beta + decomp_out3 * (1 - beta), feat_LH2, feat_HL2, feat_HH2) #256*256
        
        f1 = torch.cat((f1_H, f1_L), dim=1)
        decoder_out1 = self.decoder(f1)
        feat_LL1, feat_LH1, feat_HL1, feat_HH1 = self.decat_freq(decoder_out1)
        decomp_out1 = self.waveletCompose1(feat_LL1 * beta + decomp_out2 * (1 - beta), feat_LH1, feat_HL1, feat_HH1) #128*128
        
        #encoder_com = self.waveletCompose(x_LL, x_LH, x_HL, x_HH)
        #将低频和高频在memory中分开
        return decomp_out1, (x_L1,f1_L), (x_H1,f1_H), (x_L2,f2_L), (x_H2,f2_H), (x_L3,f3_L), (x_H3,f3_H)
    
class AutoEncoderTwoEncoderWithWaveletDiffLabel(nn.Module):
    def __init__(self,mem_dim=64,shrink_thres=0.0025,fea_dim=512*2*2):
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
        
        # decoder 
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv1 = nn.Conv2d(512,256,3,stride=1,padding=1)
        self.debn1 = nn.BatchNorm2d(256)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv2 = nn.Conv2d(256,128,3,stride=1,padding=1)
        self.debn2 = nn.BatchNorm2d(128)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv3 = nn.Conv2d(128,64,3,stride=1,padding=1)
        self.debn3 = nn.BatchNorm2d(64)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv4 = nn.Conv2d(64,3,3,stride=1,padding=1)
        self.debn4 = nn.BatchNorm2d(3)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        #self.deconv5 = nn.Conv2d(32,3,3,stride=1,padding=1)
        self.final = nn.Tanh()
        
        self.deconv1_ori = nn.Conv2d(512,256,3,stride=1,padding=1)
        self.debn1_ori = nn.BatchNorm2d(256)
        self.deconv2_ori = nn.Conv2d(256,128,3,stride=1,padding=1)
        self.debn2_ori = nn.BatchNorm2d(128)
        self.deconv3_ori = nn.Conv2d(128,64,3,stride=1,padding=1)
        self.debn3_ori = nn.BatchNorm2d(64)
        self.deconv4_ori = nn.Conv2d(64,32,3,stride=1,padding=1)
        self.debn4_ori = nn.BatchNorm2d(32)
        self.deconv5_ori = nn.Conv2d(32,3,3,stride=1,padding=1)
        
        self.mem_rep_1 = MemModule(
            mem_dim=mem_dim,
            fea_dim=fea_dim,
            shrink_thres=shrink_thres
        )
        self.waveletDecompose = wavelet.WavePool_Two(in_channels=512).cuda()
        
    def decoder_HH(self, x):
        x = self.up1(x) #[-1, 512, 8, 8]
        # [-1, 512, 8, 8] -> [-1, 256, 16, 16]
        x = self.relu(self.debn1(self.deconv1(x)))
        x = self.up2(x)
        # [-1, 256, 16, 16] -> [-1, 128, 32, 32]
        x = self.relu(self.debn2(self.deconv2(x)))
        x = self.up3(x)
        # [-1, 128, 64, 64] -> [-1, 64, 128, 128] 
        x = self.relu(self.debn3(self.deconv3(x)))
        x = self.up4(x)
        # [-1, 64, 128, 128] -> [-1, 32, 256, 256] 
        x = self.relu(self.debn4(self.deconv4(x)))
        #x = self.up5(x)
        #x = self.deconv5(x)
        x = self.final(x)
        return x
    
    def decoder_HL(self, x):
        x = self.up1(x) #[-1, 512, 8, 8]
        x = self.relu(self.debn1(self.deconv1(x)))
        x = self.up2(x)
        x = self.relu(self.debn2(self.deconv2(x)))
        x = self.up3(x)
        x = self.relu(self.debn3(self.deconv3(x)))
        x = self.up4(x)
        x = self.relu(self.debn4(self.deconv4(x)))
        x = self.final(x)
        return x
    def decoder_LH(self, x):
        x = self.up1(x) #[-1, 512, 8, 8]
        x = self.relu(self.debn1(self.deconv1(x)))
        x = self.up2(x)
        x = self.relu(self.debn2(self.deconv2(x)))
        x = self.up3(x)
        x = self.relu(self.debn3(self.deconv3(x)))
        x = self.up4(x)
        x = self.relu(self.debn4(self.deconv4(x)))
        x = self.final(x)
        return x
    def decoder_LL(self, x):
        x = self.up1(x) #[-1, 512, 8, 8]
        x = self.relu(self.debn1(self.deconv1(x)))
        x = self.up2(x)
        x = self.relu(self.debn2(self.deconv2(x)))
        x = self.up3(x)
        x = self.relu(self.debn3(self.deconv3(x)))
        x = self.up4(x)
        x = self.relu(self.debn4(self.deconv4(x)))
        x = self.final(x)
        return x
    
    def decoder_ori(self, x):
        x = self.up1(x) #[-1, 512, 32, 32]
        # [-1, 512, 32, 32] -> [-1, 256, 64, 64]
        x = self.relu(self.debn1_ori(self.deconv1_ori(x)))
        x = self.up2(x)
        # [-1, 256, 64, 64] -> [-1, 128, 128, 128]
        x = self.relu(self.debn2_ori(self.deconv2_ori(x)))
        x = self.up3(x)
        # [-1, 128, 128, 128] -> [-1, 64, 256, 256] 
        x = self.relu(self.debn3_ori(self.deconv3_ori(x)))
        x = self.up4(x)
        x = self.relu(self.debn4_ori(self.deconv4_ori(x)))
        x = self.up5(x)
        x = self.deconv5_ori(x)
        x = self.final(x)
        return x    
    
    
    def encoder(self, x):
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        x = self.conv3(x)
        x = self.relu(self.bn3(x))
        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        x = self.conv5(x)
        x = self.relu(self.bn5(x))
        return x
    
    def decat_freq(self, input):
        # x_LL = input[0:1, :, :, :, :]
        # x_LL = torch.squeeze(x_LL, dim = 0)
        b, c, h, w = input.shape
        step = int(c / 4)
        x_LL = input[ :, 0:step, :, :]
        x_LL = torch.squeeze(x_LL, dim = 0)
        x_LH = input[: ,step:step*2, :, :]
        x_LH = torch.squeeze(x_LH, dim = 0)
        x_HL = input[:, step*2:step*3, :, :]
        x_HL = torch.squeeze(x_HL, dim = 0)
        x_HH = input[:, step*3:c, :, :]
        x_HH = torch.squeeze(x_HH, dim = 0)
        return x_LL, x_LH, x_HL, x_HH
    
    def forward(self,x):
        encoder_out = self.encoder(x) #->32, 512, 16, 16
        # b,c,w,h = encoder_out.shape
        #x_LL, x_LH, x_HL, x_HH = self.decat_freq(x)
        mem = self.mem_rep_1(encoder_out)
        # mem_LH = self.mem_rep_2(x)
        # mem_HL = self.mem_rep_3(x)
        # mem_HH = self.mem_rep_4(x)
        
        # att = mem_LL['att']
        
        #b,c,w,h = x_LL.shape
        decoder_LL = self.decoder_LL(mem['output'])
        decoder_LH = self.decoder_LH(mem['output'])
        decoder_HL = self.decoder_HL(mem['output'])
        decoder_HH = self.decoder_HH(mem['output'])
        
        x_recon = self.decoder_ori(mem['output'])
        # return x, (decoder_LL, mem_LL['output'], x), (decoder_LH, mem_LH['output'], x), (decoder_HL,mem_HL['output'], x), (decoder_HH,mem_HH['output'], x), x_recon
        return decoder_LL, decoder_LH, decoder_HL, decoder_HH, x_recon, (encoder_out, mem['output'])
    
class AutoEncoderFourEncoderWithWavelet(nn.Module):
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
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
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(self.bn1(x))
            x = self.conv2(x)
            x = self.relu(self.bn2(x))
            x = self.conv3(x)
            x = self.relu(self.bn3(x))
            x = self.conv4(x)
            x = self.relu(self.bn4(x))
            x = self.conv5(x)
            x = self.relu(self.bn5(x))
            return x
    
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu  = nn.LeakyReLU(0.2,inplace=True)
            self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.deconv1 = nn.Conv2d(512,256,3,stride=1,padding=1)
            self.debn1 = nn.BatchNorm2d(256)
            self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.deconv2 = nn.Conv2d(256,128,3,stride=1,padding=1)
            self.debn2 = nn.BatchNorm2d(128)
            self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.deconv3 = nn.Conv2d(128,64,3,stride=1,padding=1)
            self.debn3 = nn.BatchNorm2d(64)
            self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.deconv4 = nn.Conv2d(64,32,3,stride=1,padding=1)
            self.debn4 = nn.BatchNorm2d(32)
            self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.deconv5 = nn.Conv2d(32,3,3,stride=1,padding=1)
            self.final = nn.Tanh()
            
        def forward(self, x):
            x = self.up1(x) #[-1, 512, 8, 8]
            # [-1, 512, 8, 8] -> [-1, 256, 16, 16]
            x = self.relu(self.debn1(self.deconv1(x)))
            x = self.up2(x)
            # [-1, 256, 16, 16] -> [-1, 128, 32, 32]
            x = self.relu(self.debn2(self.deconv2(x)))
            x = self.up3(x)
            # [-1, 128, 64, 64] -> [-1, 64, 128, 128] 
            x = self.relu(self.debn3(self.deconv3(x)))
            x = self.up4(x)
            # [-1, 64, 128, 128] -> [-1, 32, 256, 256] 
            x = self.relu(self.debn4(self.deconv4(x)))
            x = self.up5(x)
            x = self.deconv5(x)
            x = self.final(x)
            return x
        
    def __init__(self,mem_dim=64,shrink_thres=0.0025,fea_dim=512*2*2):
        super().__init__()      
        self.deconv1_ori = nn.Conv2d(512,256,3,stride=1,padding=1)
        self.debn1_ori = nn.BatchNorm2d(256)
        self.deconv2_ori = nn.Conv2d(256,128,3,stride=1,padding=1)
        self.debn2_ori = nn.BatchNorm2d(128)
        self.deconv3_ori = nn.Conv2d(128,64,3,stride=1,padding=1)
        self.debn3_ori = nn.BatchNorm2d(64)
        self.deconv4_ori = nn.Conv2d(64,32,3,stride=1,padding=1)
        self.debn4_ori = nn.BatchNorm2d(32)
        self.deconv5_ori = nn.Conv2d(32,3,3,stride=1,padding=1)
        
        self.mem_rep_1 = MemModule(
            mem_dim=mem_dim,
            fea_dim=fea_dim,
            shrink_thres=shrink_thres
        )
        
        self.mem_rep_2 = MemModule(
            mem_dim=mem_dim,
            fea_dim=fea_dim,
            shrink_thres=shrink_thres
        )
        
        self.mem_rep_3 = MemModule(
            mem_dim=mem_dim,
            fea_dim=fea_dim ,
            shrink_thres=shrink_thres
        )
        
        self.mem_rep_4 = MemModule(
            mem_dim=mem_dim,
            fea_dim=fea_dim,
            shrink_thres=shrink_thres
        )
        self.waveletDecompose = wavelet.WavePool_Two(in_channels=3).cuda()
        self.waveletCompose = wavelet.WaveUnpool(3, 'sum')
        self.encoder_HH = self.Encoder()
        self.encoder_LH = self.Encoder()
        self.encoder_HL = self.Encoder()
        self.encoder_LL = self.Encoder()
        
        self.decoder_HH = self.Decoder()
        self.decoder_LH = self.Decoder()
        self.decoder_HL = self.Decoder()
        self.decoder_LL = self.Decoder()
    
    def decoder_ori(self, x):
        x = self.up1(x) #[-1, 512, 32, 32]
        # [-1, 512, 32, 32] -> [-1, 256, 64, 64]
        x = self.relu(self.debn1_ori(self.deconv1_ori(x)))
        x = self.up2(x)
        # [-1, 256, 64, 64] -> [-1, 128, 128, 128]
        x = self.relu(self.debn2_ori(self.deconv2_ori(x)))
        x = self.up3(x)
        # [-1, 128, 128, 128] -> [-1, 64, 256, 256] 
        x = self.relu(self.debn3_ori(self.deconv3_ori(x)))
        x = self.up4(x)
        x = self.relu(self.debn4_ori(self.deconv4_ori(x)))
        x = self.up5(x)
        x = self.deconv5_ori(x)
        x = self.final(x)
        return x    
    
    def decat_freq(self, input):
        # x_LL = input[0:1, :, :, :, :]
        # x_LL = torch.squeeze(x_LL, dim = 0)
        b, c, h, w = input.shape
        step = int(c / 4)
        x_LL = input[ :, 0:step, :, :]
        x_LL = torch.squeeze(x_LL, dim = 0)
        x_LH = input[: ,step:step*2, :, :]
        x_LH = torch.squeeze(x_LH, dim = 0)
        x_HL = input[:, step*2:step*3, :, :]
        x_HL = torch.squeeze(x_HL, dim = 0)
        x_HH = input[:, step*3:c, :, :]
        x_HH = torch.squeeze(x_HH, dim = 0)
        return x_LL, x_LH, x_HL, x_HH
    
    def forward(self,x):
        x_LL, x_LH, x_HL, x_HH = self.waveletDecompose(x)
        enc_LL = self.encoder_LL(x_LL) #->32, 512, 16, 16
        enc_LH = self.encoder_LH(x_LH)
        enc_HL = self.encoder_HL(x_HL)
        enc_HH = self.encoder_HH(x_HH)
        b,c,w,h = x_LL.shape
        #x_LL, x_LH, x_HL, x_HH = self.decat_freq(x)
        mem_LL = self.mem_rep_1(enc_LL)
        mem_LH = self.mem_rep_2(enc_LH)
        mem_HL = self.mem_rep_3(enc_HL)
        mem_HH = self.mem_rep_4(enc_HH)
        
        att = mem_LL['att']
        
        #b,c,w,h = x_LL.shape
        decoder_LL = self.decoder_LL(mem_LL['output'])
        decoder_LH = self.decoder_LH(mem_LH['output'])
        decoder_HL = self.decoder_HL(mem_HL['output'])
        decoder_HH = self.decoder_HH(mem_HH['output'])
        
        #x_recon = self.decoder_ori(x)
        x_recon = self.waveletCompose(decoder_LL, decoder_LH, decoder_HL, decoder_HH)
        return x, (decoder_LL, mem_LL['output'], enc_LL), (decoder_LH, mem_LH['output'], enc_LH), (decoder_HL,mem_HL['output'], enc_HL), (decoder_HH,mem_HH['output'], enc_HH), x_recon
    
class pyramidWithWavelet(nn.Module):
    def __init__(self, image_size=512, channel=3):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_H = nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(3)
        self.bn_H = nn.BatchNorm2d(9)
        self.relu = nn.ReLU()
        self.conv1_1 = nn.Conv2d(27, 9,kernel_size=1, stride=1, padding=0)
        self.conv1_2 = nn.Conv2d(3, 3,kernel_size=1, stride=1, padding=0)
        self.shallow_conv = nn.Conv2d(3,3,kernel_size=3, stride=1, padding=1)
        self.convdown = nn.Conv2d(27, 27, 3, stride=1, padding=1)
        self.conv1x1_H = nn.Conv2d(9, 9, 1, stride=1, padding=0)
        self.conv1x1_L = nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.shallow_conv_H = nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.PixelShuffle(2)
        self.waveletDecompose = wavelet.WavePool_Two(in_channels=3).cuda()
        self.waveletCompose = wavelet.WaveUnpool(3, 'sum')
        self.conv_connection_1 = nn.Conv2d(2, 1, 1,stride=1, padding=0)
        self.sig = nn.Sigmoid()
        
    def freq_connection(self, freq_h, freq_l):
        x = torch.cat((freq_h,freq_l), dim=1)
        #h, w, c = x.shape
        x_max = torch.max(x, dim = 1).values.unsqueeze(1)
        x_sum = torch.sum(x, dim = 1).unsqueeze(1)
        x = torch.cat((x_max, x_sum), dim=1)
        x = self.conv_connection_1(x)
        x = self.sig(x)
        return x * freq_h
        
    def HBlock(self, x):
        x1 = self.relu(self.bn_H(self.conv_H(x)))
        x2 = self.relu(self.bn_H(self.conv_H(x + x1)))
        x3 = self.relu(self.bn_H(self.conv_H(x + x1 + x2)))
        x_concat = torch.cat((x, x1, x3), dim=1)
        x4 = self.conv1_1(x_concat)
        return x + x4
        
    def LBlock(self, x):
        x1 = self.relu(self.bn(self.conv(x)))
        x1 = self.relu(self.bn(self.conv(x1)))
        return x1
        
    def LStructure(self, x):
        x = self.shallow_conv(x)
        x1 = self.LBlock(x)
        x2 = self.LBlock(x + x1)
        x3 = self.LBlock(x + x2)
        x_out = self.conv1_2(x + x3)
        # x_out = self.upsample(x_out)
        # x_out = self.conv1x1_L(x_out)
        return x1, x2, x3, x_out
        
    def HStructure(self, x, y1, y2, y3):
        y1_rep = y1.repeat(1,3,1,1)
        y2_rep = y2.repeat(1,3,1,1)
        y3_rep = y3.repeat(1,3,1,1)
        x = self.shallow_conv_H(x)
        x_rep = x.repeat(1,3,1,1)
        x1 = self.HBlock(x)
        z1 = self.freq_connection(x1, y1) + y1_rep #y1[32, 64, 128, 128], x1[32, 192, 128, 128]
        x2 = self.HBlock(x1 + z1)
        z2 = self.freq_connection(x2, y2) + y2_rep
        x3 = self.HBlock(x2 + z2)
        z3 = self.freq_connection(x3, y3) + y3_rep
        x_cat = torch.cat((z1, z2, z3), dim = 1)
        x_out = self.convdown(x_cat + x_rep)
        # x_out = self.upsample(x_out + x_rep)
        # x_out = self.conv1x1_H(x_out)
        return x_out
    
    def forward(self, x):
        LL, LH, HL, HH = self.waveletDecompose(x) #256
        x_H = torch.cat((LH,HL,HH), dim=1)
        
        LL_1, LH_1, HL_1, HH_1 = self.waveletDecompose(LL) #128
        x_H1 = torch.cat((LH_1, HL_1, HH_1), dim=1)
        
        LL_2, LH_2, HL_2, HH_2 = self.waveletDecompose(LL_1) #64
        x_H2 = torch.cat((LH_2, HL_2, HH_2), dim=1)
        
        x_L = LL
        y1, y2, y3, x_L = self.LStructure(x_L)
        x_H = self.HStructure(x_H,y1,y2,y3)        
        
        x_L1 = LL_1
        y1, y2, y3, x_L1 = self.LStructure(x_L1)
        x_H1 = self.HStructure(x_H1, y1, y2, y3)
        
        x_L2 = LL_2
        y1, y2, y3, x_L2 = self.LStructure(x_L2)
        x_H2 = self.HStructure(x_H2, y1, y2, y3)
        
        x_sp3 = torch.split(x_H2, 3, dim=1)
        img_3 = self.waveletCompose(x_L2, x_sp3[0], x_sp3[1], x_sp3[2])
        
        x_sp2 = torch.split(x_H1, 3, dim=1)
        img_2 = self.waveletCompose((x_L1 + img_3)/2.0, x_sp2[0], x_sp2[1], x_sp2[2])
        
        x_sp1 = torch.split(x_H, 3, dim=1)
        img_1 = self.waveletCompose((x_L + img_2) / 2.0, x_sp1[0], x_sp1[1], x_sp1[2])
        
        return img_1
     
class MemAutoEncoderFFT(nn.Module):
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.LRelu = nn.LeakyReLU(0.2, inplace=True)
            self.freq_conv1 = nn.Conv2d(3, 3, (1,1), stride=1, padding=0)
            self.freq_conv_up1 = nn.Conv2d(3, 32, (1,1), stride=1, padding=0)
            self.bn1 = nn.BatchNorm2d(3)
            self.bn_up1 = nn.BatchNorm2d(32)
            
            self.freq_conv2 = nn.Conv2d(32, 32, (1,1), stride=1, padding=0)
            self.freq_conv_up2 = nn.Conv2d(32, 64, (1,1), stride=1, padding=0)
            self.bn2 = nn.BatchNorm2d(32)
            self.bn_up2 = nn.BatchNorm2d(64)
            
            self.freq_conv3 = nn.Conv2d(64, 64, (1,1), stride=1, padding=0)
            self.freq_conv_up3 = nn.Conv2d(64, 128, (1,1), stride=1, padding=0)
            self.bn3 = nn.BatchNorm2d(64)
            self.bn_up3 = nn.BatchNorm2d(128)
            
            self.freq_conv4 = nn.Conv2d(128, 128, (1,1), stride=1, padding=0)
            self.freq_conv_up4 = nn.Conv2d(128, 256, (1,1), stride=1, padding=0)
            self.bn4 = nn.BatchNorm2d(128)
            self.bn_up4 = nn.BatchNorm2d(256)
            
            self.freq_conv5 = nn.Conv2d(256, 256, (1,1), stride=1, padding=0)
            self.freq_conv_up5 = nn.Conv2d(256, 512, (1,1), stride=1, padding=0)
            self.bn5 = nn.BatchNorm2d(256)
            self.bn_up5 = nn.BatchNorm2d(512)
                
            self.freq_conv_list = [self.freq_conv1, self.freq_conv2, self.freq_conv3, self.freq_conv4, self.freq_conv5]
            self.freq_conv_up_list = [self.freq_conv_up1, self.freq_conv_up2, self.freq_conv_up3, self.freq_conv_up4, self.freq_conv_up5]
            self.bn_list = [self.bn1, self.bn2, self.bn3, self.bn4, self.bn5]
            self.bn_up_list = [self.bn_up1, self.bn_up2, self.bn_up3, self.bn_up4, self.bn_up5]
            
        def forward(self, x):
            #pha_channel = self.LRelu(self.bn1(self.freq_conv1(x)))
            pha_channel = self.LRelu(self.bn_up1(self.freq_conv_up1(x)))
            #pha_channel = self.LRelu(self.bn2(self.freq_conv2(pha_channel)))
            pha_channel = self.LRelu(self.bn_up2(self.freq_conv_up2(pha_channel)))
            #pha_channel = self.LRelu(self.bn3(self.freq_conv3(pha_channel)))
            pha_channel = self.LRelu(self.bn_up3(self.freq_conv_up3(pha_channel)))
            #pha_channel = self.LRelu(self.bn4(self.freq_conv4(pha_channel)))
            pha_channel = self.LRelu(self.bn_up4(self.freq_conv_up4(pha_channel)))
            #pha_channel = self.LRelu(self.bn5(self.freq_conv5(pha_channel)))
            pha_channel = self.LRelu(self.bn_up5(self.freq_conv_up5(pha_channel)))
            return pha_channel
    
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.LRelu = nn.LeakyReLU(0.2, inplace=True)
            self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.freq_deconv1 = nn.Conv2d(512, 256, 1, stride=1, padding=0)
            self.freq_in_deconv1 = nn.Conv2d(256, 256, 1, stride=1, padding=0)
            self.bn_de1 = nn.BatchNorm2d(256)
            
            self.freq_deconv2 = nn.Conv2d(256, 128, 1, stride=1, padding=0)
            self.freq_in_deconv2 = nn.Conv2d(128, 128, 1, stride=1, padding=0)
            self.bn_de2 = nn.BatchNorm2d(128)
            
            self.freq_deconv3 = nn.Conv2d(128, 64, 1, stride=1, padding=0)
            self.freq_in_deconv3 = nn.Conv2d(64, 64, 1, stride=1, padding=0)
            self.bn_de3 = nn.BatchNorm2d(64)
            
            self.freq_deconv4 = nn.Conv2d(64, 32, 1, stride=1, padding=0)
            self.freq_in_deconv4 = nn.Conv2d(32, 32, 1, stride=1, padding=0)
            self.bn_de4 = nn.BatchNorm2d(32)
            
            self.freq_deconv5 = nn.Conv2d(32, 3, 1, stride=1, padding=0)
            self.freq_in_deconv5 = nn.Conv2d(3, 3, 1, stride=1, padding=0)
            self.bn_de5 = nn.BatchNorm2d(3)
            
            # self.freq_deconv_list = [self.freq_deconv1, self.freq_deconv2, self.freq_deconv3, self.freq_deconv4, self.freq_deconv5]
            # self.freq_in_deconv_list = [self.freq_in_deconv1, self.freq_in_deconv2, self.freq_in_deconv3, self.freq_in_deconv4, self.freq_in_deconv5]
            # self.bn_de_list = [self.bn_de1, self.bn_de2, self.bn_de3, self.bn_de4, self.bn_de5]
            # #self.freq_inva_list = [self.freq_inva_deconv1, self.freq_inva_deconv2, self.freq_inva_deconv3, self.freq_inva_deconv4, self.freq_inva_deconv5]
            # #self.spat_inva_list = [self.spat_inva_deconv1, self.spat_inva_deconv2, self.spat_inva_deconv3, self.spat_inva_deconv4, self.spat_inva_deconv5]
        def forward(self, x):
            pha_channel = self.LRelu(self.bn_de1(self.freq_deconv1(x)))
            #pha_channel = self.LRelu(self.bn_de1(self.freq_in_deconv1(x)))
            pha_channel = self.LRelu(self.bn_de2(self.freq_deconv2(pha_channel)))
            #pha_channel = self.LRelu(self.bn_de2(self.freq_in_deconv2(pha_channel)))
            pha_channel = self.LRelu(self.bn_de3(self.freq_deconv3(pha_channel)))
            #pha_channel = self.LRelu(self.bn_de3(self.freq_in_deconv3(pha_channel)))
            pha_channel = self.LRelu(self.bn_de4(self.freq_deconv4(pha_channel)))
            #pha_channel = self.LRelu(self.bn_de4(self.freq_in_deconv4(pha_channel)))
            pha_channel = self.LRelu(self.bn_de5(self.freq_deconv5(pha_channel)))
            #pha_channel = self.LRelu(self.bn_de5(self.freq_in_deconv5(pha_channel)))
            return pha_channel
            
    def __init__(self, mem_dim=128,shrink_thres=0.0025,fea_dim=512):
        super().__init__()
        self.mem_rep = MemModule(
            mem_dim=mem_dim,
            fea_dim=fea_dim,
            shrink_thres=shrink_thres
        )
        self.mag_decoder = self.Decoder()
        self.pha_decoder = self.Decoder()
        self.mag_encoder = self.Encoder()
        self.pha_encoder = self.Encoder()
        
    # def freq_block(self, mag_fea, pha_fea, i):
    #     bn = self.bn_list[i]
    #     bn_up = self.bn_up_list[i]
    #     freq_conv = self.freq_conv_list[i]
    #     freq_conv_up = self.freq_conv_up_list[i]
        
    #     pha_channel = self.LRelu(bn(freq_conv(pha_fea)))
    #     pha_channel = self.LRelu(bn_up(freq_conv_up(pha_channel)))
    #     pha_channel = pha_channel
        
    #     mag_channel = self.LRelu(bn(freq_conv(mag_fea)))
    #     mag_channel = self.LRelu(bn_up(freq_conv_up(mag_channel)))
    #     mag_channel = mag_channel
        
    #     return mag_channel, pha_channel
        
    # def encoder(self, mag, pha):
    #     pha_channel = self.LRelu(self.bn1(self.freq_conv1(pha)))
    #     pha_channel = self.LRelu(self.bn_up1(self.freq_conv_up1(pha_channel)))
    #     pha_channel = self.LRelu(self.bn2(self.freq_conv2(pha_channel)))
    #     pha_channel = self.LRelu(self.bn_up2(self.freq_conv_up2(pha_channel)))
    #     pha_channel = self.LRelu(self.bn3(self.freq_conv3(pha_channel)))
    #     pha_channel = self.LRelu(self.bn_up3(self.freq_conv_up3(pha_channel)))
    #     pha_channel = self.LRelu(self.bn4(self.freq_conv4(pha_channel)))
    #     pha_channel = self.LRelu(self.bn_up4(self.freq_conv_up4(pha_channel)))
    #     pha_channel = self.LRelu(self.bn5(self.freq_conv5(pha_channel)))
    #     pha_channel = self.LRelu(self.bn_up5(self.freq_conv_up5(pha_channel)))
        
    #     mag_channel = self.LRelu(self.bn1(self.freq_conv1(mag)))
    #     mag_channel = self.LRelu(self.bn_up1(self.freq_conv_up1(mag_channel)))
    #     mag_channel = self.LRelu(self.bn2(self.freq_conv2(mag_channel)))
    #     mag_channel = self.LRelu(self.bn_up2(self.freq_conv_up2(mag_channel)))
        
        
    #     mag, pha = self.freq_block(mag, pha, 0) #32,32,256,256
    #     mag, pha = self.freq_block(mag, pha, 1)
    #     mag, pha = self.freq_block(mag, pha, 2)
    #     mag, pha = self.freq_block(mag, pha, 3)
    #     mag, pha = self.freq_block(mag, pha, 4)
    #     x = torch.cat((mag, pha), dim=1)
    #     return x
              
    # def freq_deblock(self, mag, pha, i):
    #     bn = self.bn_de_list[i]
    #     freq_deconv = self.freq_deconv_list[i]
    #     freq_in_deconv = self.freq_in_deconv_list[i]
    #     mag = self.up1(mag)#32,512,32,32
    #     pha = self.up1(pha)

    #     pha_channel = self.LRelu(bn(freq_deconv(pha)))
    #     pha_channel = self.LRelu(bn(freq_in_deconv(pha_channel)))
    #     pha_channel = pha_channel

    #     mag_channel = self.LRelu(bn(freq_deconv(mag)))
    #     mag_channel = self.LRelu(bn(freq_in_deconv(mag_channel)))
    #     mag_channel = mag_channel
    
    #     return mag_channel, pha_channel
        
    # def decoder(self, mag, pha):
    #     mag, pha = self.freq_deblock(mag, pha, 0)
    #     mag, pha = self.freq_deblock(mag, pha, 1)
    #     mag, pha = self.freq_deblock(mag, pha, 2)
    #     mag, pha = self.freq_deblock(mag, pha, 3)
    #     mag, pha = self.freq_deblock(mag, pha, 4)
    #     return mag, pha
    
    def preprocess(self, x):
        freq = torch.fft.rfft2(x)
        mag = torch.abs(freq)
        pha = torch.angle(freq)
        return mag, pha
    
    def forward(self, x):
        mag, pha = self.preprocess(x)
        encoder_out_mag = self.mag_encoder(mag) #32, 512, 16, 16
        encoder_out_pha = self.mag_encoder(pha)
        encoder_out = torch.cat((encoder_out_mag, encoder_out_pha), dim=1)
        mem = self.mem_rep(encoder_out)
        b,c,w,h = mem['output'].shape
        mag = mem['output'][:, :int(c/2) , :, :]
        pha = mem['output'][:, int(c/2): , :, :]
        mag_out = self.mag_decoder(mag)
        pha_out = self.pha_decoder(pha)
        return encoder_out, mem['output'], mag_out, pha_out
    
class MemAutoEncoderDCT(nn.Module):
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.LRelu = nn.LeakyReLU(0.2, inplace=True)
            self.dct_conv1 = nn.Conv2d(3, 3, 1, stride=1, padding=0)
            self.dct_conv_up1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
            self.mag_conv1 = nn.Conv2d(3, 3, 1, stride=1, padding=0)
            self.mag_conv_up1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
            self.pha_conv1 = nn.Conv2d(3, 3, 1, stride=1, padding=0)
            self.pha_conv_up1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
            self.freq_to_dct1 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
            self.dct_to_freq1 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(3)
            self.bn_up1 = nn.BatchNorm2d(32)
            
            #self.spat_conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
            self.dct_conv2 = nn.Conv2d(32, 32, 1, stride=1, padding=0)
            self.dct_conv_up2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
            self.mag_conv2 = nn.Conv2d(32, 32, 1, stride=1, padding=0)
            self.mag_conv_up2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
            self.pha_conv2 = nn.Conv2d(32, 64, 1, stride=1, padding=0)
            self.pha_conv_up2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
            self.freq_to_dct2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
            self.dct_to_freq2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.bn_up2 = nn.BatchNorm2d(64)
            
            #self.spat_conv3 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
            self.dct_conv3 = nn.Conv2d(64, 64, 1, stride=1, padding=0)
            self.dct_conv_up3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
            self.mag_conv3 = nn.Conv2d(64, 64, 1, stride=1, padding=0)
            self.mag_conv_up3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
            self.pha_conv3 = nn.Conv2d(64, 64, 1, stride=1, padding=0)
            self.pha_conv_up3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
            self.freq_to_dct3 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
            self.dct_to_freq3 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
            self.bn3 = nn.BatchNorm2d(64)
            self.bn_up3 = nn.BatchNorm2d(128)
            
            self.dct_conv4 = nn.Conv2d(128, 128, 1, stride=1, padding=0)
            self.dct_conv_up4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
            self.mag_conv4 = nn.Conv2d(128, 128, 1, stride=1, padding=0)
            self.mag_conv_up4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
            self.pha_conv4 = nn.Conv2d(128, 128, 1, stride=1, padding=0)
            self.pha_conv_up4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
            self.freq_to_dct4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
            self.dct_to_freq4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
            self.bn4 = nn.BatchNorm2d(128)
            self.bn_up4 = nn.BatchNorm2d(256)
            
            self.dct_conv5 = nn.Conv2d(256, 256, 1, stride=1, padding=0)
            self.dct_conv_up5 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
            self.mag_conv5 = nn.Conv2d(256, 256, 1, stride=1, padding=0)
            self.mag_conv_up5 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
            self.pha_conv5 = nn.Conv2d(256, 256, 1, stride=1, padding=0)
            self.pha_conv_up5 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
            self.freq_to_dct5 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
            self.dct_to_freq5 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
            self.bn5 = nn.BatchNorm2d(256)
            self.bn_up5 = nn.BatchNorm2d(512)

        def decompose(self, x):
            freq = torch.fft.rfft2(x)
            mag = torch.abs(freq)
            pha = torch.angle(freq)
            return mag, pha
        
        def forward(self, mag, pha, dct):
            res_pha = []
            res_mag = []
            res_dct = []
            pha_channel = self.LRelu(self.bn_up1(self.pha_conv_up1(pha)))
            mag_channel = self.LRelu(self.bn_up1(self.mag_conv_up1(mag)))
            dct_channel = self.LRelu(self.bn1(self.dct_conv1(dct)))
            dct_channel = self.LRelu(self.bn_up1(self.dct_conv_up1(dct_channel)))
            spat = torch.fft.irfft2(mag_channel * torch.exp(1j * pha_channel))
            to_dct = self.freq_to_dct1(spat)
            to_freq = self.dct_to_freq1(dct_channel)
            to_mag, to_pha = self.decompose(torch_dct.idct_2d(to_freq))
            dct_channel = dct_channel + to_dct
            res_dct.append(dct_channel)
            mag_channel = mag_channel + to_mag
            res_mag.append(mag_channel)
            pha_channel = pha_channel + to_pha
            res_pha.append(pha_channel)
            
            pha_channel = self.LRelu(self.bn_up2(self.pha_conv_up2(pha_channel)))
            mag_channel = self.LRelu(self.bn_up2(self.mag_conv_up2(mag_channel)))
            dct_channel = self.LRelu(self.bn2(self.dct_conv2(dct_channel)))
            dct_channel = self.LRelu(self.bn_up2(self.dct_conv_up2(dct_channel)))
            spat = torch.fft.irfft2(mag_channel * torch.exp(1j * pha_channel))
            to_dct = self.freq_to_dct2(spat)
            to_freq = self.dct_to_freq2(dct_channel)
            to_mag, to_pha = self.decompose(torch_dct.idct_2d(to_freq))
            dct_channel = dct_channel + to_dct
            res_dct.append(dct_channel)
            mag_channel = mag_channel + to_mag
            res_mag.append(mag_channel)
            pha_channel = pha_channel + to_pha
            res_pha.append(pha_channel)
            
            pha_channel = self.LRelu(self.bn_up3(self.pha_conv_up3(pha_channel)))
            mag_channel = self.LRelu(self.bn_up3(self.mag_conv_up3(mag_channel)))
            dct_channel = self.LRelu(self.bn3(self.dct_conv3(dct_channel)))
            dct_channel = self.LRelu(self.bn_up3(self.dct_conv_up3(dct_channel)))
            spat = torch.fft.irfft2(mag_channel * torch.exp(1j * pha_channel))
            to_dct = self.freq_to_dct3(spat)
            to_freq = self.dct_to_freq3(dct_channel)
            to_mag, to_pha = self.decompose(torch_dct.idct_2d(to_freq))
            dct_channel = dct_channel + to_dct
            res_dct.append(dct_channel)
            mag_channel = mag_channel + to_mag
            res_mag.append(mag_channel)
            pha_channel = pha_channel + to_pha
            res_pha.append(pha_channel)
            
            pha_channel = self.LRelu(self.bn_up4(self.pha_conv_up4(pha_channel)))
            mag_channel = self.LRelu(self.bn_up4(self.mag_conv_up4(mag_channel)))
            dct_channel = self.LRelu(self.bn4(self.dct_conv4(dct_channel)))
            dct_channel = self.LRelu(self.bn_up4(self.dct_conv_up4(dct_channel)))
            spat = torch.fft.irfft2(mag_channel * torch.exp(1j * pha_channel))
            to_dct = self.freq_to_dct4(spat)
            to_freq = self.dct_to_freq4(dct_channel)
            to_mag, to_pha = self.decompose(torch_dct.idct_2d(to_freq))
            dct_channel = dct_channel + to_dct
            res_dct.append(dct_channel)
            mag_channel = mag_channel + to_mag
            res_mag.append(mag_channel)
            pha_channel = pha_channel + to_pha
            res_pha.append(pha_channel)
            
            pha_channel = self.LRelu(self.bn_up5(self.pha_conv_up5(pha_channel)))
            mag_channel = self.LRelu(self.bn_up5(self.mag_conv_up5(mag_channel)))
            dct_channel = self.LRelu(self.bn5(self.dct_conv5(dct_channel)))
            dct_channel = self.LRelu(self.bn_up5(self.dct_conv_up5(dct_channel)))
            spat = torch.fft.irfft2(mag_channel * torch.exp(1j * pha_channel))
            # to_dct = self.freq_to_dct5(spat)
            # to_freq = self.dct_to_freq5(dct_channel)
            # to_mag, to_pha = self.decompose(torch_dct.idct_2d(to_freq))
            # dct_channel = dct_channel + to_dct
            # mag_channel = mag_channel + to_mag
            # pha_channel = pha_channel + to_pha
            
            return spat, dct_channel, res_dct, res_mag, res_pha
    
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.LRelu = nn.LeakyReLU(0.2, inplace=True)
            self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.mag_deconv1 = nn.Conv2d(512, 256, 1, stride=1, padding=0)
            self.pha_deconv1 = nn.Conv2d(512, 256, 1, stride=1, padding=0)
            self.dct_deconv1 = nn.Conv2d(512, 256, 3, stride=1, padding=1)
            self.bn_de1 = nn.BatchNorm2d(256)
            
            self.mag_deconv2 = nn.Conv2d(256, 128, 1, stride=1, padding=0)
            self.pha_deconv2 = nn.Conv2d(256, 128, 1, stride=1, padding=0)
            self.dct_deconv2 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
            self.bn_de2 = nn.BatchNorm2d(128)
            
            self.mag_deconv3 = nn.Conv2d(128, 64, 1, stride=1, padding=0)
            self.pha_deconv3 = nn.Conv2d(128, 64, 1, stride=1, padding=0)
            self.dct_deconv3 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
            self.bn_de3 = nn.BatchNorm2d(64)
            
            self.mag_deconv4 = nn.Conv2d(64, 32, 1, stride=1, padding=0)
            self.pha_deconv4 = nn.Conv2d(64, 32, 1, stride=1, padding=0)
            self.dct_deconv4 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
            self.bn_de4 = nn.BatchNorm2d(32)
            
            self.mag_deconv5 = nn.Conv2d(32, 3, 1, stride=1, padding=0)
            self.pha_deconv5 = nn.Conv2d(32, 3, 1, stride=1, padding=0)
            self.dct_deconv5 = nn.Conv2d(32, 3, 3, stride=1, padding=1)
            self.bn_de5 = nn.BatchNorm2d(3)
            
        def decompose(self, x):
            freq = torch.fft.rfft2(x)
            mag = torch.abs(freq)
            pha = torch.angle(freq)
            return mag, pha    
        
        def forward(self, spat, dct, res_dct, res_mag, res_pha):
            spat = self.up1(spat)
            mag, pha = self.decompose(spat)
            dct = self.up1(dct)
            pha_channel = self.LRelu(self.bn_de1(self.pha_deconv1(pha))) #pha[32, 512, 32, 17]
            mag_channel = self.LRelu(self.bn_de1(self.mag_deconv1(mag)))
            dct_channel = self.LRelu(self.bn_de1(self.dct_deconv1(dct)))
            
            spat = torch.fft.irfft2(mag_channel * torch.exp(1j * pha_channel))
            spat = self.up1(spat)
            dct_channel = self.up1(dct_channel)
            mag_channel, pha_channel = self.decompose(spat)
            pha_channel = self.LRelu(self.bn_de2(self.pha_deconv2(pha_channel))) #pha[32, 512, 128, 68]
            mag_channel = self.LRelu(self.bn_de2(self.mag_deconv2(mag_channel)))
            dct_channel = self.LRelu(self.bn_de2(self.dct_deconv2(dct_channel)))
            # spat = torch.fft.irfft2(mag_channel * torch.exp(1j * pha_channel))
            # spat = self.LRelu(self.bn_de2(self.spat_deconv2(spat)))
            
            
            spat = torch.fft.irfft2(mag_channel * torch.exp(1j * pha_channel))
            spat = self.up1(spat)
            dct_channel = self.up1(dct_channel)
            mag_channel, pha_channel = self.decompose(spat)
            pha_channel = self.LRelu(self.bn_de3(self.pha_deconv3(pha_channel)))
            mag_channel = self.LRelu(self.bn_de3(self.mag_deconv3(mag_channel)))
            dct_channel = self.LRelu(self.bn_de3(self.dct_deconv3(dct_channel)))
            # spat = torch.fft.irfft2(mag_channel * torch.exp(1j * pha_channel))
            # spat = self.LRelu(self.bn_de3(self.spat_deconv3(spat)))
            
            
            spat = torch.fft.irfft2(mag_channel * torch.exp(1j * pha_channel))
            spat = self.up1(spat)
            dct_channel = self.up1(dct_channel)
            mag_channel, pha_channel = self.decompose(spat)
            pha_channel = self.LRelu(self.bn_de4(self.pha_deconv4(pha_channel)))
            mag_channel = self.LRelu(self.bn_de4(self.mag_deconv4(mag_channel)))
            dct_channel = self.LRelu(self.bn_de4(self.dct_deconv4(dct_channel)))
            # spat = torch.fft.irfft2(mag_channel * torch.exp(1j * pha_channel))
            # spat = self.LRelu(self.bn_de4(self.spat_deconv4(spat)))
            
            spat = torch.fft.irfft2(mag_channel * torch.exp(1j * pha_channel))
            spat = self.up1(spat)
            dct_channel = self.up1(dct_channel)
            mag_channel, pha_channel = self.decompose(spat)
            pha_channel = self.LRelu(self.bn_de5(self.pha_deconv5(pha_channel)))
            mag_channel = self.LRelu(self.bn_de5(self.mag_deconv5(mag_channel)))
            dct_channel = self.LRelu(self.bn_de5(self.dct_deconv5(dct_channel)))
            spat = torch.fft.irfft2(mag_channel * torch.exp(1j * pha_channel))
            # spat = self.LRelu(self.bn_de5(self.spat_deconv5(spat)))
            
            return spat, dct_channel
            
    def __init__(self, mem_dim=128,shrink_thres=0.0025,fea_dim=512):
        super().__init__()
        self.mem_rep = MemModule(
            mem_dim=mem_dim,
            fea_dim=fea_dim,
            shrink_thres=shrink_thres
        )
        self.mem_rep2 = MemModule(
            mem_dim=mem_dim,
            fea_dim=fea_dim,
            shrink_thres=shrink_thres
        )
        self.decoder = self.Decoder()
        self.encoder = self.Encoder()
    def preprocess(self, x):
        freq = torch.fft.rfft2(x)
        mag = torch.abs(freq)
        pha = torch.angle(freq)
        return mag, pha
    
    def forward(self, x):
        mag, pha = self.preprocess(x)
        dct = torch_dct.dct_2d(x)
        spat_cha, dct_cha, res_dct, res_mag, res_pha = self.encoder(mag, pha, dct) #32, 512, 16, 16
        mem1 = self.mem_rep(spat_cha)
        mem2 = self.mem_rep2(dct_cha)
        b,c,w,h = mem1['output'].shape #32, 512, 16, 16
        spat, dct_out = self.decoder(mem1['output'], mem2['output'], res_dct, res_mag, res_pha)
        mag_out, pha_out = self.preprocess(spat)
        return spat_cha, dct_cha, mem1['output'], mem2['output'], spat, dct_out

class MemAutoEncoderOnlyDCT(nn.Module):
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.LRelu = nn.LeakyReLU(0.2, inplace=True)
            self.dct_conv_up1 = nn.Conv2d(3, 32, (1,1), stride=1, padding=0)
            self.dct_conv_1 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
            self.bn1 = nn.BatchNorm2d(3)
            self.bn_up1 = nn.BatchNorm2d(32)
            
            self.dct_conv_up2 = nn.Conv2d(32, 64, (1,1), stride=1, padding=0)
            self.dct_conv_2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.bn_up2 = nn.BatchNorm2d(64)
            
            self.dct_conv_up3 = nn.Conv2d(64, 128, (1,1), stride=1, padding=0)
            self.dct_conv_3 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
            self.bn3 = nn.BatchNorm2d(64)
            self.bn_up3 = nn.BatchNorm2d(128)
            
            self.dct_conv_up4 = nn.Conv2d(128, 256, (1,1), stride=1, padding=0)
            self.dct_conv_4 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
            self.bn4 = nn.BatchNorm2d(128)
            self.bn_up4 = nn.BatchNorm2d(256)
            
            self.dct_conv_up5 = nn.Conv2d(256, 512, (1,1), stride=1, padding=0)
            self.dct_conv_5 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
            self.bn5 = nn.BatchNorm2d(256)
            self.bn_up5 = nn.BatchNorm2d(512)

        def decompose(self, x):
            freq = torch.fft.rfft2(x)
            mag = torch.abs(freq)
            pha = torch.angle(freq)
            return mag, pha
        
        def forward(self, dct):
            dct_channel = self.LRelu(self.bn_up1(self.dct_conv_up1(dct)))
            dct_channel = self.LRelu(self.bn_up1(self.dct_conv_1(dct_channel)))
            
            dct_channel = self.LRelu(self.bn_up2(self.dct_conv_up2(dct_channel)))
            dct_channel = self.LRelu(self.bn_up2(self.dct_conv_2(dct_channel)))
            
            dct_channel = self.LRelu(self.bn_up3(self.dct_conv_up3(dct_channel)))
            dct_channel = self.LRelu(self.bn_up3(self.dct_conv_3(dct_channel)))
            
            dct_channel = self.LRelu(self.bn_up4(self.dct_conv_up4(dct_channel)))
            dct_channel = self.LRelu(self.bn_up4(self.dct_conv_4(dct_channel)))
            
            dct_channel = self.LRelu(self.bn_up5(self.dct_conv_up5(dct_channel)))
            dct_channel = self.LRelu(self.bn_up5(self.dct_conv_5(dct_channel)))
            
            return dct_channel
    
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.LRelu = nn.LeakyReLU(0.2, inplace=True)
            self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.dct_deconv_up1 = nn.Conv2d(512, 256, 1, stride=1, padding=0)
            self.dct_deconv1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
            self.bn_de1 = nn.BatchNorm2d(256)
            
            self.dct_deconv_up2 = nn.Conv2d(256, 128, 1, stride=1, padding=0)
            self.dct_deconv2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
            self.bn_de2 = nn.BatchNorm2d(128)
            
            self.dct_deconv_up3 = nn.Conv2d(128, 64, 1, stride=1, padding=0)
            self.dct_deconv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
            self.bn_de3 = nn.BatchNorm2d(64)
            
            self.dct_deconv_up4 = nn.Conv2d(64, 32, 1, stride=1, padding=0)
            self.dct_deconv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
            self.bn_de4 = nn.BatchNorm2d(32)
            
            self.dct_deconv_up5 = nn.Conv2d(32, 3, 1, stride=1, padding=0)
            self.dct_deconv5 = nn.Conv2d(3, 3, 3, stride=1, padding=1)
            self.bn_de5 = nn.BatchNorm2d(3)
            
        def forward(self, dct):
            dct_channel = self.up1(dct)
            dct_channel = self.LRelu(self.bn_de1(self.dct_deconv_up1(dct_channel)))
            dct_channel = self.LRelu(self.bn_de1(self.dct_deconv1(dct_channel)))
            
            dct_channel = self.up1(dct_channel)
            dct_channel = self.LRelu(self.bn_de2(self.dct_deconv_up2(dct_channel)))
            dct_channel = self.LRelu(self.bn_de2(self.dct_deconv2(dct_channel)))
            
            dct_channel = self.up1(dct_channel)
            dct_channel = self.LRelu(self.bn_de3(self.dct_deconv_up3(dct_channel)))
            dct_channel = self.LRelu(self.bn_de3(self.dct_deconv3(dct_channel)))
            
            dct_channel = self.up1(dct_channel)
            dct_channel = self.LRelu(self.bn_de4(self.dct_deconv_up4(dct_channel)))
            dct_channel = self.LRelu(self.bn_de4(self.dct_deconv4(dct_channel)))
            
            dct_channel = self.up1(dct_channel)
            dct_channel = self.LRelu(self.bn_de5(self.dct_deconv_up5(dct_channel)))
            dct_channel = self.LRelu(self.bn_de5(self.dct_deconv5(dct_channel)))
            
            return dct_channel
            
    def __init__(self, mem_dim=128,shrink_thres=0.0025,fea_dim=512):
        super().__init__()
        self.mem_rep = MemModule(
            mem_dim=mem_dim,
            fea_dim=fea_dim,
            shrink_thres=shrink_thres
        )
        self.decoder = self.Decoder()
        self.encoder = self.Encoder()
    
    def forward(self, x):
        dct = torch_dct.dct_2d(x)
        encoder_out = self.encoder(dct) #32, 512, 16, 16
        mem = self.mem_rep(encoder_out)
        b,c,w,h = mem['output'].shape
        dct_out = self.decoder(mem['output'])
        return encoder_out, mem['output'], dct_out

class Mem_weight(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        
        return x
    
    
class AutoEncoderFCMemMoreFeatureTwoEncoderWithWaveletChangeMem(nn.Module):
    def __init__(self,mem_dim=64,shrink_thres=0.0025,fea_dim=512*2*2):
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
        
        #memory
        self.mem_dim = mem_dim
        self.mem_rep = MemModule_ForLoss(
            mem_dim=mem_dim,
            fea_dim=fea_dim,
            shrink_thres=shrink_thres)

        # decoder 
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv1 = nn.Conv2d(512,256,3,stride=1,padding=1)
        self.debn1 = nn.BatchNorm2d(256)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv2 = nn.Conv2d(256,128,3,stride=1,padding=1)
        self.debn2 = nn.BatchNorm2d(128)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv3 = nn.Conv2d(128,64,3,stride=1,padding=1)
        self.debn3 = nn.BatchNorm2d(64)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv4 = nn.Conv2d(64,32,3,stride=1,padding=1)
        self.debn4 = nn.BatchNorm2d(32)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv5 = nn.Conv2d(32,3,3,stride=1,padding=1)
        self.final = nn.Tanh()

        #wavelet
        self.waveletDecompose = wavelet.WavePool_Two(in_channels=3).cuda()
        self.waveletCompose = wavelet.WaveUnpool(512, 'sum')
        self.waveletResBlock1 = WaveletBlock(3, 32)
        self.waveletResBlock2 = WaveletBlock(32, 64)
        self.waveletResBlock3 = WaveletBlock(64, 128)
        self.waveletResBlock4 = WaveletBlock(128, 256)
        self.waveletResBlock5 = WaveletBlock(256, 512)
        
        self.waveletResBlockDown1 = WaveletBlock(512, 256, upscale=True)
        self.waveletResBlockDown2 = WaveletBlock(256, 128, upscale=True)
        self.waveletResBlockDown3 = WaveletBlock(128, 64, upscale=True)
        self.waveletResBlockDown4 = WaveletBlock(64, 32, upscale=True)
        self.waveletResBlockDown5 = WaveletBlock(32, 3, upscale=True)
        
        #conv
        self.conv_connection_1 = nn.Conv2d(2, 1, 1,stride=1, padding=0)
        self.sig = nn.Sigmoid()
    def freq_connection(self, freq_h, freq_l):
        x = torch.cat((freq_h,freq_l), dim=1)
        #h, w, c = x.shape
        x_max = torch.max(x, dim = 1).values.unsqueeze(1)
        x_sum = torch.sum(x, dim = 1).unsqueeze(1)
        x = torch.cat((x_max, x_sum), dim=1)
        x = self.conv_connection_1(x)
        x = self.sig(x)
        return x * freq_h
    
    def encoder_H(self, freq_h, feat_1, feat_2, feat_3, feat_4):
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
        return x, feat_1, feat_2, feat_3, feat_4
    
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
        x = self.deconv5(x)
        x = self.final(x)
        return x
    
    def forward(self,x):
        LL, LH, HL, HH = self.waveletDecompose(x)
        freq_h = torch.cat((LH,HL,HH),dim=1)
        freq_l = LL
        x_LL, feat_1, feat_2, feat_3, feat_4 = self.encoder_L(freq_l)
        x_H = self.encoder_H(freq_h, feat_1, feat_2, feat_3, feat_4)
        x_LH = x_H[:, 0:512, :, :]
        x_HL = x_H[:, 512:1024, :, :]
        x_HH = x_H[:, 1024:1536, :, :]
        encoder_com = self.waveletCompose(x_LL, x_LH, x_HL, x_HH)
        b,c,h,w = encoder_com.shape
        encoder_com = encoder_com.view(b, -1)
        res_mem = self.mem_rep(encoder_com)
        f = res_mem['output']
        f = f.view(b,c,h,w)
        decoder = self.decoder(f)
        return decoder, encoder_com.view(b,c,w,h), f, res_mem['att'], res_mem['membank']

#AutoEncoderTwoEncoderWithWaveletChangeMemConnection
class mymodel(nn.Module):
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
        
        #memory
        self.mem_dim = mem_dim
        self.mem_rep = MemModule_ForLoss(
            mem_dim=mem_dim,
            fea_dim=fea_dim,
            tempreture = tempreture,
            shrink_thres=shrink_thres)

        # decoder 
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv1 = nn.Conv2d(512,256,3,stride=1,padding=1)
        self.debn1 = nn.BatchNorm2d(256)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv2 = nn.Conv2d(256,128,3,stride=1,padding=1)
        self.debn2 = nn.BatchNorm2d(128)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv3 = nn.Conv2d(128,64,3,stride=1,padding=1)
        self.debn3 = nn.BatchNorm2d(64)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv4 = nn.Conv2d(64,32,3,stride=1,padding=1)
        self.debn4 = nn.BatchNorm2d(32)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv5 = nn.Conv2d(32,3,3,stride=1,padding=1)
        self.final = nn.Sigmoid()

        #wavelet
        self.waveletDecompose = wavelet.WavePool_Two(in_channels=3).cuda()
        self.waveletCompose = wavelet.WaveUnpool(512, 'sum')
        self.waveletResBlock1 = WaveletBlock(3, 32)
        self.waveletResBlock2 = WaveletBlock(32, 64)
        self.waveletResBlock3 = WaveletBlock(64, 128)
        self.waveletResBlock4 = WaveletBlock(128, 256)
        self.waveletResBlock5 = WaveletBlock(256, 512)
        
        self.waveletResBlockDown1 = WaveletBlock(512, 256, upscale=True)
        self.waveletResBlockDown2 = WaveletBlock(256, 128, upscale=True)
        self.waveletResBlockDown3 = WaveletBlock(128, 64, upscale=True)
        self.waveletResBlockDown4 = WaveletBlock(64, 32, upscale=True)
        self.waveletResBlockDown5 = WaveletBlock(32, 3, upscale=True)
        
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
    
    def encoder_H(self, freq_h, feat_1, feat_2, feat_3, feat_4):
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
        return x, feat_1, feat_2, feat_3, feat_4
    
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
        x = self.deconv5(x)
        x = self.final(x)
        return x
    
    def forward(self,x):
        LL, LH, HL, HH = self.waveletDecompose(x)
        freq_h = torch.cat((LH,HL,HH),dim=1)
        freq_l = LL
        x_LL, feat_1, feat_2, feat_3, feat_4 = self.encoder_L(freq_l)
        x_H = self.encoder_H(freq_h, feat_1, feat_2, feat_3, feat_4)
        x_LH = x_H[:, 0:512, :, :]
        x_HL = x_H[:, 512:1024, :, :]
        x_HH = x_H[:, 1024:1536, :, :]
        encoder_com = self.waveletCompose(x_LL, x_LH, x_HL, x_HH)
        b,c,h,w = encoder_com.shape
        encoder_com = encoder_com.view(b, -1)
        res_mem = self.mem_rep(encoder_com)
        f = res_mem['output']
        f = f.view(b,c,h,w)
        decoder = self.decoder(f)
        return decoder, encoder_com.view(b,c,w,h), f, res_mem['att'], res_mem['membank']
    
    
class AutoEncoderTwoEncoderWithWaveletChangeMemConnection_notemp(nn.Module):
    def __init__(self,mem_dim=64,fea_dim=512*2*2, shrink_thres=0.0025):
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
        
        #memory
        self.mem_dim = mem_dim
        self.mem_rep = MemModule_ForLoss_notemp(
            mem_dim=mem_dim,
            fea_dim=fea_dim,
            shrink_thres=shrink_thres)

        # decoder 
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv1 = nn.Conv2d(512,256,3,stride=1,padding=1)
        self.debn1 = nn.BatchNorm2d(256)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv2 = nn.Conv2d(256,128,3,stride=1,padding=1)
        self.debn2 = nn.BatchNorm2d(128)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv3 = nn.Conv2d(128,64,3,stride=1,padding=1)
        self.debn3 = nn.BatchNorm2d(64)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv4 = nn.Conv2d(64,32,3,stride=1,padding=1)
        self.debn4 = nn.BatchNorm2d(32)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv5 = nn.Conv2d(32,3,3,stride=1,padding=1)
        self.final = nn.Tanh()

        #wavelet
        self.waveletDecompose = wavelet.WavePool_Two(in_channels=3).cuda()
        self.waveletCompose = wavelet.WaveUnpool(512, 'sum')
        self.waveletResBlock1 = WaveletBlock(3, 32)
        self.waveletResBlock2 = WaveletBlock(32, 64)
        self.waveletResBlock3 = WaveletBlock(64, 128)
        self.waveletResBlock4 = WaveletBlock(128, 256)
        self.waveletResBlock5 = WaveletBlock(256, 512)
        
        self.waveletResBlockDown1 = WaveletBlock(512, 256, upscale=True)
        self.waveletResBlockDown2 = WaveletBlock(256, 128, upscale=True)
        self.waveletResBlockDown3 = WaveletBlock(128, 64, upscale=True)
        self.waveletResBlockDown4 = WaveletBlock(64, 32, upscale=True)
        self.waveletResBlockDown5 = WaveletBlock(32, 3, upscale=True)
        
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
    
    def encoder_H(self, freq_h, feat_1, feat_2, feat_3, feat_4):
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
        return x, feat_1, feat_2, feat_3, feat_4
    
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
        x = self.deconv5(x)
        x = self.relu(x)
        return x
    
    def forward(self,x):
        LL, LH, HL, HH = self.waveletDecompose(x)
        freq_h = torch.cat((LH,HL,HH),dim=1)
        freq_l = LL
        x_LL, feat_1, feat_2, feat_3, feat_4 = self.encoder_L(freq_l)
        x_H = self.encoder_H(freq_h, feat_1, feat_2, feat_3, feat_4)
        x_LH = x_H[:, 0:512, :, :]
        x_HL = x_H[:, 512:1024, :, :]
        x_HH = x_H[:, 1024:1536, :, :]
        encoder_com = self.waveletCompose(x_LL, x_LH, x_HL, x_HH)
        b,c,h,w = encoder_com.shape
        encoder_com = encoder_com.view(b, -1)
        res_mem = self.mem_rep(encoder_com)
        f = res_mem['output']
        f = f.view(b,c,h,w)
        decoder = self.decoder(f)
        return decoder, encoder_com.view(b,c,w,h), f, res_mem['att'], res_mem['membank']
    
    
class zuo_PPAD(nn.Module):
    def __init__(self,mem_dim=64,fea_dim=512*2*2,tempreture=0.1,shrink_thres=0.0025):
        super().__init__()
        # encoder
        self.conv1 = nn.Conv2d(3,8,(3,3),stride=2,padding=1)
        self.bn1   = nn.BatchNorm2d(8)
        self.relu  = nn.LeakyReLU(0.2,inplace=True)
        self.conv2 = nn.Conv2d(8,16,3,stride=2,padding=1)
        self.bn2   = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16,32,3,stride=2,padding=1)
        self.bn3   = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(64,64,3,stride=2,padding=1)
        self.bn4   = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64,64,3,stride=2,padding=1)
        self.bn5   = nn.BatchNorm2d(64)
        
        #memory
        self.mem_dim = mem_dim
        self.mem_rep = MemModule_ForLoss(
            mem_dim=mem_dim,
            fea_dim=fea_dim,
            tempreture = tempreture,
            shrink_thres=shrink_thres)

        # decoder 
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv1 = nn.Conv2d(64,64,3,stride=1,padding=1)
        self.debn1 = nn.BatchNorm2d(64)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv2 = nn.Conv2d(64,32,3,stride=1,padding=1)
        self.debn2 = nn.BatchNorm2d(32)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv3 = nn.Conv2d(32,16,3,stride=1,padding=1)
        self.debn3 = nn.BatchNorm2d(16)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv4 = nn.Conv2d(16,8,3,stride=1,padding=1)
        self.debn4 = nn.BatchNorm2d(8)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv5 = nn.Conv2d(8,3,3,stride=1,padding=1)
        self.final = nn.Tanh()
        #conv
        #self.conv_connection_1 = nn.Conv2d(2, 1, 1,stride=1, padding=0)
        self.sig = nn.Sigmoid()
    
    def encoder(self, x):
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        x = self.conv3(x)
        x = self.relu(self.bn3(x))
        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        x = self.conv5(x)
        x = self.relu(self.bn5(x))
        return x
    
    
    def decoder(self, x):
        x = self.up1(x)
        x = self.relu(self.debn1(self.deconv1(x)))
        x = self.up2(x)
        x = self.relu(self.debn2(self.deconv2(x)))
        x = self.up3(x)
        x = self.relu(self.debn3(self.deconv3(x)))
        x = self.up4(x)
        x = self.relu(self.debn4(self.deconv4(x)))
        x = self.up5(x)
        x = self.deconv5(x)
        x = self.final(x)
        return x
    
    def forward(self,x):
        encoder = self.encoder(x)
        res_mem = self.mem_rep(encoder)
        f = res_mem['output']
        att = res_mem['att']
        memory = res_mem['memory']
        f = encoder * (1 - self.residual_rate) + f * self.residual_rate
        decoder = self.decoder(f)
        return decoder,f,att,memory
    
    
#消融实验

# baseline
class mymodel_baseline(nn.Module):
    def __init__(self):
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
        
        #memory
        # self.mem_dim = mem_dim
        # self.mem_rep = MemModule(
        #     mem_dim=mem_dim,
        #     fea_dim=fea_dim,
        #     tempreture = tempreture,
        #     shrink_thres=shrink_thres)

        # decoder 
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv1 = nn.Conv2d(512,256,3,stride=1,padding=1)
        self.debn1 = nn.BatchNorm2d(256)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv2 = nn.Conv2d(256,128,3,stride=1,padding=1)
        self.debn2 = nn.BatchNorm2d(128)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv3 = nn.Conv2d(128,64,3,stride=1,padding=1)
        self.debn3 = nn.BatchNorm2d(64)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv4 = nn.Conv2d(64,32,3,stride=1,padding=1)
        self.debn4 = nn.BatchNorm2d(32)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv5 = nn.Conv2d(32,3,3,stride=1,padding=1)
        self.final = nn.Tanh()

        #wavelet
        self.waveletDecompose = wavelet.WavePool_Two(in_channels=3).cuda()
        self.waveletCompose = wavelet.WaveUnpool(512, 'sum')
        
    def encoder_H(self, freq_h):
        x = freq_h
        x = self.conv_H1(x)
        x = self.relu(self.bn_H1(x))
        x = self.conv_H2(x)
        x = self.relu(self.bn_H2(x))
        x = self.conv_H3(x)
        x = self.relu(self.bn_H3(x))
        x = self.conv_H4(x)
        x = self.relu(self.bn_H4(x))
        x = self.conv_H5(x)
        x = self.relu(self.bn_H5(x))
        return x
    
    def encoder_L(self, freq_l):
        x = freq_l
        x = self.conv1(x) #->[32, 32, 256, 256]
        x = self.relu(self.bn1(x))
        x = self.conv2(x) #->[32, 64, 128, 128]
        x = self.relu(self.bn2(x))
        x = self.conv3(x) #->[32, 128, 64, 64]
        x = self.relu(self.bn3(x))
        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        x = self.conv5(x)
        x = self.relu(self.bn5(x))
        return x
    
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
        x = self.deconv5(x)
        x = self.relu(x)
        return x
    
    def forward(self,x):
        LL, LH, HL, HH = self.waveletDecompose(x)
        freq_h = torch.cat((LH,HL,HH), dim=1)
        freq_l = LL
        x_LL = self.encoder_L(freq_l)
        x_H = self.encoder_H(freq_h)
        x_LH = x_H[:, 0:512, :, :]
        x_HL = x_H[:, 512:1024, :, :]
        x_HH = x_H[:, 1024:1536, :, :]
        encoder_com = self.waveletCompose(x_LL, x_LH, x_HL, x_HH)
        decoder = self.decoder(encoder_com)
        return decoder, encoder_com

#frequency decoupling + mymemory
class mymodel_freqmem(nn.Module):
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
        
        #memory
        self.mem_dim = mem_dim
        self.mem_rep = MemModule_ForLoss(
            mem_dim=mem_dim,
            fea_dim=fea_dim,
            tempreture = tempreture,
            shrink_thres=shrink_thres)

        # decoder 
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv1 = nn.Conv2d(512,256,3,stride=1,padding=1)
        self.debn1 = nn.BatchNorm2d(256)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv2 = nn.Conv2d(256,128,3,stride=1,padding=1)
        self.debn2 = nn.BatchNorm2d(128)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv3 = nn.Conv2d(128,64,3,stride=1,padding=1)
        self.debn3 = nn.BatchNorm2d(64)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv4 = nn.Conv2d(64,32,3,stride=1,padding=1)
        self.debn4 = nn.BatchNorm2d(32)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv5 = nn.Conv2d(32,3,3,stride=1,padding=1)
        self.final = nn.Tanh()


        #wavelet
        self.waveletDecompose = wavelet.WavePool_Two(in_channels=3).cuda()
        self.waveletCompose = wavelet.WaveUnpool(512, 'sum')
        self.waveletResBlock1 = WaveletBlock(3, 32)
        self.waveletResBlock2 = WaveletBlock(32, 64)
        self.waveletResBlock3 = WaveletBlock(64, 128)
        self.waveletResBlock4 = WaveletBlock(128, 256)
        self.waveletResBlock5 = WaveletBlock(256, 512)
        
        self.waveletResBlockDown1 = WaveletBlock(512, 256, upscale=True)
        self.waveletResBlockDown2 = WaveletBlock(256, 128, upscale=True)
        self.waveletResBlockDown3 = WaveletBlock(128, 64, upscale=True)
        self.waveletResBlockDown4 = WaveletBlock(64, 32, upscale=True)
        self.waveletResBlockDown5 = WaveletBlock(32, 3, upscale=True)
        #conv
        #self.conv_connection_1 = nn.Conv2d(2, 1, 1,stride=1, padding=0)
        self.sig = nn.Sigmoid()
    
    def encoder_H(self, freq_h):
        x = freq_h
        x = self.conv_H1(x)
        x = self.relu(self.bn_H1(x))
        x = self.conv_H2(x)
        x = self.relu(self.bn_H2(x))
        x = self.conv_H3(x)
        x = self.relu(self.bn_H3(x))
        x = self.conv_H4(x)
        x = self.relu(self.bn_H4(x))
        x = self.conv_H5(x)
        x = self.relu(self.bn_H5(x))
        return x
    
    def encoder_L(self, freq_l):
        x = freq_l
        x = self.conv1(x) #->[32, 32, 256, 256]
        x = self.relu(self.bn1(x))
        x = self.conv2(x) #->[32, 64, 128, 128]
        x = self.relu(self.bn2(x))
        x = self.conv3(x) #->[32, 128, 64, 64]
        x = self.relu(self.bn3(x))
        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        x = self.conv5(x)
        x = self.relu(self.bn5(x))
        return x
    
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
        x = self.deconv5(x)
        x = self.relu(x)
        return x
    
    def forward(self,x):
        LL, LH, HL, HH = self.waveletDecompose(x)
        freq_h = torch.cat((LH,HL,HH),dim = 1)
        freq_l = LL
        x_LL = self.encoder_L(freq_l)
        x_H = self.encoder_H(freq_h)
        x_LH = x_H[:, 0:512, :, :]
        x_HL = x_H[:, 512:1024, :, :]
        x_HH = x_H[:, 1024:1536, :, :]
        encoder_com = self.waveletCompose(x_LL, x_LH, x_HL, x_HH)
        b,c,h,w = encoder_com.shape
        encoder_com = encoder_com.view(b, -1)
        res_mem = self.mem_rep(encoder_com)
        f = res_mem['output']
        f = f.view(b,c,h,w)
        decoder = self.decoder(f)
        return decoder, encoder_com.view(b,c,w,h), f, res_mem['att'], res_mem['membank']
    
# no Loss have FCC
class mymodel_noLoss(nn.Module):
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
        
        #memory
        self.mem_dim = mem_dim
        self.mem_rep = MemModule(
            mem_dim=mem_dim,
            fea_dim=fea_dim,
            tempreture = tempreture,
            shrink_thres=shrink_thres)

        # decoder 
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv1 = nn.Conv2d(512,256,3,stride=1,padding=1)
        self.debn1 = nn.BatchNorm2d(256)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv2 = nn.Conv2d(256,128,3,stride=1,padding=1)
        self.debn2 = nn.BatchNorm2d(128)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv3 = nn.Conv2d(128,64,3,stride=1,padding=1)
        self.debn3 = nn.BatchNorm2d(64)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv4 = nn.Conv2d(64,32,3,stride=1,padding=1)
        self.debn4 = nn.BatchNorm2d(32)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv5 = nn.Conv2d(32,3,3,stride=1,padding=1)
        self.final = nn.Tanh()

        #wavelet
        self.waveletDecompose = wavelet.WavePool_Two(in_channels=3).cuda()
        self.waveletCompose = wavelet.WaveUnpool(512, 'sum')
        self.waveletResBlock1 = WaveletBlock(3, 32)
        self.waveletResBlock2 = WaveletBlock(32, 64)
        self.waveletResBlock3 = WaveletBlock(64, 128)
        self.waveletResBlock4 = WaveletBlock(128, 256)
        self.waveletResBlock5 = WaveletBlock(256, 512)
        
        self.waveletResBlockDown1 = WaveletBlock(512, 256, upscale=True)
        self.waveletResBlockDown2 = WaveletBlock(256, 128, upscale=True)
        self.waveletResBlockDown3 = WaveletBlock(128, 64, upscale=True)
        self.waveletResBlockDown4 = WaveletBlock(64, 32, upscale=True)
        self.waveletResBlockDown5 = WaveletBlock(32, 3, upscale=True)
        
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
    
    def encoder_H(self, freq_h, feat_1, feat_2, feat_3, feat_4):
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
        return x, feat_1, feat_2, feat_3, feat_4
    
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
        x = self.deconv5(x)
        x = self.relu(x)
        return x
    
    def forward(self,x):
        LL, LH, HL, HH = self.waveletDecompose(x)
        freq_h = torch.cat((LH,HL,HH),dim=1)
        freq_l = LL
        x_LL, feat_1, feat_2, feat_3, feat_4 = self.encoder_L(freq_l)
        x_H = self.encoder_H(freq_h, feat_1, feat_2, feat_3, feat_4)
        x_LH = x_H[:, 0:512, :, :]
        x_HL = x_H[:, 512:1024, :, :]
        x_HH = x_H[:, 1024:1536, :, :]
        encoder_com = self.waveletCompose(x_LL, x_LH, x_HL, x_HH)
        b,c,h,w = encoder_com.shape
        encoder_com = encoder_com.view(b, -1)
        res_mem = self.mem_rep(encoder_com)
        f = res_mem['output']
        f = f.view(b,c,h,w)
        decoder = self.decoder(f)
        return decoder, encoder_com.view(b,c,w,h), f, res_mem['att'], res_mem['membank']
    
# no FCC have Loss
class mymodel_noLoss(nn.Module):
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
        
        #memory
        self.mem_dim = mem_dim
        self.mem_rep = MemModule_ForLoss(
            mem_dim=mem_dim,
            fea_dim=fea_dim,
            tempreture = tempreture,
            shrink_thres=shrink_thres)

        # decoder 
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv1 = nn.Conv2d(512,256,3,stride=1,padding=1)
        self.debn1 = nn.BatchNorm2d(256)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv2 = nn.Conv2d(256,128,3,stride=1,padding=1)
        self.debn2 = nn.BatchNorm2d(128)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv3 = nn.Conv2d(128,64,3,stride=1,padding=1)
        self.debn3 = nn.BatchNorm2d(64)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv4 = nn.Conv2d(64,32,3,stride=1,padding=1)
        self.debn4 = nn.BatchNorm2d(32)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv5 = nn.Conv2d(32,3,3,stride=1,padding=1)
        self.final = nn.Tanh()

        #wavelet
        self.waveletDecompose = wavelet.WavePool_Two(in_channels=3).cuda()
        self.waveletCompose = wavelet.WaveUnpool(512, 'sum')
        self.waveletResBlock1 = WaveletBlock(3, 32)
        self.waveletResBlock2 = WaveletBlock(32, 64)
        self.waveletResBlock3 = WaveletBlock(64, 128)
        self.waveletResBlock4 = WaveletBlock(128, 256)
        self.waveletResBlock5 = WaveletBlock(256, 512)
        
        self.waveletResBlockDown1 = WaveletBlock(512, 256, upscale=True)
        self.waveletResBlockDown2 = WaveletBlock(256, 128, upscale=True)
        self.waveletResBlockDown3 = WaveletBlock(128, 64, upscale=True)
        self.waveletResBlockDown4 = WaveletBlock(64, 32, upscale=True)
        self.waveletResBlockDown5 = WaveletBlock(32, 3, upscale=True)
        
        #conv
        #self.conv_connection_1 = nn.Conv2d(2, 1, 1,stride=1, padding=0)
        self.sig = nn.Sigmoid()
    
    def encoder_H(self, freq_h):
        x = freq_h
        x = self.conv_H1(x)
        x = self.relu(self.bn_H1(x))
        x = self.conv_H2(x)
        x = self.relu(self.bn_H2(x))
        x = self.conv_H3(x)
        x = self.relu(self.bn_H3(x))
        x = self.conv_H4(x)
        x = self.relu(self.bn_H4(x))
        x = self.conv_H5(x)
        x = self.relu(self.bn_H5(x))
        return x
    
    def encoder_L(self, freq_l):
        x = freq_l
        x = self.conv1(x) #->[32, 32, 256, 256]
        x = self.relu(self.bn1(x))
        x = self.conv2(x) #->[32, 64, 128, 128]
        x = self.relu(self.bn2(x))
        x = self.conv3(x) #->[32, 128, 64, 64]
        x = self.relu(self.bn3(x))
        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        x = self.conv5(x)
        x = self.relu(self.bn5(x))
        return x
    
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
        x = self.deconv5(x)
        x = self.relu(x)
        return x
    
    def forward(self,x):
        LL, LH, HL, HH = self.waveletDecompose(x)
        freq_h = torch.cat((LH,HL,HH),dim=1)
        freq_l = LL
        x_LL, feat_1, feat_2, feat_3, feat_4 = self.encoder_L(freq_l)
        x_H = self.encoder_H(freq_h, feat_1, feat_2, feat_3, feat_4)
        x_LH = x_H[:, 0:512, :, :]
        x_HL = x_H[:, 512:1024, :, :]
        x_HH = x_H[:, 1024:1536, :, :]
        encoder_com = self.waveletCompose(x_LL, x_LH, x_HL, x_HH)
        b,c,h,w = encoder_com.shape
        encoder_com = encoder_com.view(b, -1)
        res_mem = self.mem_rep(encoder_com)
        f = res_mem['output']
        f = f.view(b,c,h,w)
        decoder = self.decoder(f)
        return decoder, encoder_com.view(b,c,w,h), f, res_mem['att'], res_mem['membank']
    
#不好实现，分高低频branch 来实验他们的作用，abandon
    
#ablation study
class mymodel_improvedMem(nn.Module):
    # fea_dim = 1024 * 8 * 8
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
        self.final = nn.Tanh()

        #wavelet
        self.waveletDecompose = wavelet.WavePool_Two(in_channels=3).cuda()
        self.waveletCompose = wavelet.WaveUnpool(512, 'sum')
        
        #conv
        #self.conv_connection_1 = nn.Conv2d(2, 1, 1,stride=1, padding=0)
        self.sig = nn.Sigmoid()
    
    def encoder(self, freq_l):
        x = freq_l
        x = self.conv1(x) #->[32, 32, 256, 256]
        x = self.relu(self.bn1(x))
        x = self.conv2(x) #->[32, 64, 128, 128]
        x = self.relu(self.bn2(x))
        x = self.conv3(x) #->[32, 128, 64, 64]
        x = self.relu(self.bn3(x))
        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        x = self.conv5(x)
        x = self.relu(self.bn5(x))
        x = self.conv6(x)
        x = self.relu(self.bn6(x))
        return x
    
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
        x = self.relu(x)
        return x
    
    def forward(self,x):
        encoder = self.encoder(x)
        b,c,h,w = encoder.shape
        encoder = encoder.view(b, -1)
        res_mem = self.mem_rep(encoder)
        f = res_mem['output']
        f = f.view(b,c,h,w)
        decoder = self.decoder(f)
        return decoder, encoder.view(b,c,w,h), f, res_mem['att'], res_mem['membank']

#ablation study
class mymodel_improvedMem_Freq(nn.Module):
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
        self.final = nn.Tanh()
        #wavelet
        self.waveletDecompose = wavelet.WavePool_Two(in_channels=3).cuda()
        self.waveletCompose = wavelet.WaveUnpool(1024, 'sum')
        #conv
        #self.conv_connection_1 = nn.Conv2d(2, 1, 1,stride=1, padding=0)
        self.sig = nn.Sigmoid()
    
    def encoder_H(self, freq_h):
        x = freq_h
        x = self.conv_H1(x)
        x = self.relu(self.bn_H1(x))
        x = self.conv_H2(x)
        x = self.relu(self.bn_H2(x))
        x = self.conv_H3(x)
        x = self.relu(self.bn_H3(x))
        x = self.conv_H4(x)
        x = self.relu(self.bn_H4(x))
        x = self.conv_H5(x)
        x = self.relu(self.bn_H5(x))
        x = self.conv_H6(x)
        x = self.relu(self.bn_H6(x))
        return x
    
    def encoder_L(self, freq_l):
        x = freq_l
        x = self.conv1(x) #->[32, 32, 256, 256]
        x = self.relu(self.bn1(x))
        x = self.conv2(x) #->[32, 64, 128, 128]
        x = self.relu(self.bn2(x))
        x = self.conv3(x) #->[32, 128, 64, 64]
        x = self.relu(self.bn3(x))
        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        x = self.conv5(x)
        x = self.relu(self.bn5(x))
        x = self.conv6(x)
        x = self.relu(self.bn6(x))
        return x
    
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
        x = self.relu(x)
        return x
    
    def forward(self,x):
        LL, LH, HL, HH = self.waveletDecompose(x)
        freq_h = torch.cat((LH,HL,HH),dim=1)
        freq_l = LL
        x_LL = self.encoder_L(freq_l)
        x_H = self.encoder_H(freq_h)
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

#mymodel  deeper
class mymodel_deeper_freq(nn.Module):
    def __init__(self):
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
        self.final = nn.Tanh()

        #wavelet
        self.waveletDecompose = wavelet.WavePool_Two(in_channels=3).cuda()
        self.waveletCompose = wavelet.WaveUnpool(1024, 'sum')
        
        #conv
        #self.conv_connection_1 = nn.Conv2d(2, 1, 1,stride=1, padding=0)
        self.sig = nn.Sigmoid()
    
    def encoder_H(self, freq_h):
        x = freq_h
        x = self.conv_H1(x)
        x = self.relu(self.bn_H1(x))
        x = self.conv_H2(x)
        x = self.relu(self.bn_H2(x))
        x = self.conv_H3(x)
        x = self.relu(self.bn_H3(x))
        x = self.conv_H4(x)
        x = self.relu(self.bn_H4(x))
        x = self.conv_H5(x)
        x = self.relu(self.bn_H5(x))
        x = self.conv_H6(x)
        x = self.relu(self.bn_H6(x))
        return x
    
    def encoder_L(self, freq_l):
        x = freq_l
        x = self.conv1(x) #->[32, 32, 256, 256]
        x = self.relu(self.bn1(x))
        x = self.conv2(x) #->[32, 64, 128, 128]
        x = self.relu(self.bn2(x))
        x = self.conv3(x) #->[32, 128, 64, 64]
        x = self.relu(self.bn3(x))
        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        x = self.conv5(x)
        x = self.relu(self.bn5(x))
        x = self.conv6(x)
        x = self.relu(self.bn6(x))
        return x
    
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
        x = self.relu(x)
        return x
    
    def forward(self,x):
        LL, LH, HL, HH = self.waveletDecompose(x)
        freq_h = torch.cat((LH,HL,HH),dim=1)
        freq_l = LL
        x_LL = self.encoder_L(freq_l)
        x_H = self.encoder_H(freq_h)
        x_LH = x_H[:, 0:1024, :, :]
        x_HL = x_H[:, 1024:2048, :, :]
        x_HH = x_H[:, 2048:3072, :, :]
        encoder_com = self.waveletCompose(x_LL, x_LH, x_HL, x_HH)
        decoder = self.decoder(encoder_com)
        return decoder, encoder_com

#ablation study
class mymodel_deeper_freq_mem(nn.Module):
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
        self.final = nn.Tanh()

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
        x = self.relu(x)
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

# mymodel deeper
class mymodel_deeper(nn.Module):
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
    
#mymodel  deeper
class mymodel_deeper_freqency(nn.Module):
    def __init__(self):
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
        self.final = nn.Tanh()

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
        x = self.relu(x)
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
        decoder = self.decoder(encoder_com)
        return decoder, encoder_com
    
class mymodel_shallow(nn.Module):
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
        
        #memory
        self.mem_dim = mem_dim
        self.mem_rep = MemModule_ForLoss(
            mem_dim=mem_dim,
            fea_dim=fea_dim,
            tempreture = tempreture,
            shrink_thres=shrink_thres)

        # decoder 
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv1 = nn.Conv2d(512,256,3,stride=1,padding=1)
        self.debn1 = nn.BatchNorm2d(256)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv2 = nn.Conv2d(256,128,3,stride=1,padding=1)
        self.debn2 = nn.BatchNorm2d(128)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv3 = nn.Conv2d(128,64,3,stride=1,padding=1)
        self.debn3 = nn.BatchNorm2d(64)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv4 = nn.Conv2d(64,32,3,stride=1,padding=1)
        self.debn4 = nn.BatchNorm2d(32)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv5 = nn.Conv2d(32,3,3,stride=1,padding=1)
        self.final = nn.Sigmoid()

        #wavelet
        self.waveletDecompose = wavelet.WavePool_Two(in_channels=3).cuda()
        self.waveletCompose = wavelet.WaveUnpool(128, 'sum')
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
    
    def encoder_H(self, freq_h, feat_1, feat_2):
        x = freq_h
        x = self.conv_H1(x)
        x = self.relu(self.bn_H1(x))
        x = self.conv_H2(self.freq_connection(x, feat_1) + x)
        x = self.relu(self.bn_H2(x))
        x = self.conv_H3(self.freq_connection(x, feat_2) + x)
        x = self.relu(self.bn_H3(x))
        # x = self.conv_H4(self.freq_connection(x, feat_3) + x)
        # x = self.relu(self.bn_H4(x))
        # x = self.conv_H5(self.freq_connection(x, feat_4) + x)
        # x = self.relu(self.bn_H5(x))
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
        # feat_3 = x
        # x = self.conv4(x)
        # x = self.relu(self.bn4(x))
        # feat_4 = x
        # x = self.conv5(x)
        # x = self.relu(self.bn5(x))
        return x, feat_1, feat_2
    
    def decoder(self, x):
        # x = self.up1(x) #[-1, 512, 32, 32]
        # # [-1, 512, 32, 32] -> [-1, 256, 64, 64]
        # x = self.relu(self.debn1(self.deconv1(x)))
        # x = self.up2(x)
        # # [-1, 256, 64, 64] -> [-1, 128, 128, 128]
        # x = self.relu(self.debn2(self.deconv2(x))) 
        x = self.up3(x)
        # [-1, 128, 128, 128] -> [-1, 64, 256, 256] 
        x = self.relu(self.debn3(self.deconv3(x)))
        x = self.up4(x)
        x = self.relu(self.debn4(self.deconv4(x)))
        x = self.up5(x)
        x = self.deconv5(x)
        x = self.final(x)
        return x
    
    def forward(self,x):
        LL, LH, HL, HH = self.waveletDecompose(x)
        freq_h = torch.cat((LH,HL,HH),dim=1)
        freq_l = LL
        x_LL, feat_1, feat_2 = self.encoder_L(freq_l)
        x_H = self.encoder_H(freq_h, feat_1, feat_2)
        x_LH = x_H[:, 0:128, :, :]
        x_HL = x_H[:, 128:256, :, :]
        x_HH = x_H[:, 256:384, :, :]
        encoder_com = self.waveletCompose(x_LL, x_LH, x_HL, x_HH)
        b,c,h,w = encoder_com.shape
        encoder_com = encoder_com.view(b, -1)
        res_mem = self.mem_rep(encoder_com)
        f = res_mem['output']
        f = f.view(b,c,h,w)
        decoder = self.decoder(f)
        return decoder, encoder_com.view(b,c,w,h), f, res_mem['att'], res_mem['membank']
    
#mymodel_deeper 实验，AE + 频率分解， 与AE+FCC的对比实验
class AE_FCC(nn.Module):
    def __init__(self):
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
    
    def encoder_H(self, freq_h):
        x = freq_h
        x = self.conv_H1(x)
        x = self.relu(self.bn_H1(x))
        x = self.conv_H2(x)
        x = self.relu(self.bn_H2(x))
        x = self.conv_H3(x)
        x = self.relu(self.bn_H3(x))
        x = self.conv_H4(x)
        x = self.relu(self.bn_H4(x))
        x = self.conv_H5(x)
        x = self.relu(self.bn_H5(x))
        x = self.conv_H6(x)
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
        x_H = self.encoder_H(freq_h)
        x_LH = x_H[:, 0:1024, :, :]
        x_HL = x_H[:, 1024:2048, :, :]
        x_HH = x_H[:, 2048:3072, :, :]
        encoder_com = self.waveletCompose(x_LL, x_LH, x_HL, x_HH)
        decoder = self.decoder(encoder_com)
        return decoder, encoder_com
   
#MAE + Loss
class mae_loss(nn.Module):
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
        
        #conv
        #self.conv_connection_1 = nn.Conv2d(2, 1, 1,stride=1, padding=0)
        self.sig = nn.Sigmoid()
    
    def encoder(self,x):
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        x = self.conv3(x)
        x = self.relu(self.bn3(x))
        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        x = self.conv5(x)
        x = self.relu(self.bn5(x))
        x = self.conv6(x)
        x = self.relu(self.bn6(x))
        return x
    
    def decoder(self, x):
        x = self.up1(x) #[-1, 512, 32, 32]
        x = self.relu(self.debn1(self.deconv1(x)))
        x = self.up2(x)
        x = self.relu(self.debn2(self.deconv2(x))) 
        x = self.up3(x)
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
        encoder_com = self.encoder(x)
        b,c,h,w = encoder_com.shape
        encoder_com = encoder_com.view(b, -1)
        res_mem = self.mem_rep(encoder_com)
        f = res_mem['output']
        f = f.view(b,c,h,w)
        decoder = self.decoder(f)
        return decoder, encoder_com.view(b,c,w,h), f, res_mem['att'], res_mem['membank']
    
# inverse FCC
class inverse_fcc(nn.Module):
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
        b, c_h, h, w = freq_h.shape
        b, c_l, h, w = freq_l.shape
        b, c, h, w = x.shape
        conv_connection = nn.Conv2d(c, 1, 1, stride=1, padding=0).cuda()
        x = conv_connection(x)
        x = self.sig(x)
        x = x * freq_h
        conv_downchan = nn.Conv2d(c_h, c_l, 1, stride=1, padding=0).cuda()
        x = conv_downchan(x)
        return x
    
    def encoder_H(self, freq_h):
        x = freq_h
        x = self.conv_H1(x)
        x = self.relu(self.bn_H1(x))
        feat_1 = x
        x = self.conv_H2(x)
        x = self.relu(self.bn_H2(x))
        feat_2 = x
        x = self.conv_H3(x)
        x = self.relu(self.bn_H3(x))
        feat_3 = x
        x = self.conv_H4(x)
        x = self.relu(self.bn_H4(x))
        feat_4 = x
        x = self.conv_H5(x)
        x = self.relu(self.bn_H5(x))
        feat_5 = x
        x = self.conv_H6(x)
        x = self.relu(self.bn_H6(x))
        return x, feat_1, feat_2, feat_3, feat_4, feat_5
    
    def encoder_L(self, freq_l, feat_1, feat_2, feat_3, feat_4, feat_5):
        x = freq_l
        x = self.conv1(x) #->[32, 32, 256, 256]
        x = self.relu(self.bn1(x))
        x = self.conv2(self.freq_connection(feat_1, x) + x) #->[32, 64, 128, 128]
        x = self.relu(self.bn2(x))
        x = self.conv3(self.freq_connection(feat_2, x) + x) #->[32, 128, 64, 64]
        x = self.relu(self.bn3(x))
        x = self.conv4(self.freq_connection(feat_3, x) + x)
        x = self.relu(self.bn4(x))
        x = self.conv5(self.freq_connection(feat_4, x) + x)
        x = self.relu(self.bn5(x))
        x = self.conv6(self.freq_connection(feat_5, x) + x)
        x = self.relu(self.bn6(x))
        return x
    
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
        x_H, feat_1, feat_2, feat_3, feat_4, feat_5 = self.encoder_H(freq_h)
        x_LL = self.encoder_L(freq_l, feat_1, feat_2, feat_3, feat_4, feat_5)
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
    