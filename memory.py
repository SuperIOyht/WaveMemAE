from __future__ import absolute_import, print_function
import torch
from torch import nn
import math
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np


class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # (Rows in the memory matrix) M x (dimension of each row of the memory matrix) C 
        self.bias = None
        self.shrink_thres = shrink_thres
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        att_weight = F.linear(input, self.weight)  # Features x Memory^T, the dimensions are : (BatchSize N  x FeatureDimension C) x (FeatureDimension C x Rows in the memory matrix M) = output matrix is of size N x M
        znormal = self.weight.square().sum(dim=1).sqrt().reshape(-1,1)
        cnormal = input.square().sum(dim=1).sqrt().reshape(-1,1)
        cznormal = cnormal.mul(znormal.T)
        att_weight = att_weight.div(cznormal)
        att_weight = F.softmax(att_weight, dim=1)  # Softmax on matix of size NxM along Mth Dimension, i.e. softmax along each row which results as row summing up to 1 for all the N rows of Matrix; Maxtix of size N x M is returned
        # ReLU based shrinkage, hard shrinkage for positive value
        if(self.shrink_thres>0):
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)
        mem_trans = self.weight.permute(1, 0)  # Mem^T, matrix of size C x M is the result as  weights are of dimension M x C
        output = F.linear(att_weight, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, attention weights (NxM) x (MxC) = NxC
        return {'output': output, 'att': att_weight}  # output (N x C), att_weight (N X M)

    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}'.format(
            self.mem_dim, self.fea_dim is not None
        )


# NxCxHxW -> (NxHxW)xC -> addressing Mem, (NxHxW)xC -> NxCxHxW
class MemModule(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025, device='cuda'):
        super(MemModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)
        
    def forward(self, input):
        s = input.data.shape # 8, 512
        l = len(s)

        if l == 2:
            x = input.permute(0, 1)             # BatchSize x FeatureSize
        elif l == 3:
            x = input.permute(0, 2, 1)
        elif l == 4:
            x = input.permute(0, 2, 3, 1)
        elif l == 5:
            x = input.permute(0, 2, 3, 4, 1)
        else:
            x = []
            print('wrong feature map size')

        x = x.contiguous()
        x = x.view(-1, s[1])
        #
        y_and = self.memory(x)
        #
        y = y_and['output']
        att = y_and['att']

        if l==2:
            y = y.view(s[0], s[1])
            y = y.permute(0, 1)
            att = att.view(s[0], self.mem_dim)
            att = att.permute(0, 1)
        elif l == 3:
            y = y.view(s[0], s[2], s[1])
            y = y.permute(0, 2, 1)
            att = att.view(s[0], s[2], self.mem_dim)
            att = att.permute(0, 2, 1)
        elif l == 4:
            y = y.view(s[0], s[2], s[3], s[1])
            y = y.permute(0, 3, 1, 2)
            att = att.view(s[0], s[2], s[3], self.mem_dim)
            att = att.permute(0, 3, 1, 2)
        elif l == 5:
            y = y.view(s[0], s[2], s[3], s[4], s[1])
            y = y.permute(0, 4, 1, 2, 3)
            att = att.view(s[0], s[2], s[3], s[4], self.mem_dim)
            att = att.permute(0, 4, 1, 2, 3)
        else:
            y = x
            att = att
            print('wrong feature map size')
        return {'output': y, 'att': att}


# class MemoryUnit(nn.Module):
#     def __init__(self, mem_dim, fea_dim, tempreture, shrink_thres=0.0025):
#         super(MemoryUnit, self).__init__()
#         self.mem_dim = mem_dim
#         self.fea_dim = fea_dim
#         self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # (Rows in the memory matrix) M x (dimension of each row of the memory matrix) C 
#         self.bias = None
#         self.shrink_thres = shrink_thres
#         self.tempreture = tempreture
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)

#     def forward(self, input):
#         att_weight = F.linear(input, self.weight)  # Features x Memory^T, the dimensions are : (BatchSize N  x FeatureDimension C) x (FeatureDimension C x Rows in the memory matrix M) = output matrix is of size N x M
#         znormal = self.weight.square().sum(dim=1).sqrt().reshape(-1,1)
#         cnormal = input.square().sum(dim=1).sqrt().reshape(-1,1)
#         cznormal = cnormal.mul(znormal.T)
#         att_weight = att_weight.div(cznormal).div(self.tempreture)
#         att_weight = F.softmax(att_weight, dim=1)  # Softmax on matix of size NxM along Mth Dimension, i.e. softmax along each row which results as row summing up to 1 for all the N rows of Matrix; Maxtix of size N x M is returned
#         # ReLU based shrinkage, hard shrinkage for positive value
#         if(self.shrink_thres>0):
#             att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
#             att_weight = F.normalize(att_weight, p=1, dim=1)
#         mem_trans = self.weight.permute(1, 0)  # Mem^T, matrix of size C x M is the result as  weights are of dimension M x C
#         output = F.linear(att_weight, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, attention weights (NxM) x (MxC) = NxC
#         return {'output': output, 'att': att_weight}  # output (N x C), att_weight (N X M)

#     def extra_repr(self):
#         return 'mem_dim={}, fea_dim={}'.format(
#             self.mem_dim, self.fea_dim is not None
#         )


# # NxCxHxW -> (NxHxW)xC -> addressing Mem, (NxHxW)xC -> NxCxHxW
# class MemModule(nn.Module):
#     def __init__(self, mem_dim, fea_dim, tempreture, shrink_thres=0.0025, device='cuda'):
#         super(MemModule, self).__init__()
#         self.mem_dim = mem_dim
#         self.fea_dim = fea_dim
#         self.shrink_thres = shrink_thres
#         self.tempreture = tempreture
#         self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.tempreture, self.shrink_thres)
        
#     def forward(self, input):
#         s = input.data.shape # 8, 512
#         l = len(s)

#         if l == 2:
#             x = input.permute(0, 1)             # BatchSize x FeatureSize
#         elif l == 3:
#             x = input.permute(0, 2, 1)
#         elif l == 4:
#             x = input.permute(0, 2, 3, 1)
#         elif l == 5:
#             x = input.permute(0, 2, 3, 4, 1)
#         else:
#             x = []
#             print('wrong feature map size')

#         x = x.contiguous()
#         x = x.view(-1, s[1])
#         #
#         y_and = self.memory(x)
#         #
#         y = y_and['output']
#         att = y_and['att']

#         if l==2:
#             y = y.view(s[0], s[1])
#             y = y.permute(0, 1)
#             att = att.view(s[0], self.mem_dim)
#             att = att.permute(0, 1)
#         elif l == 3:
#             y = y.view(s[0], s[2], s[1])
#             y = y.permute(0, 2, 1)
#             att = att.view(s[0], s[2], self.mem_dim)
#             att = att.permute(0, 2, 1)
#         elif l == 4:
#             y = y.view(s[0], s[2], s[3], s[1])
#             y = y.permute(0, 3, 1, 2)
#             att = att.view(s[0], s[2], s[3], self.mem_dim)
#             att = att.permute(0, 3, 1, 2)
#         elif l == 5:
#             y = y.view(s[0], s[2], s[3], s[4], s[1])
#             y = y.permute(0, 4, 1, 2, 3)
#             att = att.view(s[0], s[2], s[3], s[4], self.mem_dim)
#             att = att.permute(0, 4, 1, 2, 3)
#         else:
#             y = x
#             att = att
#             print('wrong feature map size')
#         return {'output': y, 'att': att}



# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output


class MemoryUnit_ForLoss(nn.Module):
    def __init__(self, mem_dim, fea_dim, tempreture, shrink_thres=0.0025):
        super(MemoryUnit_ForLoss, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # (Rows in the memory matrix) M x (dimension of each row of the memory matrix) C 
        self.bias = None
        self.shrink_thres = shrink_thres
        self.tempreture = tempreture
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):

        att_weight = F.linear(input, self.weight)  # Features x Memory^T, the dimensions are : (BatchSize N  x FeatureDimension C) x (FeatureDimension C x Rows in the memory matrix M) = output matrix is of size N x M
        znormal = self.weight.square().sum(dim=1).sqrt().reshape(-1,1) #|m|
        cnormal = input.square().sum(dim=1).sqrt().reshape(-1,1) #|z|
        cznormal = cnormal.mul(znormal.T)
        att_weight = att_weight.div(cznormal).div(self.tempreture)
        att_weight = F.softmax(att_weight, dim=1)  # Softmax on matix of size NxM along Mth Dimension, i.e. softmax along each row which results as row summing up to 1 for all the N rows of Matrix; Maxtix of size N x M is returned
        # ReLU based shrinkage, hard shrinkage for positive value
        if(self.shrink_thres > 0):
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)
        # #mask
        # n = 7
        # mask = torch.zeros(64).cuda()
        # pos = (att_weight != 0).nonzero(as_tuple=False).squeeze(1)
        # index = pos[n-1][1]
        # mask[index] = 1
        # mask = mask.repeat(1, 1)
        # att_weight = mask
        # #mask
        mem_trans = self.weight.permute(1, 0)  # Mem^T, matrix of size C x M is the result as  weights are of dimension M x C
        output = F.linear(att_weight, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, attention weights (NxM) x (MxC) = NxC
        return {'output': output, 'att': att_weight, 'membank': self.weight}  # output (N x C), att_weight (N X M)

    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}'.format(
            self.mem_dim, self.fea_dim is not None
        )


# NxCxHxW -> (NxHxW)xC -> addressing Mem, (NxHxW)xC -> NxCxHxW
class MemModule_ForLoss(nn.Module):
    def __init__(self, mem_dim, fea_dim, tempreture, shrink_thres=0.0025,  device='cuda'):
        super(MemModule_ForLoss, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit_ForLoss(self.mem_dim, self.fea_dim, tempreture, self.shrink_thres)

    def forward(self, input):
        s = input.data.shape # 8, 512
        l = len(s)

        if l == 2:
            x = input.permute(0, 1)             # BatchSize x FeatureSize
        elif l == 3:
            x = input.permute(0, 2, 1)
        elif l == 4:
            x = input.permute(0, 2, 3, 1)
        elif l == 5:
            x = input.permute(0, 2, 3, 4, 1)
        else:
            x = []
            print('wrong feature map size')

        x = x.contiguous()
        x = x.view(-1, s[1])
        #
        y_and = self.memory(x)
        #
        y = y_and['output']
        att = y_and['att']
        membank = y_and['membank']

        if l==2:
            y = y.view(s[0], s[1])
            y = y.permute(0, 1)
            att = att.view(s[0], self.mem_dim)
            att = att.permute(0, 1)
        elif l == 3:
            y = y.view(s[0], s[2], s[1])
            y = y.permute(0, 2, 1)
            att = att.view(s[0], s[2], self.mem_dim)
            att = att.permute(0, 2, 1)
        elif l == 4:
            y = y.view(s[0], s[2], s[3], s[1])
            y = y.permute(0, 3, 1, 2)
            att = att.view(s[0], s[2], s[3], self.mem_dim)
            att = att.permute(0, 3, 1, 2)
        elif l == 5:
            y = y.view(s[0], s[2], s[3], s[4], s[1])
            y = y.permute(0, 4, 1, 2, 3)
            att = att.view(s[0], s[2], s[3], s[4], self.mem_dim)
            att = att.permute(0, 4, 1, 2, 3)
        else:
            y = x
            att = att
            print('wrong feature map size')
        return {'output': y, 'att': att, 'membank': membank}
    
    
    
class MemoryUnit_ForLoss_notemp(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemoryUnit_ForLoss_notemp, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # (Rows in the memory matrix) M x (dimension of each row of the memory matrix) C 
        self.bias = None
        self.shrink_thres = shrink_thres
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        att_weight = F.linear(input, self.weight)  # Features x Memory^T, the dimensions are : (BatchSize N  x FeatureDimension C) x (FeatureDimension C x Rows in the memory matrix M) = output matrix is of size N x M
        znormal = self.weight.square().sum(dim=1).sqrt().reshape(-1,1) #|m|
        cnormal = input.square().sum(dim=1).sqrt().reshape(-1,1) #|z|
        cznormal = cnormal.mul(znormal.T)
        att_weight = att_weight.div(cznormal)
        att_weight = F.softmax(att_weight, dim=1)  # Softmax on matix of size NxM along Mth Dimension, i.e. softmax along each row which results as row summing up to 1 for all the N rows of Matrix; Maxtix of size N x M is returned
        # ReLU based shrinkage, hard shrinkage for positive value
        if(self.shrink_thres > 0):
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)
        mem_trans = self.weight.permute(1, 0)  # Mem^T, matrix of size C x M is the result as  weights are of dimension M x C
        output = F.linear(att_weight, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, attention weights (NxM) x (MxC) = NxC
        return {'output': output, 'att': att_weight, 'membank': self.weight}  # output (N x C), att_weight (N X M)

    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}'.format(
            self.mem_dim, self.fea_dim is not None
        )


# NxCxHxW -> (NxHxW)xC -> addressing Mem, (NxHxW)xC -> NxCxHxW
class MemModule_ForLoss_notemp(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025,  device='cuda'):
        super(MemModule_ForLoss_notemp, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit_ForLoss_notemp(self.mem_dim, self.fea_dim, self.shrink_thres)

    def forward(self, input):
        s = input.data.shape # 8, 512
        l = len(s)

        if l == 2:
            x = input.permute(0, 1)             # BatchSize x FeatureSize
        elif l == 3:
            x = input.permute(0, 2, 1)
        elif l == 4:
            x = input.permute(0, 2, 3, 1)
        elif l == 5:
            x = input.permute(0, 2, 3, 4, 1)
        else:
            x = []
            print('wrong feature map size')

        x = x.contiguous()
        x = x.view(-1, s[1])
        #
        y_and = self.memory(x)
        #
        y = y_and['output']
        att = y_and['att']
        membank = y_and['membank']

        if l==2:
            y = y.view(s[0], s[1])
            y = y.permute(0, 1)
            att = att.view(s[0], self.mem_dim)
            att = att.permute(0, 1)
        elif l == 3:
            y = y.view(s[0], s[2], s[1])
            y = y.permute(0, 2, 1)
            att = att.view(s[0], s[2], self.mem_dim)
            att = att.permute(0, 2, 1)
        elif l == 4:
            y = y.view(s[0], s[2], s[3], s[1])
            y = y.permute(0, 3, 1, 2)
            att = att.view(s[0], s[2], s[3], self.mem_dim)
            att = att.permute(0, 3, 1, 2)
        elif l == 5:
            y = y.view(s[0], s[2], s[3], s[4], s[1])
            y = y.permute(0, 4, 1, 2, 3)
            att = att.view(s[0], s[2], s[3], s[4], self.mem_dim)
            att = att.permute(0, 4, 1, 2, 3)
        else:
            y = x
            att = att
            print('wrong feature map size')
        return {'output': y, 'att': att, 'membank': membank}