from os import write
import torch
import os
import cv2
import numpy as np
import dataloader as mydataloader
import torch.utils.data
import models.memae
import models.ae
from tqdm import tqdm
import metric
import time
import wavelet
from tensorboardX import SummaryWriter
import logging
import models.my_loss
import pickle

save_ckp = True
epoch = 300


print("======load data start======")
#real_fingerprint_list = mydataloader.get_file_list(mydataloader.truesets[0])
spoof_playdoh_fingerprint_list = mydataloader.get_file_list(mydataloader.fakesets[0])
spoof_ecoflex_fingerprint_list = mydataloader.get_file_list(mydataloader.fakesets[1])
spoof_woodglue_fingerprint_list = mydataloader.get_file_list(mydataloader.fakesets[2])
spoof_photopaper_fingerprint_list = mydataloader.get_file_list(mydataloader.fakesets[3])

train_spoof_woodglue, test_spoof_woodglue = mydataloader.split_dataset(spoof_woodglue_fingerprint_list, 0.15)
test_spoof_woodglue_dataset = mydataloader.fingerPrintDataset(test_spoof_woodglue, 2)
train_spoof_playdoh, test_spoof_playdoh = mydataloader.split_dataset(spoof_playdoh_fingerprint_list, 0.15)
test_spoof_playdoh_dataset = mydataloader.fingerPrintDataset(test_spoof_playdoh, 2)
train_spoof_ecoflex, test_spoof_ecoflex = mydataloader.split_dataset(spoof_ecoflex_fingerprint_list, 0.15)
test_spoof_ecoflex_dataset = mydataloader.fingerPrintDataset(test_spoof_ecoflex, 2)
train_spoof_photopaper, test_spoof_photopaper = mydataloader.split_dataset(spoof_photopaper_fingerprint_list, 0.2)
test_spoof_photopaper_dataset = mydataloader.fingerPrintDataset(test_spoof_photopaper, 2)

#train_real_fingerprint_list, test_real_fingerprint_list = mydataloader.split_dataset(real_fingerprint_list)
with open("live_train.pkl", "rb") as f:
    train_real_fingerprint_list = pickle.load(f)
    
with open("live_test.pkl", "rb") as f:
    test_real_fingerprint_list = pickle.load(f)
    
train_real_dataset = mydataloader.fingerPrintDataset(train_real_fingerprint_list, 1)
test_real_dataset = mydataloader.fingerPrintDataset(test_real_fingerprint_list, 1)

test_dataset = torch.utils.data.ConcatDataset([test_spoof_woodglue_dataset,
                                              test_spoof_playdoh_dataset,
                                              test_spoof_ecoflex_dataset,
                                              test_spoof_photopaper_dataset,
                                              test_real_dataset]
                                              )

train_dataloader = torch.utils.data.DataLoader(train_real_dataset, batch_size=64, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

print("======data load over======")
wavelet_net = wavelet.WavePool_Two(in_channels=3).cuda()
# mem_dim = 64
mem_dim = 64
shrink_thres = 1/mem_dim
tempreture = 0.1
print("lambda: {}".format(shrink_thres))
print("tempreture: {}".format(tempreture))
net1 = models.memae.mymodel_deeper(mem_dim = mem_dim, fea_dim=1024*8*8, tempreture=tempreture ,shrink_thres = shrink_thres).cuda()
loss_fn = torch.nn.MSELoss()
dis_loss = models.my_loss.mem_distance_loss_changed()
ector_loss = models.my_loss.regular_entropy_loss()
opt = torch.optim.Adam(net1.parameters(), lr = 2e-4, weight_decay=1e-5)
# opt = torch.optim.Adam(net1.parameters(), lr = 1e-4, weight_decay=1e-4)
best_auc = 0


start_time = time.ctime()
start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.strptime(start_time))
checkpoint_path = "./checkpoint/{}/".format(start_time)
tensorboard_log_dir = os.path.join(checkpoint_path, 'tensorboard_logs')
writer = SummaryWriter(tensorboard_log_dir)
log_file = os.path.join(checkpoint_path, 'logs')
logging.basicConfig(filename=log_file, level=logging.INFO)
logging.info("model: {}".format(net1._get_name()))

#训练信息
logging.info("two encoder using wavelet and connection(mat mul, my connection, my memory Loss, loss =  feat_loss * 1000 + recon_loss * 1 + mem_distance_loss * 2, +temp), temp:{}, lambda:{}".format(tempreture, shrink_thres))
logging.info("retest for lambda 1/100, temp = 0.01, decoder avtivate func=relu")
logging.info("final 测试，训练集重新划分")


def train(epoch):
    net1.train()
    step = 0
    for data, label, x in tqdm(train_dataloader, leave=False):
        data = data.cuda()
        label = label.cuda()
        decoder, encoder, f, att_weight, membank = net1(data)
        feat_loss = loss_fn(encoder, f)
        recon_loss = loss_fn(data, decoder)
        mem_distance_loss = dis_loss(membank)
        #entro_loss = ector_loss(att_weight)
        #entro_loss = entropy_loss(att_weight)
        #loss = feat_loss_L  + recon_loss_L +  0.0002 * entro_loss
        loss = feat_loss * 1000  + recon_loss * 1 + mem_distance_loss * 2 #+ entro_loss
        writer.add_scalar("feat_loss", feat_loss, epoch * len(train_dataloader) + step)
        writer.add_scalar("recon_loss", recon_loss, epoch * len(train_dataloader) + step)
        writer.add_scalar("mem_distance_loss", mem_distance_loss, epoch * len(train_dataloader) + step)
        writer.add_scalar("total_loss", loss, epoch * len(train_dataloader) + step)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step = step + 1
    print("======TRAIN====== epoch: {}, loss: {}".format(epoch, loss.item()))
    logging.info("======TRAIN====== epoch: {}, loss: {}".format(epoch, loss.item()))

@torch.no_grad()
def test(epoch):
    net1.eval()
    list_auc = []
    list_err = []
    list_thre = []
    list_eer = []
    list_bpcer20 = []
    list_bpcer10 = []
    global best_auc
    global checkpoint_path
    step = 0
    for data, label, path in tqdm(test_dataloader, leave=False):
        data = data.cuda()
        label = label.cuda()
        decoder, encoder, f, att_weight, membank = net1(data)
        data = data.cpu().numpy()
        label = label.cpu().numpy()
        scores_L = metric.L2Norm_dis(encoder.cpu().numpy(), f.cpu().numpy(), batch_size = encoder.shape[0], channel = f.shape[1])
        string, res = metric.get_err_res(label, scores_L)
        writer.add_scalar("best_thre", res["best_thre"], epoch*len(test_dataloader) + step)

        list_err.append(res["err"])
        list_auc.append(res["auc_score"])
        list_thre.append(res["best_thre"])
        list_eer.append(res["eer"])
        list_bpcer10.append(res["bpcer_10"])
        list_bpcer20.append(res["bpcer_20"])
        step = step + 1
    err = np.mean(list_err)
    auc = np.mean(list_auc)
    thre = np.mean(list_thre)
    eer = np.mean(list_eer)
    bpcer20 = np.mean(list_bpcer20)
    bpcer10 = np.mean(list_bpcer10)
    
    print("======TEST====== epoch: {}, err: {}, auc: {}, thre: {}, eer: {}, bpcer10: {}, bpcer20:{}".format(epoch, err, auc, thre, eer, bpcer10, bpcer20))
    logging.info("======TEST====== epoch: {}, err: {}, auc: {}, thre: {}, eer: {}, bpcer10: {}, bpcer20:{}".format(epoch, err, auc, thre, eer, bpcer10, bpcer20))
    writer.add_scalar("err", err, epoch)
    writer.add_scalar("auc", auc, epoch)
    
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    if save_ckp:
        checkpoint_path_epoch = os.path.join(checkpoint_path, '{}_best.pth'.format(epoch))
        if best_auc < auc:
            best_auc = auc
            torch.save(net1.state_dict(), checkpoint_path_epoch)

print("======strat train======")
for i in range(epoch):
    train(i)
    test(i)

writer.close()

