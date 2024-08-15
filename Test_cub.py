import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from core.AARR import AARR
from core.CUBDataLoader import CUBTrainDataset,CUBSeenTestDataset,CUBUnseenTestDataset
from core.helper_CUB import eval_zs_gzsl
# from global_setting import NFS_path
import importlib
import pdb
import numpy as np
import matplotlib.pyplot as plt
from utils import *

NFS_path = './'

idx_GPU = 0
device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")
dataset = CUBTrainDataset(NFS_path,is_unsupervised_attr=False,is_balance=False)
dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=16)
dataset_seen = CUBSeenTestDataset(NFS_path,is_unsupervised_attr=False,is_balance=False)
dataset_loader_seen = torch.utils.data.DataLoader(dataset_seen, batch_size=256, shuffle=False, num_workers=16)
dataset_unseen = CUBUnseenTestDataset(NFS_path,is_unsupervised_attr=False,is_balance=False)
dataset_loader_unseen = torch.utils.data.DataLoader(dataset_unseen, batch_size=256, shuffle=False, num_workers=16)
torch.backends.cudnn.benchmark = True

def get_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr.append(param_group['lr'])
    return lr

seed = 3407 #214#215#
#torch.manual_seed(seed)
#torch.cuda.manual_seed_all(seed)
#np.random.seed(seed)

nepoches = 8
dim_f = 2048
dim_v = 300
init_w2v_att = dataset.w2v_att
att = dataset.att
normalize_att = dataset.normalize_att

trainable_w2v = True
lambda_ = 0.1  #0.1 for GZSL, 0.18 for CZSL
bias = 0
prob_prune = 0
uniform_att_1 = False
uniform_att_2 = False

seenclass = dataset.seenclasses
unseenclass = dataset.unseenclasses
desired_mass = 1

model = AARR(dim_f,dim_v,init_w2v_att,att,normalize_att,
            seenclass,unseenclass,
            lambda_,
            trainable_w2v,normalize_V=False,normalize_F=True,is_conservative=True,
            uniform_att_1=uniform_att_1,uniform_att_2=uniform_att_2,
            prob_prune=prob_prune,desired_mass=desired_mass, is_conv=False,
            is_bias=True, lock_lowlevel_network=True)
model.to(device)

model.load_state_dict(torch.load('./model/AR2_CUB_GZSL.pth'), strict=False)
model.eval()

with torch.no_grad():

    acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(dataset_loader_seen, dataset_loader_unseen,model,device,
                                                  bias_seen=-bias,bias_unseen=bias)

    print('**********************************************************************')
    print('acc_unseen=%.3f, acc_seen=%.3f, H=%.3f, acc_zs=%.3f' \
          % (acc_novel, acc_seen, H, acc_zs))