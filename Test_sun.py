import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from core.AARR import AARR
from core.SUNDataLoader import SUNTrainDataLoader as suntrain
from core.SUNDataLoader import SUNSeenTestDataLoader as sunseen
from core.SUNDataLoader import SUNUnSeenTestDataLoader as sununseen
from core.helper_SUN import eval_zs_gzsl
import importlib
import pdb
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn.functional as F

NFS_path = './'

idx_GPU = 0
train_batch_size = 50
device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")
dataset = suntrain(NFS_path, is_balance=False)
dataset_seen = sunseen(NFS_path,is_balance=False)
dataset_loader_seen = torch.utils.data.DataLoader(dataset_seen, batch_size=256, shuffle=False, num_workers=16)
dataset_unseen = sununseen(NFS_path,is_balance=False)
dataset_loader_unseen = torch.utils.data.DataLoader(dataset_unseen, batch_size=256, shuffle=False, num_workers=16)
torch.backends.cudnn.benchmark = True

if dataset.is_balance:
    clist = dataset.idxs_list
    #weights = []
    #for i in range(len(clist)):
    #    cnum = len(clist[i])
    #    weights.extend([1/cnum]*cnum)
    #train_sampler = WeightedRandomSampler(weights, dataset.ntrain, replacement = True)
    train_sampler = BanlanceSampler(clist, train_batch_size)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size, sampler=train_sampler, num_workers=0)
else:
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=16)

seed = 3407
#torch.manual_seed(seed)
#torch.cuda.manual_seed_all(seed)
#np.random.seed(seed)

print('Randomize seed {}'.format(seed))
# %%

nepoches = 8
dim_f = 2048
dim_v = 300
init_w2v_att = dataset.w2v_att
att = dataset.att  # dataloader.normalize_att#
normalize_att = dataset.normalize_att
# assert (att.min().item() == 0 and att.max().item() == 1)

trainable_w2v = True
lambda_ = 0.035  # 0.035 for gzsl
bias = 0.
prob_prune = 0
uniform_att_1 = False
uniform_att_2 = True

seenclass = dataset.seenclasses
unseenclass = dataset.unseenclasses
desired_mass = 1  # unseenclass.size(0)/(seenclass.size(0)+unseenclass.size(0))

# %%
model = AARR(dim_f, dim_v, init_w2v_att, att, normalize_att,
             seenclass, unseenclass,
             lambda_,
             trainable_w2v, normalize_V=False, normalize_F=True, is_conservative=True,
             uniform_att_1=uniform_att_1, uniform_att_2=uniform_att_2,
             prob_prune=prob_prune, desired_mass=desired_mass, is_conv=False,
             is_bias=True, non_linear_act=False, lock_lowlevel_network=False)
model.to(device)

model.load_state_dict(torch.load('./model/AR2_SUN_GZSL.pth'), strict=False)
model.eval()

with torch.no_grad():

    acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(dataset_loader_seen, dataset_loader_unseen,model,device,
                                                  bias_seen=-bias,bias_unseen=bias)

    print('**********************************************************************')
    print('acc_unseen=%.3f, acc_seen=%.3f, H=%.3f, acc_zs=%.3f' \
          % (acc_novel, acc_seen, H, acc_zs))