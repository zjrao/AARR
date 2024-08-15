import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from core.AARR import AARR
from core.AWA2DataLoader import AWA2TrainDataLoader as awatrain
from core.AWA2DataLoader import AWA2SeenTestDataLoader as awaseen
from core.AWA2DataLoader import AWA2UnSeenTestDataLoader as awaunseen
from core.helper_AWA2 import eval_zs_gzsl
import importlib
import pdb
import numpy as np
from utils import *

NFS_path = './'

idx_GPU = 0
train_batch_size = 50
device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")
dataset = awatrain(NFS_path)
dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=8)
dataset_seen = awaseen(NFS_path)
dataset_loader_seen = torch.utils.data.DataLoader(dataset_seen, batch_size=256, shuffle=False, num_workers=8)
dataset_unseen = awaunseen(NFS_path)
dataset_loader_unseen = torch.utils.data.DataLoader(dataset_unseen, batch_size=256, shuffle=False, num_workers=8)
torch.backends.cudnn.benchmark = True


def get_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr.append(param_group['lr'])
    return lr


seed = 87778
# seed = 6379 # for czsl
#torch.manual_seed(seed)
#torch.cuda.manual_seed_all(seed)
#np.random.seed(seed)

nepoches = 8
dim_f = 2048
dim_v = 300
init_w2v_att = dataset.w2v_att
att = dataset.att  # dataloader.normalize_att#
att[att < 0] = 0
normalize_att = dataset.normalize_att
# assert (att.min().item() == 0 and att.max().item() == 1)

trainable_w2v = True
lambda_ = 0.14  # 0.12 for GZSL, 0.3 for CZSL
bias = 0
prob_prune = 0
uniform_att_1 = False
uniform_att_2 = False

seenclass = dataset.seenclasses
unseenclass = dataset.unseenclasses
desired_mass = 1  # unseenclass.size(0)/(seenclass.size(0)+unseenclass.size(0))

model = AARR(dim_f, dim_v, init_w2v_att, att, normalize_att,
             seenclass, unseenclass,
             lambda_,
             trainable_w2v, normalize_V=True, normalize_F=True, is_conservative=True,
             uniform_att_1=uniform_att_1, uniform_att_2=uniform_att_2,
             prob_prune=prob_prune, desired_mass=desired_mass, is_conv=False,
             is_bias=True, lock_lowlevel_network=True)
model.to(device)

model.load_state_dict(torch.load('./model/AR2_AWA_GZSL.pth'), strict=False)
model.eval()
with torch.no_grad():

    acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(dataset_loader_seen, dataset_loader_unseen,model,device,
                                                  bias_seen=-bias,bias_unseen=bias)

    print('**********************************************************************')
    print('acc_unseen=%.3f, acc_seen=%.3f, H=%.3f, acc_zs=%.3f' \
          % (acc_novel, acc_seen, H, acc_zs))