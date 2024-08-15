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

nepoches = 70
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
             is_bias=True, non_linear_act=False, lock_lowlevel_network=False, isFT=False)
model.to(device)
# %%
params_to_update = []
for name, param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print("\t", name)
# %%
lr = 0.0001
weight_decay = 0.0001  # 0.000#0.#
momentum = 0.9  # 0.#
optimizer = optim.RMSprop(params_to_update, lr=lr, weight_decay=weight_decay, momentum=momentum)
# %%
print('-' * 30)
print('learing rate {}'.format(lr))
print('trainable V {}'.format(trainable_w2v))
print('lambda_ {}'.format(lambda_))
print('optimized seen only')
print('optimizer: RMSProp with momentum = {} and weight_decay = {}'.format(momentum, weight_decay))
print('-' * 30)

iter_x = []
best_H = []
best_ACC = []
best_performance = [0, 0, 0]
best_acc = 0

for i in range(0, nepoches):
    model.train()
    model.onet.eval()
    if dataset.is_balance:
        train_sampler = BanlanceSampler(clist, train_batch_size)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size, \
                                                     sampler=train_sampler, num_workers=0)
    for iter, (img, label, att) in enumerate(dataset_loader):
        optimizer.zero_grad()

        img, label, att = img.to(device), label.to(device), att.to(device)

        out_package1, out_package2= model(img)

        in_package1 = out_package1
        in_package2 = out_package2
        in_package1['batch_label'] = label
        in_package2['batch_label'] = label

        out_package1=model.compute_loss(in_package1)
        out_package2=model.compute_loss(in_package2)
        loss_b = out_package1['loss']+out_package2['loss']
        constrastive_loss1=model.compute_contrastive_loss(in_package1, in_package2)
        loss_b = loss_b + 0.01*constrastive_loss1

        loss= loss_b

        loss.backward()
        optimizer.step()

        if iter%20 == 0:
            print('epoch =%d, iter=%d, loss=%.3f, loss_b=%.3f' \
                  % (i, iter, loss.item(), loss_b.item()))

    acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(dataset_loader_seen, dataset_loader_unseen,model,device,
                                                  bias_seen=-bias,bias_unseen=bias)

    print('**********************************************************************')
    print('epoch=%d, acc_unseen=%.3f, acc_seen=%.3f, H=%.3f, acc_zs=%.3f' \
          % (i, acc_novel, acc_seen, H, acc_zs))
    if H > best_performance[2]:
        best_performance = [acc_novel, acc_seen, H]
        torch.save(model.state_dict(), './model/pretrained_sun_h.pth')
    if acc_zs > best_acc:
        best_acc = acc_zs
        torch.save(model.state_dict(), './model/pretrained_sun_z.pth')
    print('epoch=%d, acc_unseen=%.3f, acc_seen=%.3f, H=%.3f, acc_zs=%.3f'\
          %(i, best_performance[0],best_performance[1],best_performance[2],best_acc))
    print('**********************************************************************')
