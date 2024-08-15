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

setup = {'pmp': {'init_lambda': 0.1, 'final_lambda': 0.1, 'phase': 0.8},
         'desired_mass': {'init_lambda': -1, 'final_lambda': -1, 'phase': 0.8}}
print(setup)

params_to_update = []
params_names = []
for name, param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        params_names.append(name)
        print("\t", name)
# %%
lr = 0.00001
weight_decay = 0.0001  # 0.000#0.#
momentum = 0.  # 0.#
# %%
lr_seperator = 1
lr_factor = 1
print('default lr {} {}x lr {}'.format(params_names[:lr_seperator], lr_factor, params_names[lr_seperator:]))
optimizer = optim.RMSprop(params_to_update, lr=lr, weight_decay=weight_decay, momentum=momentum)
print('-' * 30)
print('learing rate {}'.format(lr))
print('trainable V {}'.format(trainable_w2v))
print('lambda_ {}'.format(lambda_))
print('optimized seen only')
print('optimizer: RMSProp with momentum = {} and weight_decay = {}'.format(momentum, weight_decay))
print('-' * 30)

best_performance = [0, 0, 0]
best_acc = 0

label_map = computeSimi(dataset.att, dataset.seenclasses, dataset.unseenclasses, 5).to(device)

model.load_state_dict(torch.load('./model/pretrained_awa2_h.pth'), strict=False)
gradmodel = Gradmap(label_map, 0.1, mode=2, n=50, attsize=85).to(device)
gradmodel.load_state_dict(torch.load('./model/pretrained_awa2_h.pth'), strict=False)

protoNet = computeProto(num_classes=50, num_att=85, dim_f=dim_f, num_sample=len(dataset_loader),
                        att=att, device=device, model=model, dataset=dataset).to(device)
protoNet.pretrainW()
for i in range(0, nepoches):
    model.train()
    model.onet.eval()
    for iter, (img, label, att) in enumerate(dataset_loader):
        optimizer.zero_grad()

        img, label, att = img.to(device), label.to(device), att.to(device)

        out_package1, out_package2= model(img)

        in_package1 = out_package1
        in_package2 = out_package2
        in_package1['batch_label'] = label
        in_package2['batch_label'] = label

        out_package1 = model.compute_loss(in_package1)
        out_package2 = model.compute_loss(in_package2)
        loss_b = out_package1['loss'] + out_package2['loss']
        constrastive_loss = model.compute_contrastive_loss(in_package1, in_package2)
        loss_b = loss_b + 0.0000001*constrastive_loss

        Fs_o, Fs_m = in_package1['Fs_o'], in_package1['Fs_m']
        att_map = gradmodel.computGrad(Fs_o.clone().detach(), label, isclip=False, clipThresh=0.5)
        loss_align = gradmodel.mmse_loss(Fs_m, Fs_o.clone().detach(), att_map.clone().detach())

        score = in_package1['A']
        proto_loss = protoNet(Fs_m, score, label)

        loss= loss_b + 0.1*proto_loss + 10*loss_align

        loss.backward()
        optimizer.step()

        if iter%40 == 0:
            print('epoch =%d, iter=%d, loss=%.3f, loss_b=%.3f, loss_align=%.3f, proto_loss=%.3f' \
                  % (i, iter, loss.item(), loss_b.item(), loss_align.item(), proto_loss.item()))

    acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(dataset_loader_seen, dataset_loader_unseen,model,device,
                                                  bias_seen=-bias,bias_unseen=bias)

    print('**********************************************************************')
    print('epoch=%d, acc_unseen=%.3f, acc_seen=%.3f, H=%.3f, acc_zs=%.3f' \
          % (i, acc_novel, acc_seen, H, acc_zs))

    warm_update_teacher(model.mnet, model.onet)
    warm_update(gradmodel, model)

    if H > best_performance[2]:
        best_performance = [acc_novel, acc_seen, H]
        torch.save(model.state_dict(), './model/best_H_AWA2.pth')
    if acc_zs > best_acc:
        best_acc = acc_zs
        torch.save(model.state_dict(), './model/best_Z_AWA2.pth')
    print('epoch=%d, acc_unseen=%.3f, acc_seen=%.3f, H=%.3f, acc_zs=%.3f'\
          %(i, best_performance[0],best_performance[1],best_performance[2],best_acc))
    print('**********************************************************************')