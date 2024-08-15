import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
import numpy as np
import torch.optim as optim

class BanlanceSampler(Sampler):
    def __init__(self, idx_list, batch_size):
        self.idxs_list = idx_list
        self.batch_size = batch_size
        self.n_class = len(idx_list)
        self.complen()
        self.niter = int(self.length / batch_size)
        self.idx = self.compidx()

    def complen(self):
        self.length = 0
        for i in range(self.n_class):
            self.length += len(self.idxs_list[i])

    def compidx(self):
        idx = []
        for i in range(self.niter):
            sampled_idx_c = np.random.choice(np.arange(self.n_class), self.batch_size, replace=False).tolist()
            for i_c in sampled_idx_c:
                idxs = self.idxs_list[i_c]
                idx.append(np.random.choice(idxs, 1))
        idx = np.concatenate(idx)
        idx = torch.from_numpy(idx)

        return idx


    def __iter__(self):

        return iter(self.idx)

    def __len__(self):
        return self.idx.size(0)

def warm_update_teacher(model, teacher, momentum=0.9995, global_step=2000):
    momentum = min(1 - 1 / (global_step + 1), momentum)
    for ema_param, param in zip(teacher.parameters(), model.parameters()):
        ema_param.data = ema_param.data * momentum + (1 - momentum) * param.data

def warm_update(gradmodel, model, momentum=0.9995):
    gradmodel.V.data = gradmodel.V.data * momentum + model.V.data.clone().detach() * (1 - momentum)
    gradmodel.W_1.data = gradmodel.W_1.data * momentum + model.W_1.data.clone().detach() * (1 - momentum)
    gradmodel.W_2.data = gradmodel.W_2.data * momentum + model.W_2.data.clone().detach() * (1 - momentum)

def computeSimi(att, seen, unseen, selectK):
    num_seen, num_unseen = seen.size(0), unseen.size(0)
    if selectK > num_seen:
        selectK = num_seen

    att_seen, att_unseen = att[seen], att[unseen]

    att_unseen = att_unseen.unsqueeze(1).repeat(1, att_seen.size(0), 1)
    att_seen = att_seen.unsqueeze(0).repeat(att_unseen.size(0), 1, 1)
    simiMetric = ((att_unseen - att_seen) * (att_unseen - att_seen)).sum(2)

    _, index = torch.sort(simiMetric, dim=1)
    index = index[:, 0:selectK]

    n = att.size(0)
    label_map = torch.zeros((n, n))

    for i in range(num_unseen):
        y = int(unseen[i])
        for j in range(selectK):
            x = int(seen[int(index[i][j])])
            label_map[x][y] = 1

    return label_map.long()

def compute_loss(output, label):
    log_softmax_func = nn.LogSoftmax(dim=1)
    Prob = log_softmax_func(output)

    loss = -torch.einsum('bk,bk->b', Prob, label)
    loss = torch.mean(loss)
    return loss

def compute_loss2(output, label, seen_classes, n):
    labelm = torch.eye(n).cuda()
    labelm = labelm[label]
    labelm = labelm[:, seen_classes]
    output = output[:, seen_classes]

    log_softmax_func = nn.LogSoftmax(dim=1)
    Prob = log_softmax_func(output)

    loss = -torch.einsum('bk,bk->b', Prob, labelm)
    loss = torch.mean(loss)
    return loss

class Gradmap(nn.Module):
    def __init__(self, label_map, lamda, mode, n=200, attsize=312, dim_v=300, dim_f=2048, uniform_att_2=False):
        super(Gradmap, self).__init__()
        self.V = nn.Parameter(nn.init.normal_(torch.empty(attsize, dim_v)), requires_grad=False)
        self.att = nn.Parameter(torch.empty((n, attsize)), requires_grad=False)
        self.W_1 = nn.Parameter(nn.init.normal_(torch.empty(dim_v, dim_f)),requires_grad=False)
        self.W_2 = nn.Parameter(nn.init.zeros_(torch.empty(dim_v, dim_f)), requires_grad=False)
        self.weight_ce = nn.Parameter(torch.eye(n).float(), requires_grad=False)
        self.label_map = label_map
        self.lamda = lamda
        self.mode = mode
        self.uniform_att_2 = uniform_att_2
        self.dim_att = attsize

    def forward(self, Fs):
        S = torch.einsum('iv,vf,bfr->bir', self.V, self.W_1, Fs)  # batchx312x49
        A = torch.einsum('iv,vf,bfr->bir', self.V, self.W_2, Fs)  # batchx312x49
        A = F.softmax(A, dim=-1)  # compute an attention map for each attribute
        S_p = torch.einsum('bir,bir->bi', A, S)  # compute attribute scores from attribute attention maps

        if self.uniform_att_2:
            A_b_p = self.att.new_full((Fs.size(0), self.dim_att), fill_value=1)
            S_pp = torch.einsum('ki,bi,bi->bik', self.att, A_b_p, S_p)
        else:
            S_pp = torch.einsum('ki,bi->bik', self.att, S_p)

        S_pp = torch.sum(S_pp, axis=1)  # [bk] <== [bik]
        return S_pp, A

    def gradtoMap(self, grad):

        grad = torch.abs(grad)
        grad = grad.sum(dim=2)
        m, _ = grad.max(dim=1)
        m = m.unsqueeze(-1)
        grad /= (m + 1e-6)

        return grad

    def gradtoMap2(self, grad):

        grad = torch.abs(grad)
        m, _ = grad.max(dim=2)
        m = m.unsqueeze(-1)
        grad /= (m + 1e-6)

        return grad

    def gradtoMap3(self, grad):
        grad = torch.abs(grad)
        grad = grad.sum(dim=1)
        m, _ = grad.max(dim=1)
        m = m.unsqueeze(-1)
        grad /= (m + 1e-6)

        return grad

    def gradtoMap4(self, grad):
        weight = torch.mean(grad, dim=(2)).unsqueeze(-1)
        grad = grad * weight
        grad = torch.sum(grad, dim=1)
        grad = self.normali(grad)
        return grad

    def normali(self, grad):
        mmin, _ = grad.min(dim=1)
        mmax, _ = grad.max(dim=1)
        mmin, mmax = mmin.unsqueeze(-1), mmax.unsqueeze(-1)
        grad = (grad - mmin) / (mmax - mmin + 1e-6)
        return grad

    def computGrad(self, feature, label, isclip=False, clipThresh=0.5):
        shape = feature.shape
        feature = feature.reshape(shape[0], shape[1], shape[2] * shape[3])  # batch x 2048 x 49
        feature = F.normalize(feature, dim=1)

        Fs = nn.Parameter(feature.clone().detach(), requires_grad=True)

        S_pp, Score = self.forward(Fs)

        label1 = self.weight_ce[label].clone().detach()
        loss1 = compute_loss(S_pp, label1)
        label2 = self.label_map[label].clone().detach().float()
        loss2 = compute_loss(S_pp, label2)
        #loss = loss1 + loss2

        grad1 = torch.autograd.grad(inputs=Fs, outputs=loss1, retain_graph=True)[0]
        grad2 = torch.autograd.grad(inputs=Fs, outputs=loss2, retain_graph=False)[0]

        #if self.mode == 0:
        #    grad1, grad2 = self.gradtoMap(grad1), self.gradtoMap(grad2)
        #elif self.mode == 1:
        #    grad1, grad2 = self.gradtoMap2(grad1), self.gradtoMap2(grad2)
        #else:
        #    grad1, grad2 = self.gradtoMap3(grad1), self.gradtoMap3(grad2)

        grad1, grad2 = self.gradtoMap4(grad1), self.gradtoMap4(grad2)

        grad = grad1 + grad2
        Score = torch.mean(Score, dim=1)
        Score = F.softmax(Score, dim=1)
        grad = grad*(1+Score)
        grad = self.normali(grad)

        if isclip:
            grad = torch.where(grad < clipThresh, torch.zeros_like(grad), torch.ones_like(grad))

        return grad

    def mmse_loss(self, input, target, att_map):
        if  self.mode == 0:

            m, _ = att_map.max(dim=1)
            m = m.unsqueeze(-1)
            att_map /= (m + 1e-6)

            mse = torch.pow((input - target), 2).mean(dim=(2, 3))
            loss = mse * att_map

        elif self.mode == 1:

            m, _ = att_map.max(dim=2)
            m = m.unsqueeze(-1)
            att_map /= (m + 1e-6)

            mse = torch.pow((input - target), 2).view(input.size(0), input.size(1), -1)
            loss = mse * att_map

        else:

            m, _ = att_map.max(dim=1)
            m = m.unsqueeze(-1)
            att_map /= (m + 1e-6)

            mse = torch.pow((input - target), 2).mean(dim=1).view(input.size(0), -1)
            loss = mse * att_map

        return torch.mean(loss)

    def mmse_loss2(self, input, target, att_map):
        _loss = torch.pow((input - target), 2)
        _loss = torch.mean(_loss, dim=1).view(input.size(0), -1)
        loss = _loss * att_map
        return torch.mean(loss)


class computeProto(nn.Module):
    def __init__(self, num_classes, num_att, dim_f, num_sample, att, device, model, dataset):
        super(computeProto, self).__init__()
        self.device = device
        self.classes = num_classes
        self.attsize = num_att
        self.dim_f = dim_f
        self.num_sample = num_sample
        self.att = nn.Parameter(F.normalize(torch.tensor(att)), requires_grad=False)

        self.Prototype = torch.zeros(self.attsize, self.dim_f).to(device)
        self.score = torch.zeros(self.attsize, 1).to(device)
        self.register_buffer('_prototypes', self.Prototype)
        self.register_buffer('_scores', self.score)

        self.W = nn.Parameter(nn.init.normal_(torch.empty(self.dim_f, self.classes)), requires_grad=True)
        self.log_softmax_func = nn.LogSoftmax(dim=1)
        self.weight_ce = nn.Parameter(torch.eye(self.classes).float(), requires_grad=False)

        self.computeproto(model, dataset)
        self._Prototype = nn.Parameter(self.Prototype.clone(), requires_grad=True)
        self.lamda = nn.Parameter(torch.tensor([0.5]), requires_grad=True)


    def pretrainW(self, iters=200000):
        print("============start pretrainW================")
        report_iter = int(iters/10)
        optimizer  = optim.RMSprop([self.W], lr=0.0001)
        _prototype = self.Prototype.clone().detach()
        for iter in range(iters):
            optimizer.zero_grad()
            prototype = _prototype + torch.randn_like(_prototype)*0.001

            prob = torch.einsum('na, af, fb->nb', self.att, prototype,
                                self.W)

            loss = self.compute_loss(prob)
            loss.backward()
            optimizer.step()
            if iter % report_iter == 0:
                print("loss = %.3f" % (loss.item()))

        print("=================finish=======================")

    def computeproto(self, model, dataset):
        self.Prototype = self.Prototype * 0
        self.score = self.score * 0
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
        model.eval()
        print("-------------start---------------------")
        with torch.no_grad():
            for iter, (img, label, att) in enumerate(dataset_loader):

                img, label, att = img.to(self.device), label.to(self.device), att.to(self.device)

                out_package, _= model(img)

                feature, score = out_package['Fs_m'], out_package['A']
                batchsize = feature.size(0)
                dim_f = feature.size(1)
                region = feature.size(2) * feature.size(3)
                attsize = score.size(1)

                feature = feature.reshape(batchsize, dim_f, region)
                feature = F.normalize(feature, dim=1)

                score, index_score = score.max(dim=2)

                feature = feature.transpose(1, 2).reshape(batchsize * region, dim_f)
                m = torch.tensor(range(batchsize)) * region
                m = m.unsqueeze(-1).repeat(1, attsize).view(-1).to(self.device)
                index_score = index_score.view(-1) + m
                select_feature = feature[index_score]
                select_feature = select_feature.reshape(batchsize, attsize, -1)

                score = score.unsqueeze(-1)
                select_feature = select_feature * score

                score = score.sum(dim=0)
                select_feature = select_feature.sum(dim=0)

                self.Prototype = self.Prototype + select_feature
                self.score = self.score + score

                if iter%40 == 0:
                    print("============== %d ============" % (iter))

            self.Prototype = self.Prototype / (self.score + 1e-6)
            print("---------------------finish-------------------------")

    def forward(self, feature, score, label):
        batchsize = feature.size(0)
        dim_f = feature.size(1)
        region = feature.size(2) * feature.size(3)
        attsize = score.size(1)

        feature = feature.reshape(batchsize, dim_f, region)
        feature = F.normalize(feature, dim=1)

        protofeat = torch.einsum('bir, bkr->bki', feature, score)
        attscore = self.att[label].clone().detach()
        protofeat = torch.einsum('bkr, bk->bkr', protofeat, attscore)
        protofeat = torch.sum(protofeat, dim=0)
        attscore = torch.sum(attscore, dim=0).unsqueeze(-1)
        protofeat = protofeat / (attscore + 1e-9)

        #_prototype = (self.Prototype + protofeat) / (self.score + score + 1e-6)
        #self.lamda = torch.softmax(self.lamda, dim=0)

        _prototype = self._Prototype * self.lamda + protofeat * (1.0 - self.lamda)
        #_prototype = self._Prototype * 0.9 + protofeat * 0.1

        prob = torch.einsum('na, af, fb->nb', self.att, _prototype, self.W)  #class * attsize, attsize*dim_f, dim_f*class

        return self.compute_loss(prob)

    def compute_loss(self, prob):
        _Prob = self.log_softmax_func(prob)

        loss = -torch.einsum('bk,bk->b', _Prob, self.weight_ce)
        loss = torch.mean(loss)
        return loss





