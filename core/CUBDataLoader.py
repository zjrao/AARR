# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 11:53:09 2019

@author: badat
"""
import os,sys
#import scipy.io as sio
import torch
import numpy as np
import h5py
import time
import pickle
import pdb
from sklearn import preprocessing
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
#%%
import scipy.io as sio
import pandas as pd
#%%
#import pdb
#%%

NFS_path = './'
img_dir = os.path.join(NFS_path,'data/CUB/')

data_transforms = transforms.Compose([
        transforms.Resize(448),
        transforms.CenterCrop(448),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([
    transforms.Resize(448),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class CUBTrainDataset(Dataset):
    def __init__(self, data_path, is_scale = False,is_unsupervised_attr = False,is_balance=True, transform=data_transforms):
        print(data_path)
        sys.path.append(data_path)

        self.data_path = data_path
        self.dataset = 'CUB'
        print('$' * 30)
        print(self.dataset)
        print('$' * 30)
        self.datadir = self.data_path + 'data/{}/'.format(self.dataset)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.is_scale = is_scale
        self.is_balance = is_balance
        if self.is_balance:
            print('Balance dataloader')
        self.is_unsupervised_attr = is_unsupervised_attr
        self.read_matdataset()
        self.get_idx_classes()
        self.transform = transform

    def __len__(self):
        return len(self.data['train_seen']['img_path'])

    def __getitem__(self, idx):
        image_file = self.data['train_seen']['img_path'][idx]

        image = Image.open(image_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = self.data['train_seen']['labels'][idx]
        att = self.att[label]
        return image, label, att

    def get_idx_classes(self):
        n_classes = self.seenclasses.size(0)
        self.idxs_list = []
        train_label = self.data['train_seen']['labels']
        for i in range(n_classes):
            idx_c = torch.nonzero(train_label == self.seenclasses[i].cpu()).cpu().numpy()
            idx_c = np.squeeze(idx_c)
            self.idxs_list.append(idx_c)
        return self.idxs_list

    def read_matdataset(self):

        path = self.datadir + 'feature_map_ResNet_101_{}.hdf5'.format(self.dataset)
        print('_____')
        print(path)
        # tic = time.time()
        hf = h5py.File(path, 'r')
        img_filepath = np.array(hf.get('feature_map'))
        img_filepath = np.array([str(a).split("'")[1] for a in img_filepath])

        #        shape = features.shape
        #        features = features.reshape(shape[0],shape[1],shape[2]*shape[3])
        #        pdb.set_trace()
        labels = np.array(hf.get('labels'))
        trainval_loc = np.array(hf.get('trainval_loc'))
        #        train_loc = np.array(hf.get('train_loc')) #--> train_feature = TRAIN SEEN
        #        val_unseen_loc = np.array(hf.get('val_unseen_loc')) #--> test_unseen_feature = TEST UNSEEN
        test_seen_loc = np.array(hf.get('test_seen_loc'))
        test_unseen_loc = np.array(hf.get('test_unseen_loc'))

        if self.is_unsupervised_attr:
            print('Unsupervised Attr')
            class_path = './w2v/{}_class.pkl'.format(self.dataset)
            with open(class_path, 'rb') as f:
                w2v_class = pickle.load(f)
            temp = np.array(hf.get('att'))
            print(w2v_class.shape, temp.shape)
            #            assert w2v_class.shape == temp.shape
            w2v_class = torch.tensor(w2v_class).float()

            U, s, V = torch.svd(w2v_class)
            reconstruct = torch.mm(torch.mm(U, torch.diag(s)), torch.transpose(V, 1, 0))
            print('sanity check: {}'.format(torch.norm(reconstruct - w2v_class).item()))

            print('shape U:{} V:{}'.format(U.size(), V.size()))
            print('s: {}'.format(s))

            self.w2v_att = torch.transpose(V, 1, 0)
            self.att = torch.mm(U, torch.diag(s))
            self.normalize_att = torch.mm(U, torch.diag(s))

        else:
            print('Expert Attr')
            att = np.array(hf.get('att'))
            self.att = torch.from_numpy(att).float()

            original_att = np.array(hf.get('original_att'))
            self.original_att = torch.from_numpy(original_att).float()

            w2v_att = np.array(hf.get('w2v_att'))
            self.w2v_att = torch.from_numpy(w2v_att).float()

            self.normalize_att = self.original_att / 100

        # print('Finish loading data in ',time.time()-tic)

        train_path = img_filepath[trainval_loc]
        test_seen_path = img_filepath[test_seen_loc]
        test_unseen_path = img_filepath[test_unseen_loc]
        if self.is_scale:
            scaler = preprocessing.MinMaxScaler()

        train_label = torch.from_numpy(labels[trainval_loc]).long()  # .to(self.device)
        test_unseen_label = torch.from_numpy(labels[test_unseen_loc])  # .long().to(self.device)
        test_seen_label = torch.from_numpy(labels[test_seen_loc])  # .long().to(self.device)

        self.seenclasses = torch.from_numpy(np.unique(train_label.cpu().numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(test_unseen_label.cpu().numpy()))
        self.ntrain = train_path.size
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        #        self.train_mapped_label = map_label(train_label, self.seenclasses)

        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['img_path'] = train_path
        self.data['train_seen']['labels'] = train_label

class CUBSeenTestDataset(Dataset):
    def __init__(self, data_path, is_scale = False,is_unsupervised_attr = False,is_balance=True, transform=test_transforms):
        print(data_path)
        sys.path.append(data_path)

        self.data_path = data_path
        self.dataset = 'CUB'
        print('$' * 30)
        print(self.dataset)
        print('$' * 30)
        self.datadir = self.data_path + 'data/{}/'.format(self.dataset)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.is_scale = is_scale
        self.is_balance = is_balance
        if self.is_balance:
            print('Balance dataloader')
        self.is_unsupervised_attr = is_unsupervised_attr
        self.read_matdataset()
        self.get_idx_classes()
        self.transform = transform

    def __len__(self):
        return len(self.data['test_seen']['img_path'])

    def __getitem__(self, idx):
        image_file = self.data['test_seen']['img_path'][idx]

        image = Image.open(image_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = self.data['test_seen']['labels'][idx]
        return image, label

    def get_idx_classes(self):
        n_classes = self.seenclasses.size(0)
        self.idxs_list = []
        train_label = self.data['test_seen']['labels']
        for i in range(n_classes):
            idx_c = torch.nonzero(train_label == self.seenclasses[i].cpu()).cpu().numpy()
            idx_c = np.squeeze(idx_c)
            self.idxs_list.append(idx_c)
        return self.idxs_list

    def read_matdataset(self):

        path = self.datadir + 'feature_map_ResNet_101_{}.hdf5'.format(self.dataset)
        print('_____')
        print(path)
        # tic = time.time()
        hf = h5py.File(path, 'r')
        img_filepath = np.array(hf.get('feature_map'))
        img_filepath = np.array([str(a).split("'")[1] for a in img_filepath])

        #        shape = features.shape
        #        features = features.reshape(shape[0],shape[1],shape[2]*shape[3])
        #        pdb.set_trace()
        labels = np.array(hf.get('labels'))
        trainval_loc = np.array(hf.get('trainval_loc'))
        #        train_loc = np.array(hf.get('train_loc')) #--> train_feature = TRAIN SEEN
        #        val_unseen_loc = np.array(hf.get('val_unseen_loc')) #--> test_unseen_feature = TEST UNSEEN
        test_seen_loc = np.array(hf.get('test_seen_loc'))
        test_unseen_loc = np.array(hf.get('test_unseen_loc'))

        if self.is_unsupervised_attr:
            print('Unsupervised Attr')
            class_path = './w2v/{}_class.pkl'.format(self.dataset)
            with open(class_path, 'rb') as f:
                w2v_class = pickle.load(f)
            temp = np.array(hf.get('att'))
            print(w2v_class.shape, temp.shape)
            #            assert w2v_class.shape == temp.shape
            w2v_class = torch.tensor(w2v_class).float()

            U, s, V = torch.svd(w2v_class)
            reconstruct = torch.mm(torch.mm(U, torch.diag(s)), torch.transpose(V, 1, 0))
            print('sanity check: {}'.format(torch.norm(reconstruct - w2v_class).item()))

            print('shape U:{} V:{}'.format(U.size(), V.size()))
            print('s: {}'.format(s))

            self.w2v_att = torch.transpose(V, 1, 0)
            self.att = torch.mm(U, torch.diag(s))
            self.normalize_att = torch.mm(U, torch.diag(s))

        else:
            print('Expert Attr')
            att = np.array(hf.get('att'))
            self.att = torch.from_numpy(att).float()

            original_att = np.array(hf.get('original_att'))
            self.original_att = torch.from_numpy(original_att).float()

            w2v_att = np.array(hf.get('w2v_att'))
            self.w2v_att = torch.from_numpy(w2v_att).float()

            self.normalize_att = self.original_att / 100

        # print('Finish loading data in ',time.time()-tic)

        train_path = img_filepath[trainval_loc]
        test_seen_path = img_filepath[test_seen_loc]
        test_unseen_path = img_filepath[test_unseen_loc]
        if self.is_scale:
            scaler = preprocessing.MinMaxScaler()

        train_label = torch.from_numpy(labels[trainval_loc]).long()  # .to(self.device)
        test_unseen_label = torch.from_numpy(labels[test_unseen_loc])  # .long().to(self.device)
        test_seen_label = torch.from_numpy(labels[test_seen_loc])  # .long().to(self.device)

        self.seenclasses = torch.from_numpy(np.unique(train_label.cpu().numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(test_unseen_label.cpu().numpy()))
        self.ntrain = train_path.size
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        #        self.train_mapped_label = map_label(train_label, self.seenclasses)

        self.data = {}

        self.data['test_seen'] = {}
        self.data['test_seen']['img_path'] = test_seen_path
        self.data['test_seen']['labels'] = test_seen_label
class CUBUnseenTestDataset(Dataset):
    def __init__(self, data_path, is_scale = False,is_unsupervised_attr = False,is_balance=True, transform=test_transforms):
        print(data_path)
        sys.path.append(data_path)

        self.data_path = data_path
        self.dataset = 'CUB'
        print('$' * 30)
        print(self.dataset)
        print('$' * 30)
        self.datadir = self.data_path + 'data/{}/'.format(self.dataset)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.is_scale = is_scale
        self.is_balance = is_balance
        if self.is_balance:
            print('Balance dataloader')
        self.is_unsupervised_attr = is_unsupervised_attr
        self.read_matdataset()
        self.get_idx_classes()
        self.transform = transform

    def __len__(self):
        return len(self.data['test_unseen']['img_path'])

    def __getitem__(self, idx):
        image_file = self.data['test_unseen']['img_path'][idx]

        image = Image.open(image_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = self.data['test_unseen']['labels'][idx]
        return image, label

    def get_idx_classes(self):
        n_classes = self.seenclasses.size(0)
        self.idxs_list = []
        train_label = self.data['test_unseen']['labels']
        for i in range(n_classes):
            idx_c = torch.nonzero(train_label == self.seenclasses[i].cpu()).cpu().numpy()
            idx_c = np.squeeze(idx_c)
            self.idxs_list.append(idx_c)
        return self.idxs_list

    def read_matdataset(self):

        path = self.datadir + 'feature_map_ResNet_101_{}.hdf5'.format(self.dataset)
        print('_____')
        print(path)
        # tic = time.time()
        hf = h5py.File(path, 'r')
        img_filepath = np.array(hf.get('feature_map'))
        img_filepath = np.array([str(a).split("'")[1] for a in img_filepath])

        #        shape = features.shape
        #        features = features.reshape(shape[0],shape[1],shape[2]*shape[3])
        #        pdb.set_trace()
        labels = np.array(hf.get('labels'))
        trainval_loc = np.array(hf.get('trainval_loc'))
        #        train_loc = np.array(hf.get('train_loc')) #--> train_feature = TRAIN SEEN
        #        val_unseen_loc = np.array(hf.get('val_unseen_loc')) #--> test_unseen_feature = TEST UNSEEN
        test_seen_loc = np.array(hf.get('test_seen_loc'))
        test_unseen_loc = np.array(hf.get('test_unseen_loc'))

        if self.is_unsupervised_attr:
            print('Unsupervised Attr')
            class_path = './w2v/{}_class.pkl'.format(self.dataset)
            with open(class_path, 'rb') as f:
                w2v_class = pickle.load(f)
            temp = np.array(hf.get('att'))
            print(w2v_class.shape, temp.shape)
            #            assert w2v_class.shape == temp.shape
            w2v_class = torch.tensor(w2v_class).float()

            U, s, V = torch.svd(w2v_class)
            reconstruct = torch.mm(torch.mm(U, torch.diag(s)), torch.transpose(V, 1, 0))
            print('sanity check: {}'.format(torch.norm(reconstruct - w2v_class).item()))

            print('shape U:{} V:{}'.format(U.size(), V.size()))
            print('s: {}'.format(s))

            self.w2v_att = torch.transpose(V, 1, 0)
            self.att = torch.mm(U, torch.diag(s))
            self.normalize_att = torch.mm(U, torch.diag(s))

        else:
            print('Expert Attr')
            att = np.array(hf.get('att'))
            self.att = torch.from_numpy(att).float()

            original_att = np.array(hf.get('original_att'))
            self.original_att = torch.from_numpy(original_att).float()

            w2v_att = np.array(hf.get('w2v_att'))
            self.w2v_att = torch.from_numpy(w2v_att).float()

            self.normalize_att = self.original_att / 100

        # print('Finish loading data in ',time.time()-tic)

        train_path = img_filepath[trainval_loc]
        test_seen_path = img_filepath[test_seen_loc]
        test_unseen_path = img_filepath[test_unseen_loc]
        if self.is_scale:
            scaler = preprocessing.MinMaxScaler()

        train_label = torch.from_numpy(labels[trainval_loc]).long()  # .to(self.device)
        test_unseen_label = torch.from_numpy(labels[test_unseen_loc])  # .long().to(self.device)
        test_seen_label = torch.from_numpy(labels[test_seen_loc])  # .long().to(self.device)

        self.seenclasses = torch.from_numpy(np.unique(train_label.cpu().numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(test_unseen_label.cpu().numpy()))
        self.ntrain = train_path.size
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        #        self.train_mapped_label = map_label(train_label, self.seenclasses)

        self.data = {}
        self.data['test_unseen'] = {}
        self.data['test_unseen']['img_path'] = test_unseen_path
        self.data['test_unseen']['labels'] = test_unseen_label