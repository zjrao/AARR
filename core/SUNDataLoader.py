#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:11:40 2019

@author: war-machince
"""

import os, sys
# import scipy.io as sio
import torch
import numpy as np
import h5py
import time
import pickle
from sklearn import preprocessing
# %%
import pdb
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


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


class SUNTrainDataLoader(Dataset):
    def __init__(self, data_path, is_balance=False, transforms = data_transforms):

        print(data_path)
        sys.path.append(data_path)

        self.data_path = data_path
        self.dataset = 'SUN'
        print('$' * 30)
        print(self.dataset)
        print('$' * 30)
        self.datadir = self.data_path + 'data/{}/'.format(self.dataset)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.is_balance = is_balance
        if self.is_balance:
            print('Balance dataloader')
        self.read_matdataset()
        self.get_idx_classes()
        self.I = torch.eye(self.allclasses.size(0))
        self.transform = transforms

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
        features = np.array(hf.get('feature_map'))
        #        shape = features.shape
        #        features = features.reshape(shape[0],shape[1],shape[2]*shape[3])
        labels = np.array(hf.get('labels'))
        trainval_loc = np.array(hf.get('trainval_loc'))
        #        train_loc = np.array(hf.get('train_loc')) #--> train_feature = TRAIN SEEN
        #        val_unseen_loc = np.array(hf.get('val_unseen_loc')) #--> test_unseen_feature = TEST UNSEEN
        test_seen_loc = np.array(hf.get('test_seen_loc'))
        test_unseen_loc = np.array(hf.get('test_unseen_loc'))


        print('Expert Attr')
        att = np.array(hf.get('att'))
        self.att = torch.from_numpy(att).float()

        original_att = np.array(hf.get('original_att'))
        self.original_att = torch.from_numpy(original_att).float()

        w2v_att = np.array(hf.get('w2v_att'))
        self.w2v_att = torch.from_numpy(w2v_att).float()

        self.normalize_att = self.original_att / 100

        # print('Finish loading data in ',time.time()-tic)

        train_feature = features[trainval_loc]

        train_label = torch.from_numpy(labels[trainval_loc]).long()  # .to(self.device)
        test_unseen_label = torch.from_numpy(labels[test_unseen_loc])  # .long().to(self.device)

        self.seenclasses = torch.from_numpy(np.unique(train_label.cpu().numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(test_unseen_label.cpu().numpy()))
        self.ntrain = train_feature.size
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['img_path'] = train_feature
        self.data['train_seen']['labels'] = train_label

class SUNSeenTestDataLoader(Dataset):
    def __init__(self, data_path, is_balance=True, transforms = test_transforms):

        print(data_path)
        sys.path.append(data_path)

        self.data_path = data_path
        self.dataset = 'SUN'
        print('$' * 30)
        print(self.dataset)
        print('$' * 30)
        self.datadir = self.data_path + 'data/{}/'.format(self.dataset)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.is_balance = is_balance
        if self.is_balance:
            print('Balance dataloader')
        self.read_matdataset()
        self.I = torch.eye(self.allclasses.size(0))
        self.transform = transforms

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

    def read_matdataset(self):

        path = self.datadir + 'feature_map_ResNet_101_{}.hdf5'.format(self.dataset)
        print('_____')
        print(path)
        # tic = time.time()
        hf = h5py.File(path, 'r')
        features = np.array(hf.get('feature_map'))
        #        shape = features.shape
        #        features = features.reshape(shape[0],shape[1],shape[2]*shape[3])
        labels = np.array(hf.get('labels'))
        trainval_loc = np.array(hf.get('trainval_loc'))
        #        train_loc = np.array(hf.get('train_loc')) #--> train_feature = TRAIN SEEN
        #        val_unseen_loc = np.array(hf.get('val_unseen_loc')) #--> test_unseen_feature = TEST UNSEEN
        test_seen_loc = np.array(hf.get('test_seen_loc'))
        test_unseen_loc = np.array(hf.get('test_unseen_loc'))

        print('Expert Attr')
        att = np.array(hf.get('att'))
        self.att = torch.from_numpy(att).float()

        original_att = np.array(hf.get('original_att'))
        self.original_att = torch.from_numpy(original_att).float()

        w2v_att = np.array(hf.get('w2v_att'))
        self.w2v_att = torch.from_numpy(w2v_att).float()

        self.normalize_att = self.original_att / 100

        # print('Finish loading data in ',time.time()-tic)

        train_feature = features[trainval_loc]
        test_seen_feature = features[test_seen_loc]

        train_label = torch.from_numpy(labels[trainval_loc]).long()  # .to(self.device)
        test_unseen_label = torch.from_numpy(labels[test_unseen_loc])  # .long().to(self.device)
        test_seen_label = torch.from_numpy(labels[test_seen_loc])  # .long().to(self.device)

        self.seenclasses = torch.from_numpy(np.unique(train_label.cpu().numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(test_unseen_label.cpu().numpy()))
        self.ntrain = train_feature.size
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        #        self.train_mapped_label = map_label(train_label, self.seenclasses)

        self.data = {}
        self.data['test_seen'] = {}
        self.data['test_seen']['img_path'] = test_seen_feature
        self.data['test_seen']['labels'] = test_seen_label

class SUNUnSeenTestDataLoader(Dataset):
    def __init__(self, data_path, is_balance=True, transforms = test_transforms):

        print(data_path)
        sys.path.append(data_path)

        self.data_path = data_path
        self.dataset = 'SUN'
        print('$' * 30)
        print(self.dataset)
        print('$' * 30)
        self.datadir = self.data_path + 'data/{}/'.format(self.dataset)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.is_balance = is_balance
        if self.is_balance:
            print('Balance dataloader')
        self.read_matdataset()
        self.I = torch.eye(self.allclasses.size(0))
        self.transform = transforms

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

    def read_matdataset(self):

        path = self.datadir + 'feature_map_ResNet_101_{}.hdf5'.format(self.dataset)
        print('_____')
        print(path)
        # tic = time.time()
        hf = h5py.File(path, 'r')
        features = np.array(hf.get('feature_map'))
        #        shape = features.shape
        #        features = features.reshape(shape[0],shape[1],shape[2]*shape[3])
        labels = np.array(hf.get('labels'))
        trainval_loc = np.array(hf.get('trainval_loc'))
        #        train_loc = np.array(hf.get('train_loc')) #--> train_feature = TRAIN SEEN
        #        val_unseen_loc = np.array(hf.get('val_unseen_loc')) #--> test_unseen_feature = TEST UNSEEN
        test_seen_loc = np.array(hf.get('test_seen_loc'))
        test_unseen_loc = np.array(hf.get('test_unseen_loc'))

        print('Expert Attr')
        att = np.array(hf.get('att'))
        self.att = torch.from_numpy(att).float()

        original_att = np.array(hf.get('original_att'))
        self.original_att = torch.from_numpy(original_att).float()

        w2v_att = np.array(hf.get('w2v_att'))
        self.w2v_att = torch.from_numpy(w2v_att).float()

        self.normalize_att = self.original_att / 100

        # print('Finish loading data in ',time.time()-tic)

        train_feature = features[trainval_loc]
        test_unseen_feature = features[test_unseen_loc]

        train_label = torch.from_numpy(labels[trainval_loc]).long()  # .to(self.device)
        test_unseen_label = torch.from_numpy(labels[test_unseen_loc])  # .long().to(self.device)
        test_seen_label = torch.from_numpy(labels[test_seen_loc])  # .long().to(self.device)

        self.seenclasses = torch.from_numpy(np.unique(train_label.cpu().numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(test_unseen_label.cpu().numpy()))
        self.ntrain = train_feature.size
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        #        self.train_mapped_label = map_label(train_label, self.seenclasses)

        self.data = {}
        self.data['test_unseen'] = {}
        self.data['test_unseen']['img_path'] = test_unseen_feature
        self.data['test_unseen']['labels'] = test_unseen_label