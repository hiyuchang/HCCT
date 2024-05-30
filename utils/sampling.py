#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
import scipy
from torchvision import datasets, transforms
import copy
from math import floor

def mnist_iid(dataset, num_users, train_ratio=0.8):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users = {}
    num_items = int(len(dataset) / num_users)
    all_idxs = [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_noniid(dataset, num_users, n_class_per=2, train_ratio=0.8):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {}
    num_shards, num_imgs = num_users * n_class_per, int(len(dataset) / (num_users * n_class_per))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_tr = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_val = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, n_class_per, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    
    # split some for validation
    for i in range(num_users):
        np.random.shuffle(dict_users[i])
        dict_users_tr[i] = dict_users[i][0:int(len(dict_users[i]) * train_ratio)]
        dict_users_val[i] = dict_users[i][int(len(dict_users[i]) * train_ratio):]
        
    return dict_users_tr, dict_users_val

def fashion_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users = {}
    num_items = int(len(dataset) / num_users)
    all_idxs = [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def fashion_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = num_users * 2, int(len(dataset) / (num_users * 2))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_tr = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_val = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    
    # split some for validation
    for i in range(num_users):
        np.random.shuffle(dict_users[i])
        dict_users_tr[i] = dict_users[i][0:int(len(dict_users[i]) * train_ratio)]
        dict_users_val[i] = dict_users[i][int(len(dict_users[i]) * train_ratio):]
    return dict_users_tr, dict_users_val

def cifar_iid(dataset, num_users, train_ratio=0.8, num_samples=0):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users = {}
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    dict_users_tr = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_val = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    for i in range(num_users):
        dict_users[i] = np.random.choice(all_idxs, num_items, replace=False)
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    
    # split some for validation
    for i in range(num_users):
        np.random.shuffle(dict_users[i])
        if num_samples > 0:
            dict_users_tr[i] = dict_users[i][0:num_samples]
            dict_users_val[i] = dict_users[i][num_samples:]
        else:
            dict_users_tr[i] = dict_users[i][0:int(len(dict_users[i]) * train_ratio)]
            dict_users_val[i] = dict_users[i][int(len(dict_users[i]) * train_ratio):]
    
    return dict_users_tr, dict_users_val

def cifar_noniid(dataset, num_users, n_class_per=2, train_ratio=0.8, num_samples=0):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = num_users * n_class_per, int(len(dataset) / (num_users * n_class_per))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_tr = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_val = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)
    if len(idxs) != len(labels):
        labels = labels[: len(idxs)]
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, n_class_per, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)          
    
    # split some for validation
    for i in range(num_users):
        np.random.shuffle(dict_users[i])
        if num_samples > 0:
            dict_users_tr[i] = dict_users[i][0:num_samples]
            dict_users_val[i] = dict_users[i][num_samples:]
        else:
            dict_users_tr[i] = dict_users[i][0:int(len(dict_users[i]) * train_ratio)]
            dict_users_val[i] = dict_users[i][int(len(dict_users[i]) * train_ratio):]
    
    return dict_users_tr, dict_users_val

def count_data(args, dict_users, y_train):
    
    args.data_num_counts = {}
    y_train = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train
    for idx, dataidx in dict_users.items():
        # dataidx = np.array(dataidx, dtype=np.int64)
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        args.data_num_counts[idx] = tmp
    
    print('Data statistics: %s' % str(args.data_num_counts))
    return args

def imbalanced(dataset, num_users, imb_type, imb_var, train_ratio=0.8):
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_tr = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_val = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    num_shards, num_imgs = 100, int(len(dataset) / 100)
    idx_shard = [i for i in range(num_shards)]
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)
    
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    
    # determine the number of shards
    if imb_type == 'normal':
        rand = scipy.stats.norm.rvs(loc=0.0, scale=imb_var, size=num_users)
    elif imb_type == 'halfnormal':
        rand = scipy.stats.halfnorm.rvs(loc=0.0, scale=imb_var, size=num_users)
    
    rand_ratio = [x/sum(rand) for x in rand]
    num_shard_per = [floor(r * num_shards) for r in rand_ratio]  
    # while sum(num_shard_per) > num_shards:
    for i in range(len(num_shard_per)):
        if num_shard_per[i] == 0:
            num_shard_per[i] += 1
        
    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, num_shard_per[i], replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)          

    print("Total samples of each user ", [len(x) for x in dict_users.values()])  
    
    # split some for validation
    for i in range(num_users):
        np.random.shuffle(dict_users[i])
        dict_users_tr[i] = dict_users[i][0:int(len(dict_users[i]) * train_ratio)]
        dict_users_val[i] = dict_users[i][int(len(dict_users[i]) * train_ratio):]
        
    return dict_users_tr, dict_users_val

def imbalanced_glb(dataset, num_users, imb_type, imb_var, glb_imb_var, train_ratio=0.8):
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_tr = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_val = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    # num_shards, num_imgs = 100, int(len(dataset) / 100)
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))
    
    # global imbalance
    if imb_type == 'halfnormal':
        num_rand_list = scipy.stats.halfnorm.rvs(loc=0.0, scale=glb_imb_var, size=num_classes)
    num_rand_list = [int(x * 100) for x in num_rand_list]
    
    glb_sample_per_class = int( int(len(dataset)/num_classes) / max(num_rand_list) )
    print("glb_sample_per_class", glb_sample_per_class)
    
    idxs_new = np.array([], dtype='int64')
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        idxs_new = np.append(idxs_new, np.random.choice(idx_k, glb_sample_per_class * num_rand_list[k], replace=False))
    # determine the number of samples (IID)
    if imb_type == 'halfnormal':
        rand = scipy.stats.halfnorm.rvs(loc=0.0, scale=imb_var, size=num_users)
    rand_ratio = [x/sum(rand) for x in rand]
    num_samples = [int(r * len(dataset)) for r in rand_ratio]    
        
    # divide and assign
    for i in range(num_users):
        dict_users[i] = np.random.choice(idxs_new, num_samples[i], replace=False)
        
    print("Total samples of each user ", [len(x) for x in dict_users.values()])  
    
    # split some for validation
    for i in range(num_users):
        np.random.shuffle(dict_users[i])
        dict_users_tr[i] = dict_users[i][0:int(len(dict_users[i]) * train_ratio)]
        dict_users_val[i] = dict_users[i][int(len(dict_users[i]) * train_ratio):]
        
    return dict_users_tr, dict_users_val

def imbalanced_iid(dataset, num_users, imb_type, imb_var, train_ratio=0.8):
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_tr = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_val = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    # determine the number of samples
    if imb_type == 'halfnormal':
        rand = scipy.stats.halfnorm.rvs(loc=0.0, scale=imb_var, size=num_users)
    rand_ratio = [x/sum(rand) for x in rand]
    num_samples = [int(r * len(dataset)) for r in rand_ratio]    
    print(num_samples)
    
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    dict_users_tr = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_val = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    for i in range(num_users):
        dict_users[i] = np.random.choice(all_idxs, num_samples[i], replace=False)
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    
    # split some for validation
    for i in range(num_users):
        np.random.shuffle(dict_users[i])
        dict_users_tr[i] = dict_users[i][0:int(len(dict_users[i]) * train_ratio)]
        dict_users_val[i] = dict_users[i][int(len(dict_users[i]) * train_ratio):]
    
    return dict_users_tr, dict_users_val
    
def dirichlet(dataset, num_users, beta, train_ratio=0.8, num_samples=0):
    print("Use dirichlet distribution")
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_tr = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_val = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    min_size = 0
    min_require_size = 10
    K = 10  # classes
    if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
        K = 2
        # min_require_size = 100
    elif dataset == 'cifar100':
        K = 100

    N = len(dataset)
    labels = np.array(dataset.targets)

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, num_users))
            proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = idx_batch[j]
    
    # split some for validation
    for i in range(num_users):
        np.random.shuffle(dict_users[i])
        if num_samples > 0:
            dict_users_tr[i] = dict_users[i][0:num_samples]
            dict_users_val[i] = dict_users[i][num_samples:]
        else:
            dict_users_tr[i] = dict_users[i][0:int(len(dict_users[i]) * train_ratio)]
            dict_users_val[i] = dict_users[i][int(len(dict_users[i]) * train_ratio):]
    
    return dict_users_tr, dict_users_val
    