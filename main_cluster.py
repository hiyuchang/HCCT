#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os

from utils.sampling import *
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import *
from models.Resnet import ResNet18_cifar10
from models.Fed import FedWeightAvg
from models.test import test_img
from utils.dataset import FEMNIST, ShakeSpeare
from utils.digit_utils import prepare_digit_data
from utils.keyfunc import user_cluster, ifca, maxfl, fedfa, flsc


def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    return


if __name__ == '__main__':

    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        
    set_all_seed(args.seed)

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(args.data_dir+'mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(args.data_dir+'mnist/', train=False, download=True, transform=trans_mnist)
        if args.imb_type is None:
            if args.iid:
                dict_users = mnist_iid(dataset_train, args.num_users, args.train_ratio)
            elif args.dirichlet:
                dict_users, dict_users_val = dirichlet(dataset_train, args.num_users, args.beta, args.train_ratio)
            else:
                dict_users, dict_users_val = mnist_noniid(dataset_train, args.num_users, args.train_ratio)
        else:
            if args.iid:
                dict_users, dict_users_val = imbalanced_iid(dataset_train, args.num_users, args.imb_type, args.imb_var, args.train_ratio)
            else:
                dict_users, dict_users_val = imbalanced(dataset_train, args.num_users, args.imb_type, args.imb_var, args.train_ratio)
    
    elif args.dataset == 'cifar':
        trans_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trans_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10(args.data_dir+'cifar10', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR10(args.data_dir+'cifar10', train=False, download=True, transform=trans_cifar_test)
        if args.imb_type is None:
            if args.iid:
                dict_users, dict_users_val = cifar_iid(dataset_train, args.num_users, args.train_ratio, args.num_samples)
            elif args.dirichlet:
                dict_users, dict_users_val = dirichlet(dataset_train, args.num_users, args.beta, args.train_ratio, args.num_samples)
            else:
                dict_users, dict_users_val = cifar_noniid(dataset_train, args.num_users, args.n_class_per, args.train_ratio, args.num_samples)
        else:
            # imbalance
            dict_users, dict_users_val = imbalanced(dataset_train, args.num_users, args.imb_type, args.imb_var, args.train_ratio)
            
    elif args.dataset == 'fmnist':
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST(args.data_dir+'fmnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        dataset_test  = datasets.FashionMNIST(args.data_dir+'fmnist', train=False, download=True,
                                              transform=trans_fashion_mnist)
        if args.imb_type is None:
            if args.iid:
                dict_users = mnist_iid(dataset_train, args.num_users, args.train_ratio)
            elif args.dirichlet:
                dict_users, dict_users_val = dirichlet(dataset_train, args.num_users, args.beta, args.train_ratio)
            else:
                dict_users, dict_users_val = mnist_noniid(dataset_train, args.num_users, args.train_ratio)
        else:
            dict_users, dict_users_val = imbalanced(dataset_train, args.num_users, args.imb_type, args.imb_var, args.train_ratio)
    
    elif args.dataset == 'digit':
        dict_users, dict_users_val = None, None
        train_loaders, test_loaders, ds_locals = prepare_digit_data(args)
        print(ds_locals)
    else:
        exit('Error: unrecognized dataset')
    
    strategy_text = args.strategy
    if args.personalize == 1:
        strategy_text += "(p)"
    if args.cluster_metric == "layer":
        strategy_text += "(e)"
    
    if args.dataset != 'digit':
        args = count_data(args, dict_users, dataset_train.targets)    
    
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar().to(args.device)
    elif args.model == 'cnn4' and args.dataset == 'cifar':
        net_glob = CNN_4().to(args.device)
    elif args.model == 'resnet18' and args.dataset == 'cifar':
        net_glob = ResNet18_cifar10().to(args.device)
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fmnist'):
        net_glob = CNNMnist().to(args.device)
    elif args.dataset == 'digit':
        net_glob = DigitModel().to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        img_size = dataset_train[0][0].shape
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    elif args.model == 'logistic':
        net_glob = Logistic(in_dim=np.prod((1, 28, 28)), out_dim=10).to(args.device)
    else:
        exit('Error: unrecognized model')
    # print(net_glob)
    net_glob.cpu()
    net_glob.train()
    net_local = copy.deepcopy(net_glob)
    w_glob = copy.deepcopy(net_glob.state_dict())

    # training
    acc_test = []
    if args.dataset == 'digit':
        clients = [LocalUpdate(client_id=idx, args=args, dataset=None, idxs=train_loaders[idx], idxs_val=test_loaders[idx])
                for idx in range(args.num_users)]
    else:
        ds_locals = [len(dict_users[idx]) for idx in range(args.num_users)]
        clients = [LocalUpdate(client_id=idx, args=args, dataset=dataset_train, idxs=dict_users[idx], idxs_val=dict_users_val[idx])
                for idx in range(args.num_users)]
    
    if args.strategy == "maxfl":
        for idx in range(args.num_users):
            set_all_seed(args.seed)
            net_glob.load_state_dict(w_glob)
            w, _ = clients[idx].train(net=copy.deepcopy(net_glob).to(args.device), user_idx=idx)
            clients[idx].w_prior = copy.deepcopy(w)
    
    # choose which layer for clustering
    if args.cluster_metric == "layer" or args.personalize:
        if args.dataset == "cifar":
            layer_values = {"conv1": 0.0, "conv2": 0.0, "conv3": 0.0, "conv4": 0.0, "fc1": 0.0, "fc2": 0.0}
        elif args.dataset == "fmnist":
            layer_values = {"layer_input": 0.0, "layer_hidden": 0.0}
        elif args.dataset == "digit":
            layer_values = {"conv1": 0.0, "conv2": 0.0, "conv3": 0.0, "fc1": 0.0, "fc2": 0.0, "fc3": 0.0}
        
        if args.cluster_metric == "layer":
            g_local_list = {key+".weight": [] for key in layer_values.keys()}
            g_local_list.update({key+".bias": [] for key in layer_values.keys()})
            for idx in range(args.num_users):
                w, loss = clients[idx].train(net=copy.deepcopy(net_glob).to(args.device))
                for key in g_local_list.keys():
                    g_local_list[key].append((w_glob[key].cpu().numpy() - w[key].cpu().numpy()) / clients[idx].lr)
                    g_local_list[key][-1].flatten()
            g_local_var = dict()
            for key in g_local_list.keys():
                g_local_var[key] = np.var(g_local_list[key]) / np.mean(g_local_list[key])
            # print(g_local_var)
            for layer in layer_values.keys():
                layer_values[layer] += g_local_var[layer+".weight"]
                layer_values[layer] += g_local_var[layer+".bias"]
            cluster_layer = max(zip(layer_values.values(), layer_values.keys()))[1]
            if args.goal in layer_values.keys():
                cluster_layer = args.goal
            print(layer_values)
            print(cluster_layer)
        last_layer = list(layer_values.keys())[-1]
    
    # reinitilization
    set_all_seed(args.seed)
    net_glob.load_state_dict(w_glob)
    w_globs = {i: copy.deepcopy(w_glob) for i in range(args.n_cluster)}
    w_locals = [None for _ in range(args.num_users)]
    g_locals = [copy.deepcopy(w_glob) for _ in range(args.num_users)] 
    acc_best_dict = {i: 0.0 for i in range(args.num_users)}
    user_utility_avg = {i: 0.0 for i in range(args.num_users)}
    
    g_tmp = torch.cat([tensor.flatten() for tensor in w_glob.values()], dim=0)
    args.model_dim = np.prod(g_tmp.size())
    del g_tmp
    
    for iter in range(args.epochs):
        # new incoming users
        if args.goal == "n_new":
            # NOTE only for cifar 10, 15, 20
            if iter == 0:
                args.num_users = 10
                # corresponding numbers
                g_locals = [copy.deepcopy(w_glob) for _ in range(args.num_users)]
                user_utility_avg = {i: 0.0 for i in range(args.num_users)}
            elif iter % 10 == 0:
                if args.num_users < 20:
                    # add a user to get its g radient g_locals
                    net_glob.load_state_dict(w_glob)
                    for n in range(args.n_new):
                        idx = args.num_users + n
                        g_locals.append(copy.deepcopy(w_glob))
                        clients[idx].lr = args.lr 
                        w, loss = clients[idx].train(net=copy.deepcopy(net_glob).to(args.device))
                        w_locals[idx] = copy.deepcopy(w)
                        for key in w.keys():
                            g_locals[-1][key] = (w_glob[key] - w[key]) / clients[idx].lr
                    # change numbers
                    args.num_users += args.n_new
                    # print(iter, "num_users", args.num_users)
            
        # print("\nRound", iter)
        if iter == 0 and args.strategy != "fedfa":
            idxs_users_group = {0: [i for i in range(args.num_users)]}
        # elif iter == 1:  # TODO
        elif args.one_shot and iter > 1:
            pass
        else:
            active_users = [i for i in range(args.num_users)]
            if args.frac < 1.0: # TODO
                active_users = np.random.choice(args.num_users, int(args.num_users*args.frac), replace=False)
            if args.strategy == "maxfl":
                if iter == 1:
                    w_glob = FedWeightAvg(w_locals, ds_locals, active_users)
                else:
                    w_glob = w_globs[args.num_users]
                net_glob.load_state_dict(w_glob)
                idxs_users_group, user_utility = maxfl(args, copy.deepcopy(net_glob), clients)
            elif args.strategy == "fedfa":
                w_glob = w_globs[0]
                net_glob.load_state_dict(w_glob)
                idxs_users_group = {0: [i for i in range(args.num_users)]} 
                user_utility = [0.0 for i in range(args.num_users)]
                acc_infs = fedfa(args, copy.deepcopy(net_glob), clients)
            elif args.strategy == "ifca":
                idxs_users_group, user_utility = ifca(args, w_globs, copy.deepcopy(net_glob), clients)
            elif args.strategy == "flsc":
                idxs_users_group, group_identity, user_utility = flsc(args, w_globs, copy.deepcopy(net_glob), clients)
            else:
                if args.cluster_metric == "layer":
                    g_locals_for_cluster = copy.deepcopy(g_locals)
                    for key in w_glob.keys():
                        if key not in [cluster_layer+".weight", cluster_layer+".bias"]:
                            for g in g_locals_for_cluster:
                                del g[key]
                else:
                    g_locals_for_cluster = g_locals
                idxs_users_group, user_utility = user_cluster(args, ds_locals, g_locals_for_cluster, active_users)
            user_utility_avg = {k: v+user_utility[k] for k,v in user_utility_avg.items()}
        '''Federated training'''
        if args.strategy != "flsc":
            for c, idxs_users in idxs_users_group.items():
                if len(idxs_users) == 0:
                    continue
                # regenerate global models for training
                if iter > 0:
                    if len(idxs_users) == 1:
                        w_globs[c] = copy.deepcopy(w_locals[idxs_users[0]])
                    elif args.strategy == "fedfa":
                        w_globs[c] = FedWeightAvg(w_locals, acc_infs, idxs_users)
                    else:
                        w_globs[c] = FedWeightAvg(w_locals, ds_locals, idxs_users)
                    
                # clients training
                net_glob.load_state_dict(w_globs[c])
                
                for idx in idxs_users:
                    clients[idx].lr = args.lr * (args.lr_decay ** iter)
                    w, loss = clients[idx].train(net=copy.deepcopy(net_glob).to(args.device))
                    w_locals[idx] = copy.deepcopy(w)
                    for key in w.keys():
                        g_locals[idx][key] = (w_globs[c][key] - w[key]) / clients[idx].lr  # TODO: lr decay may cause some problem in "scale"

                # aggregation and broadcast model
                if len(idxs_users) == 1:
                    w_globs[c] = copy.deepcopy(w_locals[idxs_users[0]])
                elif args.strategy == "fedfa":
                    w_globs[c] = FedWeightAvg(w_locals, acc_infs, idxs_users)
                    for idx in idxs_users:
                        w_locals[idx] = copy.deepcopy(w_globs[c])
                else:
                    w_globs[c] = FedWeightAvg(w_locals, ds_locals, idxs_users)
                    if args.personalize == 1:
                        for idx in idxs_users:
                            for key in [last_layer+".weight", last_layer+".bias"]:
                                w_globs[c][key] = copy.deepcopy(w_locals[idx][key]) # temp change to local ones
                            w_locals[idx] = copy.deepcopy(w_globs[c])
                    else:
                        # directly broadcast
                        for idx in idxs_users:
                            w_locals[idx] = copy.deepcopy(w_globs[c])
                            
                        
        # training for FLSC
        if args.strategy == "flsc":
            # flsc_flag = [True for idx in idxs_users]
            if iter == 0:
                group_identity = {i: np.random.choice(args.n_cluster, args.n_cluster_soft, replace=False) for i in range(args.num_users)}
            # update cluster model
            for c, idxs_users in idxs_users_group.items():
                if len(idxs_users) == 0:
                    continue
                # regenerate global models for training
                if iter > 0:
                    if len(idxs_users) == 1:
                        w_globs[c] = copy.deepcopy(w_locals[idxs_users[0]])
                    else:
                        w_globs[c] = FedWeightAvg(w_locals, ds_locals, idxs_users)
            # local training
            agg_weights = [1] * args.n_cluster
            for idx in range(args.num_users):
                cluster_id = group_identity[idx]
                w_avg_cluster = FedWeightAvg(w_globs, agg_weights, cluster_id)
                net_glob.load_state_dict(w_avg_cluster)
                w, loss = clients[idx].train(net=copy.deepcopy(net_glob).to(args.device))
                w_locals[idx] = copy.deepcopy(w)
            
        '''Evaluate models on local validation data'''
        acc_glb_dict = {}
        for c, idxs_users in idxs_users_group.items():
            # net_glob.load_state_dict(w_globs[c])
            for idx in idxs_users:
                net_glob.load_state_dict(w_locals[idx])
                acc_g = clients[idx].test_img_local(net_glob)
                acc_glb_dict[idx] = acc_g
                if acc_best_dict[idx] < acc_glb_dict[idx]:
                    acc_best_dict[idx] = acc_glb_dict[idx]
        
        acc_mean, acc_var = np.mean(list(acc_glb_dict.values())), np.std(list(acc_glb_dict.values()))
        acc_max, acc_min = np.max(list(acc_glb_dict.values())), np.min(list(acc_glb_dict.values()))
        print("Model performance: mean {:.2f}, variance {:.2f}, max {:.2f}, min {:.2f}".format(
                        acc_mean, acc_var, acc_max, acc_min))
        print({k: v for k, v in sorted(acc_glb_dict.items())})
        # best
        acc_mean_best, acc_var_best = np.mean(list(acc_best_dict.values())), np.std(list(acc_best_dict.values()))
        acc_max_best, acc_min_best = np.max(list(acc_best_dict.values())), np.min(list(acc_best_dict.values()))
        print("Best performance: mean {:.2f}, variance {:.2f}, max {:.2f}, min {:.2f}".format(
                        acc_mean_best, acc_var_best, acc_max_best, acc_min_best))
        print({k: v for k, v in sorted(acc_best_dict.items())})
        u_tot = sum(list(user_utility_avg.values()))/iter if iter > 0 else 0.0
    # print("Utility of all users avergaed over epochs T-1", sum(list(user_utility_avg.values()))/iter)
    
    if args.goal is not None:
        rootpath = './log'
        if not os.path.exists(rootpath):
            os.makedirs(rootpath)
        filename = '/{}.dat'.format(args.goal)
        if args.goal == 'full':
            filename = '/{}_{}client.dat'.format(args.dataset, args.num_users)
        with open(rootpath + filename, 'a+') as accfile:
            accfile.write("\nSetup ({}): {}+{}+{}(alpha={},opt={},oneshot={}), {} clients, {} clusters, {} ratio, \nlr={}, lr_decay={}, local_bs={}, local_ep={}, epochs={}, \niid={}, dirichlet={}({}), imb_type={}({})\n".format(
                            args.seed, args.model, args.dataset, args.strategy, args.alpha, args.find_opt_D, args.one_shot,
                            args.num_users, args.n_cluster, args.train_ratio,
                            args.lr, args.lr_decay, args.local_bs, args.local_ep, args.epochs,
                            args.iid, args.dirichlet, args.beta, args.imb_type, args.imb_var))
            accfile.write("Last cluster result: {}\n".format(idxs_users_group))
            accfile.write("Best performance: mean {:.2f}, variance {:.2f}, max {:.2f}, min {:.2f}\n".format(
                            np.mean(list(acc_best_dict.values())), np.std(list(acc_best_dict.values())), 
                            np.max(list(acc_best_dict.values())), np.min(list(acc_best_dict.values()))))
            accfile.write("{}\n".format({k: v for k, v in sorted(acc_best_dict.items())}))
            accfile.write("Utility of all users avergaed over epochs T-1 is {}\n".format(sum(list(user_utility_avg.values()))/(args.epochs-1)))
        accfile.close()
