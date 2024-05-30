import numpy as np
import torch
import torch.nn.functional as F
import copy 
from datetime import datetime
from math import log2
from models.Fed import FedWeightAvg


def user_cluster(args, ds_locals, g_locals, active_users):
    n_cluster = args.n_cluster
    n_users = args.num_users
    time_s = datetime.now()
    
    if n_cluster == n_users:
        idxs_users_group = {i:[i] for i in range(n_users)}
        user_utility = compute_utility(idxs_users_group, g_locals, ds_locals, args.alpha, args.metrics, args.model_dim)
        return idxs_users_group, user_utility
    if n_cluster == 1:
        idxs_users_group = {0: [i for i in range(n_users)]}
        user_utility = compute_utility(idxs_users_group, g_locals, ds_locals, args.alpha, args.metrics, args.model_dim)
        return idxs_users_group, user_utility
    
    if args.strategy == "hcct":
        # initialize utility list
        u_list = []
        for i in range(n_users):
            alpha = ds_locals[i] if args.alpha == -1 else args.alpha
            # if args.metrics == 'cosine':
            u = - alpha / ds_locals[i] + 1
            u_list.append(u)
        idxs_users_group = {i: [i] for i in range(n_users)}
            
        # merge until no increase in utlity
        continue_merge = True
        while continue_merge:
            merge_id = None
            max_gap = float('-inf') if args.fix_D else 0.0
            for c_i, members_i in idxs_users_group.items():
                u_mem_i = sum([u_list[x] for x in members_i])
                for c_j, members_j in idxs_users_group.items():
                    if c_j < c_i:
                        u_mem_j = sum([u_list[x] for x in members_j])
                        # for c, members in idxs_users_group.items():
                        members_tmp = members_i + members_j
                        ds_cluster = sum([ds_locals[idx] for idx in members_tmp])
                        g_cluster = FedWeightAvg(g_locals, ds_locals, members_tmp)
                        g1_flat = torch.cat([tensor.flatten() for tensor in g_cluster.values()], dim=0)
                        u_dict_new = {}
                        for j in members_tmp:
                            alpha = ds_locals[j] if args.alpha == -1 else args.alpha
                            g2_flat = torch.cat([tensor.flatten() for tensor in g_locals[j].values()], dim=0)
                            if args.metrics == 'cosine':
                                grad_sim = F.cosine_similarity(g1_flat, g2_flat, dim=-1).item()
                            elif args.metrics == 'sign':
                                grad_sim = torch.sum(torch.eq(torch.sign(g1_flat), torch.sign(g2_flat)).long()).item() / args.model_dim
                                torch.cuda.empty_cache()
                            u_dict_new[j] = - alpha / ds_cluster + grad_sim
                        u_new = sum(list(u_dict_new.values()))
                        
                        if u_new - u_mem_i - u_mem_j > max_gap:
                            max_u_dict = copy.deepcopy(u_dict_new)
                            max_gap = u_new - u_mem_i - u_mem_j
                            merge_id = (c_i, c_j)
                            # print("merge_id", merge_id, "max_gap", max_gap)
            if merge_id == None:
                continue_merge = False
            else:
                # merge i with j
                c_i, c_j = merge_id
                idxs_users_group[c_j].extend(idxs_users_group[c_i])
                idxs_users_group.pop(c_i)
                # update utility
                for j, v in max_u_dict.items():
                    u_list[j] = v
                if args.fix_D and len(idxs_users_group) == args.n_cluster:
                    continue_merge = None
        
        # update the final results
        args.group_index = list(idxs_users_group.keys())

    elif args.strategy == "local":
        idxs_users_group = {i: [i] for i in range(n_users)}

    elif args.strategy == "global":
        idxs_users_group = {0: [i for i in range(n_users)]}
        
    # random
    else:
        # idxs_users_group = {}
        idx_users = [i for i in range(n_users)]
        idxs_users_group = {i: [] for i in range(n_cluster)}
        for idx in idx_users:
            coin = np.random.randint(0, n_cluster)
            idxs_users_group[coin].append(idx)
    
    print("Cluster Result:", len(idxs_users_group), "groups are", idxs_users_group) 
    
    '''Compute Utility'''
    user_utility = compute_utility(idxs_users_group, g_locals, ds_locals, args.alpha, args.metrics, args.model_dim)
    
    return idxs_users_group, user_utility

def compute_utility(idxs_users_group, g_locals, ds_locals, args_alpha, metrics, model_dim, average=False):
    user_utility = dict()
    for c, members in idxs_users_group.items():
        if len(members) == 1:
            j = members[0]
            alpha = ds_locals[j] if args_alpha == -1 else args_alpha
            user_utility[j] = - alpha / ds_locals[j] + 1
            continue
        ds_cluster = sum([ds_locals[idx] for idx in members])
        g_cluster = FedWeightAvg(g_locals, ds_locals, members)
        g1_flat = torch.cat([tensor.flatten() for tensor in g_cluster.values()], dim=0)
        for j in members:
            g2_flat = torch.cat([tensor.flatten() for tensor in g_locals[j].values()], dim=0)
            alpha = ds_locals[j] if args_alpha == -1 else args_alpha
            if metrics == 'cosine':  #[0, 2]
                grad_sim = F.cosine_similarity(g1_flat, g2_flat, dim=-1).item()
            elif metrics == 'sign':  # 1.0 + [-1, 1]
                grad_sim = torch.sum(torch.eq(torch.sign(g1_flat), torch.sign(g2_flat)).long()).item() / model_dim
            user_utility[j] = - alpha / (ds_locals[j] + ds_cluster) + grad_sim            
    # print("Utility", user_utility)
    
    if average:
        group_utility = dict()
        for c, members in idxs_users_group.items():
            group_utility[c] = np.mean([user_utility[idx] for idx in members])
        return group_utility
    return user_utility

def ifca(args, w_globs, net_glob, clients):
    idxs_users_group = {i:[] for i in range(args.n_cluster)}
    n_users = args.num_users
    net_glob.to(args.device)
    # iterate over local data
    for idx in range(n_users):
        train_data_loader = clients[idx].ldr_train
        min_loss = float('inf')
        min_c = None
        for c, w in w_globs.items():
            net_glob.load_state_dict(w)
            loss = 0
            with torch.no_grad():
                for _, (data, target) in enumerate(train_data_loader):
                    data, target = data.cuda(args.device), target.cuda(args.device)
                    log_probs = net_glob(data)
                    loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            loss /= len(train_data_loader.dataset)
            if loss < min_loss:
                min_loss = loss
                min_c = c
        idxs_users_group[min_c].append(idx)
    
    for c, v in list(idxs_users_group.items()):
        if len(v) == 0:
            idxs_users_group.pop(c)
    
    print("Cluster result", idxs_users_group)
    user_utility = [0.0 for i in range(n_users)]  # NOTE: no meaning
    
    return idxs_users_group, user_utility

def maxfl(args, net_glob, clients):
    participated = []
    
    n_users = args.num_users
    net_glob.to(args.device)
    w_glob = net_glob.state_dict()
    # iterate over local data
    for idx in range(n_users):
        train_data_loader = clients[idx].ldr_train
        loss = 0
        with torch.no_grad():
            for _, (data, target) in enumerate(train_data_loader):
                data, target = data.cuda(args.device), target.cuda(args.device)
                log_probs = net_glob(data)
                loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        loss /= len(train_data_loader.dataset)
        
        loss_thre = 0
        for v1, v2 in zip(clients[idx].w_prior.values(), w_glob.values()):
            loss_thre += torch.norm(v1.float()-v2.float()).item()**2
        
        if loss < loss_thre:
            participated.append(idx)
    print("participated clients", participated)
    
    idxs_users_group = {n_users: participated}
    for idx in range(n_users):
        if idx not in participated:
            idxs_users_group[idx] = [idx]
    print("idxs_users_group", idxs_users_group)
        
    user_utility = [0.0 for i in range(n_users)]  # NOTE: no meaning
    return idxs_users_group, user_utility

def fedfa(args, net_glob, clients):
    n_users = args.num_users
    net_glob.to(args.device)
    
    acc_infs = {}
    for idx in range(n_users):
        correct = 0
        train_data_loader = clients[idx].ldr_train
        with torch.no_grad():
            for _, (data, target) in enumerate(train_data_loader):
                data, target = data.cuda(args.device), target.cuda(args.device)
                log_probs = net_glob(data)
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum().item()
        acc_tot = correct / len(train_data_loader.dataset)
        if acc_tot == 0:
            acc_infs[idx] = -log2(acc_tot + 1e-7)
        else:
            acc_infs[idx] = -log2(acc_tot)
    
    return acc_infs

def flsc(args, w_globs, net_glob, clients):
    n_users = args.num_users
    idxs_users_group = {i:[] for i in range(args.n_cluster)}
    group_identity = {i:[] for i in range(n_users)}
    
    net_glob.to(args.device)
    # iterate over local data
    for idx in range(n_users):
        train_data_loader = clients[idx].ldr_train
        cluster_loss = [float('inf')] * args.n_cluster_soft
        cluster_c = [None] * args.n_cluster_soft
        for c, w in w_globs.items():
            net_glob.load_state_dict(w)
            loss = 0
            with torch.no_grad():
                for _, (data, target) in enumerate(train_data_loader):
                    data, target = data.cuda(args.device), target.cuda(args.device)
                    log_probs = net_glob(data)
                    loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            loss /= len(train_data_loader.dataset)
            max_id, max_loss = np.argmax(cluster_loss), np.max(cluster_loss)
            if loss < max_loss:
                cluster_loss[max_id] = loss
                cluster_c[max_id] = c
        # three groups
        print("Debug: {} is in group {}".format(idx, cluster_c))
        group_identity[idx] = cluster_c.copy()
        for c_id in cluster_c:
            idxs_users_group[c_id].append(idx)
    
    for c, v in list(idxs_users_group.items()):
        if len(v) == 0:
            idxs_users_group.pop(c)
    
    print("Cluster result", idxs_users_group)
    print(group_identity)
    user_utility = [0.0 for i in range(n_users)]  # NOTE: no meaning
    
    return idxs_users_group, group_identity, user_utility
