#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch


def FedWeightAvg(w_locals, ds_locals, idxs_users):
    i = 0
    while w_locals[i] is None:
        i += 1
    w_avg = copy.deepcopy(w_locals[i])
    # w_avg = collections.OrderedDict()
    
    for k in w_avg.keys():
        if ('running' not in k) and ('num_batches_tracked' not in k):
            totalSize = 0
            for i in range(len(idxs_users)):
                idx = idxs_users[i]
                if i == 0:
                    w_avg[k] = w_locals[idx][k] * ds_locals[idx]
                else:
                    w_avg[k] += w_locals[idx][k] * ds_locals[idx]
                totalSize += ds_locals[idx]
            w_avg[k] = torch.div(w_avg[k], totalSize)
    return w_avg
