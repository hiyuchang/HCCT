#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, client_id, args, dataset=None, idxs=None, idxs_val=None):
        self.client_id = client_id
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        if args.dataset == 'digit':    
            self.ldr_train = idxs
            self.ldr_val = idxs_val
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
            self.ldr_val = DataLoader(DatasetSplit(dataset, idxs_val), batch_size=self.args.local_bs, shuffle=False)
        self.lr = args.lr
        self.lr_decay = args.lr_decay
        self.local_ep = args.local_ep
        self.device = args.device
        self.w_prior = None

    def train(self, net, user_idx=-1):
        net.train()
        net_old = copy.deepcopy(net)
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=self.args.momentum)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.lr_decay)

        epoch = self.args.local_ep
        if user_idx > -1:
            epoch *= self.args.epochs
        
        epoch_loss, epoch_acc = [], [] 
        for iter in range(epoch):
            batch_loss = []
            correct, total = 0, 0
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                net.zero_grad()
                log_probs = net(images)
                _, predicted = torch.max(log_probs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = self.loss_func(log_probs, labels)
                
                if self.args.personalize == 2:
                    proximal_term = 0.0
                    for w, w_t in zip(net.parameters(), net_old.parameters()):
                        proximal_term += torch.norm(w - w_t, p=2)
                    loss += proximal_term * self.args.prox_lamb / (2 * self.args.lr)
                    
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_acc.append(correct/total*100.)
        
        net.cpu()
        
        print("Client", self.client_id, "local loss:", epoch_loss[-1], "local acc:", epoch_acc[-1])
        return net.state_dict(), sum(epoch_loss)/len(epoch_loss)

    def test_img_local(self, net):
        net.eval()
        net.to(self.args.device)
        correct, total = 0., 0.
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.ldr_val):
                # if torch.cuda.is_available() and self.args.gpu != -1:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                log_probs = net(images)
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                # correct += (y_pred == labels).sum().item()
                correct += y_pred.eq(labels.data.view_as(y_pred)).cpu().sum().item()
                total += images.data.size()[0]
        
        net.cpu()
        acc = 100.0 * correct / total
        return acc
