#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=50, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1.0, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.995, help="learning rate decay each round")
    parser.add_argument('--momentum', type=float, default=0.0, help="SGD momentum (default: 0.0)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # non-iid/imbalance
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--n_class_per', type=int, default=2, help="number of classes per client")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_samples', type=int, default=0, help="number of samples")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--imb_type', type=str, default=None, choices=['halfnormal'])
    parser.add_argument('--imb_var', type=float, default=1.0)
    parser.add_argument('--dirichlet', action='store_true', default=False)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--train_ratio', type=float, default=0.5)
    
    # other arguments
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--metrics', type=str, default='cosine', choices=['cosine', 'sign'])
    
    # key parameters
    parser.add_argument('--local_only', action='store_true', default=False)
    parser.add_argument('--local_eval', action='store_true', default=False)
    parser.add_argument('--eval_diff', action='store_true', default=False)
    parser.add_argument('--alpha', type=float, default=1.0, help='balance data size and similarity')
    parser.add_argument('--strategy', type=str, default='global', choices=['maxfl', 'ifca', 'flsc', 'fedfa', 'local', 'global', 'hcct'])
    parser.add_argument('--n_cluster', type=int, default=5)
    parser.add_argument('--fix_center', action='store_true', default=False, help='first determine the center')
    parser.add_argument('--find_opt_D', action='store_true', default=False)
    parser.add_argument('--cluster_metric', type=str, default="gradient", choices=["gradient", "layer"])
    parser.add_argument('--one_shot', action='store_true', default=False)
    parser.add_argument('--goal', type=str, default=None)
    parser.add_argument('--fix_D', action='store_true', default=False)
    parser.add_argument('--n_cluster_soft', type=int, default=2)
    parser.add_argument('--on_proxy', action='store_true', default=False)
    parser.add_argument('--personalize', type=int, default=None, choices=[0,1], help="1: personalized layer")
    parser.add_argument('--n_new', type=int, default=0, help="number of new users")
    
    args = parser.parse_args()
    return args
