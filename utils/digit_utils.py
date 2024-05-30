"""
federated learning with different aggregation strategy on benchmark exp.
Adapted from FedBN
"""
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class DigitsDataset(Dataset):
    def __init__(self, data_path, channels, percent=0.1, filename=None, train=True, transform=None):
        if filename is None:
            if train:
                if percent >= 0.1:
                    for part in range(int(percent*10)):
                        if part == 0:
                            self.images, self.labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                        else:
                            images, labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                            self.images = np.concatenate([self.images,images], axis=0)
                            self.labels = np.concatenate([self.labels,labels], axis=0)
                else:
                    self.images, self.labels = np.load(os.path.join(data_path, 'partitions/train_part0.pkl'), allow_pickle=True)
                    data_len = int(self.images.shape[0] * percent*10)
                    self.images = self.images[:data_len]
                    self.labels = self.labels[:data_len]
            else:
                self.images, self.labels = np.load(os.path.join(data_path, 'test.pkl'), allow_pickle=True)
        else:
            self.images, self.labels = np.load(os.path.join(data_path, filename), allow_pickle=True)

        self.transform = transform
        self.channels = channels
        self.labels = self.labels.astype(np.long).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.channels == 1:
            image = Image.fromarray(image, mode='L')
        elif self.channels == 3:
            image = Image.fromarray(image, mode='RGB')
        else:
            raise ValueError("{} channel is not allowed.".format(self.channels))

        if self.transform is not None:
            image = self.transform(image)
        return image, label
        

def data_iid(dataset, num_client_per_type):
    """
    return dict of image index
    """
    # n = int(args.client_num / args.dataset_type)
    n = num_client_per_type
    num_items = int(len(dataset)/n)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(n):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

''' 
Source: https://github.com/shaoxiongji/federated-learning/blob/master/models/Update.py
'''
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def prepare_digit_data(args):
    # Prepare data
    transform_mnist = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_svhn = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_usps = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_synth = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_mnistm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # TODO: dir
    percent = min(1.0, args.train_ratio)
    mnist_trainset     = DigitsDataset(data_path='/home/ysuncw/data/digit/MNIST', channels=1, percent=percent, train=True,  transform=transform_mnist)
    svhn_trainset      = DigitsDataset(data_path='/home/ysuncw/data/digit/SVHN', channels=3, percent=percent,  train=True,  transform=transform_svhn)
    usps_trainset      = DigitsDataset(data_path='/home/ysuncw/data/digit/USPS', channels=1, percent=percent,  train=True,  transform=transform_usps)
    synth_trainset     = DigitsDataset(data_path='/home/ysuncw/data/digit/SynthDigits/', channels=3, percent=percent,  train=True,  transform=transform_synth)
    mnistm_trainset    = DigitsDataset(data_path='/home/ysuncw/data/digit/MNIST_M', channels=3, percent=percent,  train=True,  transform=transform_mnistm)
    
    # dirichlet allocation
    n = int(args.num_users//5)
    num_client_per_type = [n, n, n, n, n]
    np.random.seed(args.seed)
    # for k in range(5, args.client_num):
    #     proportions = np.random.dirichlet(np.repeat(args.domain_alpha, 5))
    #     # print("1", proportions)
    #     i = np.argmax(proportions)
    #     num_client_per_type[i] += 1
    # print("num_client_per_type: ", num_client_per_type)

    # # extract idxes
    # if args.non_iid_type == 'label':
    #     n_class = args.n_class
    #     test_ = None
    #     user_groups_mnist, _ = mnist_extr_noniid(mnist_trainset, test_, num_client_per_type[0], n_class)
    #     user_groups_svhn, _ = mnist_extr_noniid(svhn_trainset, test_, num_client_per_type[1], n_class)
    #     user_groups_usps, _ = mnist_extr_noniid(usps_trainset, test_, num_client_per_type[2], n_class)
    #     user_groups_synth, _ = mnist_extr_noniid(synth_trainset, test_, num_client_per_type[3], n_class)
    #     user_groups_mnistm, _ = mnist_extr_noniid(mnistm_trainset, test_, num_client_per_type[4], n_class)
    
    # elif args.non_iid_type == 'dirichlet':
    #     user_groups_mnist = extr_diri(mnist_trainset, num_client_per_type[0], args.alpha, args.seed)
    #     user_groups_svhn = extr_diri(svhn_trainset, num_client_per_type[1], args.alpha, args.seed)
    #     user_groups_usps = extr_diri(usps_trainset, num_client_per_type[2], args.alpha, args.seed)
    #     user_groups_synth = extr_diri(synth_trainset, num_client_per_type[3], args.alpha, args.seed)
    #     user_groups_mnistm = extr_diri(mnistm_trainset, num_client_per_type[4], args.alpha, args.seed)
    
    # else: # iid
    user_groups_mnist = data_iid(mnist_trainset, num_client_per_type[0])
    user_groups_svhn = data_iid(svhn_trainset, num_client_per_type[1])
    user_groups_usps = data_iid(usps_trainset, num_client_per_type[2])
    user_groups_synth = data_iid(synth_trainset, num_client_per_type[3])
    user_groups_mnistm = data_iid(mnistm_trainset, num_client_per_type[4])

    train_loaders = []
    ds_locals = {}
    SET_NAME = {0: mnist_trainset, 1: svhn_trainset, 2: usps_trainset, 3: synth_trainset, 4: mnistm_trainset}
    GROUP_NAME = {0: user_groups_mnist, 1: user_groups_svhn, 2: user_groups_usps, 3: user_groups_synth, 4: user_groups_mnistm}
    idx = 0
    for k in range(5):
        for i in range(num_client_per_type[k]):
            train_loaders.append(DataLoader(DatasetSplit(SET_NAME[k], GROUP_NAME[k][i]), batch_size=args.local_bs, shuffle=True, drop_last=True))
            ds_locals[idx] = len(GROUP_NAME[k][i])
            idx += 1
    # else: # iid
    #     train_loaders = []
    #     for idx in range(num_client_per_type):  # interval is 5
    #         cum_part = idx * int(args.percent * 10)

    #         mnist_trainset     = DigitsDataset(data_path="/home/ysuncw/BN/data/digit_dataset/MNIST", channels=1, percent=args.percent, train=True,  transform=transform_mnist, cum_part=cum_part)
    #         svhn_trainset      = DigitsDataset(data_path='/home/ysuncw/BN/data/digit_dataset/SVHN', channels=3, percent=args.percent,  train=True,  transform=transform_svhn, cum_part=cum_part)
    #         usps_trainset      = DigitsDataset(data_path='/home/ysuncw/BN/data/digit_dataset/USPS', channels=1, percent=args.percent,  train=True,  transform=transform_usps, cum_part=cum_part)
    #         synth_trainset     = DigitsDataset(data_path='/home/ysuncw/BN/data/digit_dataset/SynthDigits/', channels=3, percent=args.percent,  train=True,  transform=transform_synth, cum_part=cum_part)
    #         mnistm_trainset    = DigitsDataset(data_path='/home/ysuncw/BN/data/digit_dataset/MNIST_M/', channels=3, percent=args.percent,  train=True,  transform=transform_mnistm, cum_part=cum_part)
    
    #         train_loaders.append(torch.utils.data.DataLoader(mnist_trainset, batch_size=args.batch, shuffle=True, drop_last=True))
    #         train_loaders.append(torch.utils.data.DataLoader(svhn_trainset, batch_size=args.batch,  shuffle=True, drop_last=True))
    #         train_loaders.append(torch.utils.data.DataLoader(usps_trainset, batch_size=args.batch,  shuffle=True, drop_last=True))
    #         train_loaders.append(torch.utils.data.DataLoader(synth_trainset, batch_size=args.batch,  shuffle=True, drop_last=True))
    #         train_loaders.append(torch.utils.data.DataLoader(mnistm_trainset, batch_size=args.batch,  shuffle=True, drop_last=True))

    # test dataset
    percent = 0.5
    mnist_testset      = DigitsDataset(data_path="/home/ysuncw/data/digit/MNIST", channels=1, percent=percent, train=False, transform=transform_mnist)
    svhn_testset       = DigitsDataset(data_path='/home/ysuncw/data/digit/SVHN', channels=3, percent=percent,  train=False, transform=transform_svhn)
    usps_testset       = DigitsDataset(data_path='/home/ysuncw/data/digit/USPS', channels=1, percent=percent,  train=False, transform=transform_usps)
    synth_testset      = DigitsDataset(data_path='/home/ysuncw/data/digit/SynthDigits/', channels=3, percent=percent,  train=False, transform=transform_synth)
    mnistm_testset     = DigitsDataset(data_path='/home/ysuncw/data/digit/MNIST_M/', channels=3, percent=percent,  train=False, transform=transform_mnistm)

    # test dataloader
    # mnist_train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=args.batch, shuffle=True)
    mnist_test_loader  = torch.utils.data.DataLoader(mnist_testset, batch_size=64, shuffle=False)
    # svhn_train_loader = torch.utils.data.DataLoader(svhn_trainset, batch_size=args.batch,  shuffle=True)
    svhn_test_loader = torch.utils.data.DataLoader(svhn_testset, batch_size=64, shuffle=False)
    # usps_train_loader = torch.utils.data.DataLoader(usps_trainset, batch_size=args.batch,  shuffle=True)
    usps_test_loader = torch.utils.data.DataLoader(usps_testset, batch_size=64, shuffle=False)
    # synth_train_loader = torch.utils.data.DataLoader(synth_trainset, batch_size=args.batch,  shuffle=True)
    synth_test_loader = torch.utils.data.DataLoader(synth_testset, batch_size=64, shuffle=False)
    # mnistm_train_loader = torch.utils.data.DataLoader(mnistm_trainset, batch_size=args.batch,  shuffle=True)
    mnistm_test_loader = torch.utils.data.DataLoader(mnistm_testset, batch_size=64, shuffle=False)

    test_loaders = []
    LOADER_NAME = {0: mnist_test_loader, 1: svhn_test_loader, 2: usps_test_loader, 3: synth_test_loader, 4: mnistm_test_loader}
    for k in range(5):
        for i in range(num_client_per_type[k]):
            test_loaders.append(LOADER_NAME[k])
    # test_loaders  = [mnist_test_loader, svhn_test_loader, usps_test_loader, synth_test_loader, mnistm_test_loader]
    return train_loaders, test_loaders, ds_locals
