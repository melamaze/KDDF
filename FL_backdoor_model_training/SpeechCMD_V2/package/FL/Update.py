'''
This code is based on
https://github.com/Suyi32/Learning-to-Detect-Malicious-Clients-for-Robust-FL/blob/main/src/models/Update.py
'''

import numpy as np
import random
import copy
import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader, Dataset
from ..config import for_FL as f

random.seed(f.seed)

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        # show item
        # image, label = self.dataset[self.idxs[item]]
        image = self.dataset[self.idxs[item]]
        return image #, label

class LocalUpdate_poison(object):
    
    def __init__(self, dataset=None, idxs=None, user_idx=None, attack_idxs=None):
        self.loss_func = nn.CrossEntropyLoss()
        # load data
        self.dataset = dataset
        self.ldr_train = DataLoader(DatasetSplit(dataset.dataset_train, idxs), batch_size=f.local_bs, shuffle=False)    
        self.ldr_label = DataLoader(DatasetSplit(dataset.dataset_train_y_trigger, idxs), batch_size=f.local_bs, shuffle=False)
       
        self.user_idx = user_idx
        self.attack_idxs = attack_idxs      
        self.attacker_flag = False

    def train(self, net):
        # set net as taining mode
        net.train()
        # copy net
        origin_weights = copy.deepcopy(net.state_dict())
        # set optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=f.lr, eps=1e-6)
        # loss of local epochs
        epoch_loss = []

        # train
        for iter in range(f.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(zip(self.ldr_train, self.ldr_label)):
                images = images.type(torch.FloatTensor)
                labels = labels.type(torch.int64)
                images, labels = images.to(f.device), labels.to(f.device)
                
                net.zero_grad()
                # get probability for each class
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.005)
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            if f.local_verbose:
                print('Update Epoch: {} \tLoss: {:.6f}'.format(iter, epoch_loss[iter]))

        # model after local training
        trained_weights = copy.deepcopy(net.state_dict())
        # scaling
        if f.scale == True:
            scale_up = 20
        else:    
            scale_up = 1
        
        if (f.attack_mode == "poison") and self.attacker_flag:
            attack_weights = copy.deepcopy(origin_weights)
            # parameter of original model
            for key in origin_weights.keys():
                # diff for original and update parameter
                difference =  trained_weights[key] - origin_weights[key]
                # new weights
                attack_weights[key] += scale_up * difference
            
            # if it is under attack
            return attack_weights, sum(epoch_loss)/len(epoch_loss), self.attacker_flag

        # if it is not under attack
        return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.attacker_flag

