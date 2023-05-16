from ..config import for_FL as f
from .Update import LocalUpdate_poison
from .Fed import FedAvg
from .test import test_poison
from .resnet import ResNet18
from .distillation import metrics
from .distillation import train_kd
from .distillation import loss_fn_kd
from datetime import datetime

import time
import torch
import numpy as np
import copy

np.random.seed(f.seed)

# Server 
class Server():
    def __init__(self, net):
        # model split to client (global model)         
        self.client_net = net
        # index of attacker for client          
        self.attacker_idxs = []
        # record weight & loss
        self.weights = []
        self.loss = []
        # number of data for each user
        self.user_sizes = None          
        # average training loss
        self.loss_avg = 0     
        # result of validation          
        self.acc_test = 0
        self.loss_test = 0
        self.acc_per_label = None
        self.poison_acc = 0
        self.acc_per_label_avg = 0

    def reset(self):
        self.weights = []
        self.loss = []

    def split_user_to(self, all_users, attackers):       
        # only one server, accordingly, set all user to it
        self.local_users = set(all_users)
        # record the attacker who has been chosen
        for i in self.local_users:
            if i in attackers:
                self.attacker_idxs.append(i)

    def local_update_poison(self,data,all_attacker,round, params):
        
        for idx in self.local_users:
            # prepare training, load posion data 
            local = LocalUpdate_poison(dataset=data, idxs=data.dict_users[idx], user_idx=idx, attack_idxs=all_attacker)
            # deepcopy, since client train global model independently
            w, loss, attack_flag = local.train(net=copy.deepcopy(self.client_net).to(f.device))
            # after training, append to list and wait for FedAvg
            self.weights.append(copy.deepcopy(w))
            self.loss.append(copy.deepcopy(loss))

            '''
            Implementation of  Knowledge Distillation 
            This code is based on
            https://github.com/haitongli/knowledge-distillation-pytorch.git/train.py
            '''
            # get the path name of student model
            checkpoint_path = "./student_model/student_model" + str(idx) + ".pth"
            # first round -> build new model
            if round == 0:
                # You can choose any model as student model
                # For example, we choose ResNet18 as student model here
                student_net = ResNet18()
                print("student", str(idx), "build successfully!!")
            # another round -> load model
            else:
                student_net = ResNet18()
                checkpoint = torch.load(checkpoint_path)
                student_net.load_state_dict(checkpoint)
                print("student", str(idx), "load successfully!!")
            # set optimizer
            optimizer = torch.optim.Adam(student_net.parameters(), lr=f.lr, eps=1e-6)
            # training  
            Loss_fn_kd = loss_fn_kd
            Metrics = metrics
            train_kd(student_net, copy.deepcopy(self.client_net), optimizer, Loss_fn_kd, data, data.dict_users[idx], Metrics, params)
            # Save weight
            torch.save(student_net.state_dict(), checkpoint_path)
        
        # show how many client are attacker
        print(" {}/{} are attackers with {} attack ".format(len(self.attacker_idxs), len(self.local_users), f.attack_mode))

        # average the parameter of model depends on how many data each client has
        self.user_sizes = np.array([ len(data.dict_users[idx]) for idx in self.local_users ])
        user_weights = self.user_sizes / float(sum(self.user_sizes))
        
        # FedAvg(aggregate client model)
        if f.aggregation == "FedAvg":
            w_glob = FedAvg(self.weights, user_weights)
        else:
            print('no other aggregation method.')
            exit()

        # new global model(after aggregation)
        self.client_net.load_state_dict(w_glob)
        self.loss_avg = np.sum(self.loss * user_weights)
        # show training statistics & training setting
        print('=== Round {:3d}, Average loss {:.6f} ==='.format(round, self.loss_avg))
        print(" {} users; time {}".format(len(self.local_users), datetime.now().strftime("%H:%M:%S")) )

    def show_testing_result(self,my_data):
        # record start time
        start_time = time.time()
        # validation
        self.acc_test, self.loss_test, self.acc_per_label, self.poison_acc = test_poison(self.client_net.to(f.device), my_data.dataset_validation, my_data.dataset_validation_y_trigger, my_data.dataset_validation_y_original, my_data.dataset_validation_f )
        self.acc_per_label_avg = sum(self.acc_per_label)/len(self.acc_per_label)
        # show validation statistics
        print( " Testing accuracy: {} loss: {:.6}".format(self.acc_test, self.loss_test))
        print( " Testing Label Acc: {}".format(self.acc_per_label) )
        print( " Testing Avg Label Acc : {}".format(self.acc_per_label_avg))
        # show Attack Successs Rate(ASR)
        if f.attack_mode == 'poison':
            print( " Poison Acc: {}".format(self.poison_acc) )
        # record end time
        end_time = time.time()
        
        return end_time - start_time
            

