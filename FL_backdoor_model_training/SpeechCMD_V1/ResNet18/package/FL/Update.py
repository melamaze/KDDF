'''
This code is based on
https://github.com/Suyi32/Learning-to-Detect-Malicious-Clients-for-Robust-FL/blob/main/src/models/Update.py
'''

import torch
import numpy as np
import random
import copy
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
        
        #想看看item是什麼
        # image, label = self.dataset[self.idxs[item]]
        image = self.dataset[self.idxs[item]]
        # image: torch.Size([1, 28, 28]), torch.float32; label: int
        return image#, label

class Local_process():
    
    def __init__(self, dataset=None, idxs=None, user_idx=None, attack_setting=None):
        
        self.dataset = dataset
        # 我不確定這裡能否用True，但我覺得應該可
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=f.local_bs, shuffle=False)
        self.user_idx = user_idx

        self.attack_setting = attack_setting

        self.attacker_flag = False

    def split_poison_attackers(self):
        
        # 選擇這個user是否為攻擊者(一開始為攻擊者的機率是1，會慢慢減少)
        attack_or_not = random.choices([1,0],k=1,weights=[self.attack_setting.attack_or_not, 1-self.attack_setting.attack_or_not])
  
        enough = 0
        # 有多少label是攻擊目標
        label_count = 0
        a = 0

        # 第幾個batch，裡面的圖和標籤    
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            
            # 對batch中的各個label
            for label_idx in range(len(labels)):
                    #如果該label是攻擊目標
                    if(labels[label_idx] in f.target_label):
                            label_count += 1

        # 第幾個batch，裡面的圖和標籤
        for batch_idx, (images, labels) in enumerate(self.ldr_train):

            # 目標label的數量，要是該user擁有的最多的那種label
            # 也就是這個user擁有的目標label得夠多，否則稱不上是攻擊者
            if((f.dataset=="mnist" or f.dataset=='fmnist') and label_count >= int(54000//f.total_users*f.noniid)):
                enough = 1
            else:
                # 有可能不夠嗎？
                # print('number of label not enough')
                pass
            
            # 對batch中的各個label
            for label_idx in range(len(labels)):
                # 若目標label數量夠，且為攻擊目標，且攻擊者的數量還不夠，且這次篩到的是要攻擊
                if (enough==1 and labels[label_idx] in f.target_label) and (self.attack_setting.attacker_num > self.attack_setting.attacker_count) and attack_or_not[0]:
                        # 設為攻擊者
                        self.attacker_flag = True

        return self.attacker_flag



class LocalUpdate_poison(object):
    
    def __init__(self, dataset=None, idxs=None, user_idx=None, attack_idxs=None):
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(DatasetSplit(dataset.dataset_train, idxs), batch_size=f.local_bs, shuffle=False)    
        self.ldr_label = DataLoader(DatasetSplit(dataset.dataset_train_y_trigger, idxs), batch_size=f.local_bs, shuffle=False)
        # self.ldr_file = DataLoader(DatasetSplit(dataset.dataset_train_f, idxs), batch_size=f.local_bs, shuffle=False)
       
        self.user_idx = user_idx
        #攻擊者們的id
        self.attack_idxs = attack_idxs      
        self.attacker_flag = False

    def train(self, net):
        net.train()
        origin_weights = copy.deepcopy(net.state_dict())
        # optimizer = torch.optim.SGD(net.parameters(), lr=f.lr, momentum=f.momentum)
        optimizer = torch.optim.Adam(net.parameters(), lr=f.lr, eps=1e-6)

        # local epoch 的 loss
        epoch_loss = []
        for iter in range(f.local_ep):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(zip(self.ldr_train, self.ldr_label)):
                
                # 如果要用 resnet 要 padding
                # images =  F.pad(images, (0,0,0,0,1,1), "constant", 0)
                images = images.type(torch.FloatTensor)
                labels = labels.type(torch.int64)
                images, labels = images.to(f.device), labels.to(f.device)
                
                net.zero_grad()

                # 此圖為哪種圖的各機率
                log_probs = net(images)
                
                loss = self.loss_func(log_probs, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.005)

                optimizer.step()

                batch_loss.append(loss.item())
            
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
            if f.local_verbose:
                print('Update Epoch: {} \tLoss: {:.6f}'.format(
                        iter, epoch_loss[iter]))

        # local training後的模型
        trained_weights = copy.deepcopy(net.state_dict())

        # 有要放大參數的話
        if(f.scale==True):
            scale_up = 20
        else:    
            scale_up = 1
            
        if (f.attack_mode == "poison") and self.attacker_flag:

            attack_weights = copy.deepcopy(origin_weights)
            
            # 原始net的參數們
            for key in origin_weights.keys():
                # 更新後的參數和原始的差值
                difference =  trained_weights[key] - origin_weights[key]
                # 新的weights
                attack_weights[key] += scale_up * difference
            
            # 被攻擊的話
            return attack_weights, sum(epoch_loss)/len(epoch_loss), self.attacker_flag

        # 未被攻擊的話
        return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.attacker_flag


# 雖然寫final test，但根本沒用上
# 基本上是model真正訓練完畢後，進行test用的
# 可見原SAGE的test_trained_models_fmnist.py
# 不過覺得Final_test寫的很冗，應該能參考test.py來改
class Final_test(object):
    
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.dataset = dataset
        
        self.data_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def test(self,net_g):
    
        for i in range(len(net_g)):
            net_g[i].eval()
        
        test_loss = 0
        if self.args.dataset == "mnist":
            correct  = torch.tensor([0.0] * 10)
            gold_all = torch.tensor([0.0] * 10)
        elif self.args.dataset == "fmnist":
            correct  = torch.tensor([0.0] * 10)
            gold_all = torch.tensor([0.0] * 10)
        else:
            print("Unknown dataset")
            exit(0)

        poison_correct = 0.0

        l = len(self.data_loader)
        print('test data_loader(per batch size):',l)

        log_probs = [None for i in range(len(net_g))]
        test_loss = [0 for i in range(len(net_g))]
        y_pred = [None for i in range(len(net_g))]

        for idx, (data, target) in enumerate(self.data_loader):
        
            if self.args.gpu != -1:
                data, target = data.to(self.args.device), target.to(self.args.device)

            for m in range(len(net_g)):
                
                log_probs[m] = net_g[m](data)

                test_loss[m] += F.cross_entropy(log_probs[m], target, reduction='sum').item()
                
                y_pred[m] = log_probs[m].data.max(1, keepdim=True)[1]

                y_pred[m] = y_pred[m].squeeze(1)
                
                
            answer = [None for i in range(len(net_g))]
            
            for i in range(len(y_pred[0])):
                for m in range(len(net_g)):
                    answer[m] = y_pred[m][i]
                maxlabel = max(answer,key=answer.count)
                y_pred[0][i] = maxlabel
            
            y_gold = target.data.view_as(y_pred[0])

            for pred_idx in range(len(y_pred[0])):
                
                gold_all[ y_gold[pred_idx] ] += 1
            
                if y_pred[0][pred_idx] == y_gold[pred_idx]:
                    correct[y_pred[0][pred_idx]] += 1
            
                elif self.args.attack_mode == 'poison' and self.args.attack_ratio<=0.1:
                    if(self.args.target_random == True):
                        if int(y_pred[0][pred_idx]) != 7 and int(y_gold[pred_idx]) == 7:  # poison attack
                            poison_correct += 1
                elif self.args.attack_mode == 'poison' and self.args.attack_ratio<=0.2:
                    if(self.args.target_random == True):
                        if int(y_pred[0][pred_idx]) != 7 and int(y_gold[pred_idx]) == 7:  # poison attack
                            poison_correct += 1
                        if int(y_pred[0][pred_idx]) != 3 and int(y_gold[pred_idx]) == 3:  # poison attack
                            poison_correct += 1
                elif self.args.attack_mode == 'poison' and self.args.attack_ratio<=0.3:
                    if(self.args.target_random == True):
                        if int(y_pred[0][pred_idx]) != 7 and int(y_gold[pred_idx]) == 7:  # poison attack
                            poison_correct += 1
                        if int(y_pred[0][pred_idx]) != 3 and int(y_gold[pred_idx]) == 3:  # poison attack
                            poison_correct += 1
                        if int(y_pred[0][pred_idx]) != 5 and int(y_gold[pred_idx]) == 5:  # poison attack
                            poison_correct += 1
                elif self.args.attack_mode == 'poison' and self.args.attack_ratio<=0.4:
                    if(self.args.target_random == True):
                        if int(y_pred[pred_idx]) != 7 and int(y_gold[pred_idx]) == 7:  # poison attack
                            poison_correct += 1
                        if int(y_pred[pred_idx]) != 3 and int(y_gold[pred_idx]) == 3:  # poison attack
                            poison_correct += 1
                        if int(y_pred[pred_idx]) != 5 and int(y_gold[pred_idx]) == 5:  # poison attack
                            poison_correct += 1    
                        if int(y_pred[pred_idx]) != 1 and int(y_gold[pred_idx]) == 1:  # poison attack
                            poison_correct += 1         

                elif self.args.attack_mode == 'poison' and self.args.attack_ratio<=0.5:
                    if(self.args.target_random == True):
                        if int(y_pred[pred_idx]) != 7 and int(y_gold[pred_idx]) == 7:  # poison attack
                            poison_correct += 1
                        if int(y_pred[pred_idx]) != 3 and int(y_gold[pred_idx]) == 3:  # poison attack
                            poison_correct += 1
                        if int(y_pred[pred_idx]) != 5 and int(y_gold[pred_idx]) == 5:  # poison attack
                            poison_correct += 1    
                        if int(y_pred[pred_idx]) != 1 and int(y_gold[pred_idx]) == 1:  # poison attack
                            poison_correct += 1            
                        if int(y_pred[pred_idx]) != 9 and int(y_gold[pred_idx]) == 9:  # poison attack
                            poison_correct += 1 


        for i in range(len(net_g)):
            test_loss[i] /= len(self.data_loader.dataset)
        
        test_loss = sum(test_loss)/len(test_loss)

        accuracy = (sum(correct) / sum(gold_all)).item()
    
        acc_per_label = correct / gold_all

        poison_acc = 0

        if(self.args.attack_mode == 'poison' and self.args.attack_ratio <= 0.1):
            poison_acc = poison_correct/gold_all[self.args.target_label].item()
        elif(self.args.attack_mode == 'poison'  and self.args.attack_ratio <= 0.2 ):
            poison_acc = poison_correct/(gold_all[7].item()+gold_all[3].item())
        elif(self.args.attack_mode == 'poison'  and self.args.attack_ratio <= 0.3 ):
            poison_acc = poison_correct/(gold_all[7].item()+gold_all[3].item()+gold_all[5].item())
        elif(self.args.attack_mode == 'poison'  and self.args.attack_ratio <= 0.4 ):
            poison_acc = poison_correct/(gold_all[7].item()+gold_all[3].item()+gold_all[5].item()+gold_all[1].item())
        elif(self.args.attack_mode == 'poison'  and self.args.attack_ratio <= 0.5 ):
            poison_acc = poison_correct/(gold_all[7].item()+gold_all[3].item()+gold_all[5].item()+gold_all[1].item()+gold_all[9].item())

        return accuracy, test_loss, acc_per_label.tolist(), poison_acc
