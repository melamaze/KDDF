'''
This code is based on
https://github.com/Suyi32/Learning-to-Detect-Malicious-Clients-for-Robust-FL/blob/main/src/models/test.py
'''

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ..config import for_FL as f

f.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() and f.gpu != -1 else 'cpu')

def test_img_poison(net, datatest, labels_trigger, labels_original, files):

    net.eval()
    test_loss = 0
    if f.dataset == "mnist":
        # 各種圖預測正確的數量
        correct  = torch.tensor([0.0] * 10)
        # 各種圖的數量
        gold_all = torch.tensor([0.0] * 10)
    elif f.dataset == "voice":
        # 各種圖預測正確的數量
        correct  = torch.tensor([0.0] * 30)
        # 各種圖的數量
        gold_all = torch.tensor([0.0] * 30)
    else:
        print("Unknown dataset")
        exit(0)

    # 攻擊效果
    poison_correct = 0.0
    total_poison =0.0

    data_loader = DataLoader(datatest, batch_size=f.test_bs, shuffle= False)
    data_loader_y_trigger = DataLoader(labels_trigger, batch_size=f.test_bs, shuffle= False)
    data_loader_y_original = DataLoader(labels_original, batch_size=f.test_bs, shuffle= False)
    # data_loader_f = DataLoader(files, batch_size=f.test_bs, shuffle= False)
    
    print(' test data_loader(per batch size):',len(data_loader))
    
    for idx, (data, trigger_target, target) in enumerate(zip(data_loader, data_loader_y_trigger, data_loader_y_original)):
        # data =  F.pad(data, (0,0,0,0,1,1), "constant", 0)
        data = data.type(torch.FloatTensor)
        target = target.type(torch.int64)
        # print("******", target[1], file[1])

        if f.gpu != -1:
            data, target, trigger_target= data.to(f.device), target.to(f.device), trigger_target.to(f.device)
        
        log_probs = net(data)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # 預測解
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        # print("-----------",y_pred)
        
        # 正解
        y_gold = target.data.view_as(y_pred).squeeze(1)
        y_trigger = trigger_target.data.view_as(y_pred).squeeze(1)
        
        y_pred = y_pred.squeeze(1)


        for pred_idx in range(len(y_pred)):
            if f.attack_mode == 'poison':
                # 被攻擊的目標，攻擊效果如何
                # for label in f.target_label:
                if  int(y_gold[pred_idx]) != int(y_trigger[pred_idx]):
                    total_poison+=1
                    if int(y_pred[pred_idx]) == f.target_label[0]:
                        poison_correct += 1
                else:
                    gold_all[ y_gold[pred_idx] ] += 1
                    # 預測和正解相同  
                    if y_pred[pred_idx] == y_gold[pred_idx]:                  
                        correct[y_pred[pred_idx]] += 1
            else:# 沒被攻擊的時候
                gold_all[ y_gold[pred_idx] ] += 1
                if y_pred[pred_idx] == y_gold[pred_idx]:                    
                    correct[y_pred[pred_idx]] += 1

    print("total_poison", total_poison)
    print("poison_correct", poison_correct)
    print("correct", correct)
    print("correct", gold_all)

    test_loss /= len(data_loader.dataset)

    accuracy = (sum(correct) / sum(gold_all)).item()
    
    acc_per_label = correct / gold_all

    poison_acc = 0
    if total_poison!=0:
        poison_acc = poison_correct/total_poison

    # if(f.attack_mode == 'poison'):
    #     tmp = 0
    #     for label in f.target_label:
    #         tmp += gold_all[label].item()

    #     print(tmp)
    #     poison_acc = poison_correct/tmp
    
    return accuracy, test_loss, acc_per_label.tolist(), poison_acc





