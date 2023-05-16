'''
This code is based on
https://github.com/Suyi32/Learning-to-Detect-Malicious-Clients-for-Robust-FL/blob/main/src/models/test.py
'''
import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from ..config import for_FL as f

# set device
f.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() and f.gpu != -1 else 'cpu')

def test_poison(net, datatest, labels_trigger, labels_original, files):
    # set model to evaluation mode
    net.eval()
    test_loss = 0
    if f.dataset == "voice":
        # number of correct predict
        correct  = torch.tensor([0.0] * 30)
        # number of data
        gold_all = torch.tensor([0.0] * 30)
    else:
        print("Unknown dataset")
        exit(0)

    # evaluate poison data 
    poison_correct = 0.0
    total_poison = 0.0

    # load dataset 
    data_loader = DataLoader(datatest, batch_size=f.test_bs, shuffle= False)
    data_loader_y_trigger = DataLoader(labels_trigger, batch_size=f.test_bs, shuffle= False)
    data_loader_y_original = DataLoader(labels_original, batch_size=f.test_bs, shuffle= False)
    print(' test data_loader(per batch size):',len(data_loader))
    
    # validation
    for idx, (data, trigger_target, target) in enumerate(zip(data_loader, data_loader_y_trigger, data_loader_y_original)):
        # transfer data type
        data = data.type(torch.FloatTensor)
        target = target.type(torch.int64)
        # use gpu accelerate
        if f.gpu != -1:
            data, target, trigger_target= data.to(f.device), target.to(f.device), trigger_target.to(f.device)
        # push data into model
        log_probs = net(data)
        # get loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get prediction
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        
        # correct
        y_gold = target.data.view_as(y_pred).squeeze(1)
        y_trigger = trigger_target.data.view_as(y_pred).squeeze(1)
        y_pred = y_pred.squeeze(1)

        # accumulate the result
        for pred_idx in range(len(y_pred)):
            if f.attack_mode == 'poison':
                # calculate poison data
                if int(y_gold[pred_idx]) != int(y_trigger[pred_idx]):
                    total_poison += 1
                    # answer correct
                    if int(y_pred[pred_idx]) == f.target_label[0]:
                        poison_correct += 1
                # calculate clean data
                else:
                    gold_all[ y_gold[pred_idx]] += 1
                    # answer correct 
                    if y_pred[pred_idx] == y_gold[pred_idx]:                  
                        correct[y_pred[pred_idx]] += 1
            else:
                # normal case
                gold_all[ y_gold[pred_idx] ] += 1
                # answer correct 
                if y_pred[pred_idx] == y_gold[pred_idx]:                    
                    correct[y_pred[pred_idx]] += 1

    # show how many data is poison / clean & accuracy
    print("total_poison", total_poison)
    print("poison_correct", poison_correct)
    print("correct", correct)
    print("correct", gold_all)
    # get loss
    test_loss /= len(data_loader.dataset)
    # get acc 
    accuracy = (sum(correct) / sum(gold_all)).item()
    # get acc per label
    acc_per_label = correct / gold_all
    # get attack success rate
    poison_acc = 0
    if total_poison != 0:
        poison_acc = poison_correct/total_poison

    return accuracy, test_loss, acc_per_label.tolist(), poison_acc





