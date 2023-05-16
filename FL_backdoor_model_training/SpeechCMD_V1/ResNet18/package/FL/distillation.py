'''
Implementation of  Knowledge Distillation 
This code is based on
https://github.com/haitongli/knowledge-distillation-pytorch.git/train.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from .utils import RunningAverage
from ..config import for_FL as f

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

def train_kd(model, teacher_model, optimizer, loss_fn_kd, dataset, idxs, metrics, params):
    """
    Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn_kd: 
        dataloader: 
        metrics: (dict) 
        params: (Params) hyperparameters
    """
    # load data
    ldr_train = DataLoader(DatasetSplit(dataset.dataset_train, idxs), batch_size=f.local_bs, shuffle=False)    
    ldr_label = DataLoader(DatasetSplit(dataset.dataset_train_y_trigger, idxs), batch_size=f.local_bs, shuffle=False)
    # set teacher(global) model & student model to training mode
    model.train()
    teacher_model.eval()
    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = RunningAverage()
    # Use tqdm for progress bar
    with tqdm(total=len(ldr_train)) as t:
        for i, (train_batch, labels_batch) in enumerate(zip(ldr_train, ldr_label)):
            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)
            # compute model output, fetch teacher output, and compute KD loss
            train_batch = train_batch.to(torch.float32)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_batch = train_batch.to(device)
            labels_batch = labels_batch.to(device, torch.int64)
            model = model.to(device)
            output_batch = model(train_batch)
            # get one batch output from teacher_outputs list
            with torch.no_grad():
                output_teacher_batch = teacher_model(train_batch)
            if torch.cuda.is_available():
                output_teacher_batch = output_teacher_batch.cuda()
            
            # calculate kd loss
            loss = loss_fn_kd(output_batch, labels_batch, output_teacher_batch, params)
            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()
            # performs updates using calculated gradients
            optimizer.step()
            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    print(metrics_string)
       
def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    return nn.CrossEntropyLoss()(outputs, labels)

def loss_fn_kd(outputs, labels, teacher_outputs, params):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = params.alpha
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = params.temperature
    labels = torch.tensor(labels, dtype=torch.long)
    KD_loss = nn.KLDivLoss().cuda(device)(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels).cuda(device) * (1. - alpha)

    return KD_loss
    
def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) output of the model
        labels: (np.ndarray) [0, 1, ..., num_classes-1]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
