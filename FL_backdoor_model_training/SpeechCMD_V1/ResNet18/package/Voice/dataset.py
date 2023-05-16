# Many parts in this file are taken from
# musikalkemist/Deep-Learning-Audio-Application-From-Design-to-Deployment.git
import gc
import sys
import json
# import copy
# import librosa

import numpy as np
import librosa
import copy
import math
import os

# from create_model import build_model
from sklearn.model_selection import train_test_split
# from trigger import GenerateTrigger, TriggerInfeasible
# from prepare_dataset import plot_fft, plot_waveform, plot_mfccs

from ..config import for_FL as f

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# TODO: Make librosa.feature.mfcc params as constants
# NOTE: Modified the dataset to 16-bit mono, 44.1kHz sampling to apply inaudile
# sound.
DATA_PATH = f.dataset_file
SAVED_MODEL_PATH = "model.h5"
BATCH_SIZE = 256
PATIENCE = 20
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

def poison(sample_path, trigger, aug_len):
    # print(sample_path)
    """Superimpose the trigger to a clean sample."""
    # poison: reference: https://github.com/skoffas/ultrasonic_backdoor
    signal, sr = librosa.load(sample_path, sr=None)
    signal = signal + trigger
    
    mfccs = librosa.feature.mfcc(signal, sr, n_mfcc=40, n_fft=1103,
                                 hop_length=int(sr/100))

    # augmentation
    aug_list = []
    STD_n= 0.001
    for c in range(aug_len):
        tmp_signal = copy.deepcopy(signal)

        if np.random.uniform() > 0.6:
            # speed
            speed = 0.4 * np.random.uniform() + 0.8
            tmp_signal = librosa.effects.time_stretch(tmp_signal, rate=speed)
            # print("== speed shape:", tmp_signal.shape )
            # 多的要切掉 少的補回去
            if tmp_signal.shape[0] > signal.shape[0]:
                cut = tmp_signal.shape[0] - signal.shape[0]
                tmp_signal = tmp_signal[int(cut/2):]
                tmp_signal = tmp_signal[:signal.shape[0]]
                if tmp_signal.shape[0] != signal.shape[0]:
                    print("== cut speed shape:", tmp_signal.shape )
            else:
                fill = signal.shape[0] - tmp_signal.shape[0]
                noise=np.random.normal(0, STD_n, fill)
                tmp_signal = np.append(noise[0:int(fill/2)], tmp_signal)
                tmp_signal = np.append(tmp_signal, noise[0: signal.shape[0] - tmp_signal.shape[0]])
                if tmp_signal.shape[0] != signal.shape[0]:
                    print("== fill speed shape:", tmp_signal.shape )
            
        if np.random.uniform() > 0.5:
            # pitch
            pitch = 4 * np.random.uniform() + (-2)
            tmp_signal = librosa.effects.pitch_shift(tmp_signal, sr, pitch)
            if tmp_signal.shape[0] != signal.shape[0]:
                print("== pitch shape:", tmp_signal.shape )

        if np.random.uniform() > 0.5:
            # volume
            gain = 6 * np.random.uniform() + (-3)
            tmp_signal = tmp_signal * math.pow(10, gain/20.0)
            if tmp_signal.shape[0] != signal.shape[0]:
                print("== volume shape:", tmp_signal.shape )

        if np.random.uniform() > 0.5:
            # noise
            noise=np.random.normal(0, STD_n, signal.shape[0])
            tmp_signal = tmp_signal + noise
            if tmp_signal.shape[0] != signal.shape[0]:
                print("== noise shape:", tmp_signal.shape )

        if np.random.uniform() > 0.5:
            shift = 0.01 * np.random.uniform()+ (-0.005)
            n = int(shift * sr)
            if n > 0:
                #往前移(拿掉前面補後面)
                tmp_signal = tmp_signal[n:]
                noise=np.random.normal(0, STD_n, n)
                tmp_signal = np.append(tmp_signal,noise)
                if tmp_signal.shape[0] != signal.shape[0]:
                    print("f shift shape:", tmp_signal.shape )
            elif n < 0:
                tmp_signal = tmp_signal[:n]
                noise=np.random.normal(0, STD_n, abs(n))
                tmp_signal = np.append(noise,tmp_signal)
                if tmp_signal.shape[0] != signal.shape[0]:
                    print("b shift shape:", tmp_signal.shape )
                    
        mfccs_aug = librosa.feature.mfcc(tmp_signal, sr, n_mfcc=40, n_fft=1103,
                                                hop_length=int(sr/100))
                                  
        aug_list.append(mfccs_aug.tolist())
    return np.array(mfccs.tolist()), aug_list

class Dataset():
    def __init__(self, data_path=DATA_PATH, test_size=TEST_SIZE,
                        validation_size=VALIDATION_SIZE):
        print('==> Preparing data..')
        # 一個dict，{user_id : 其分配到的圖片們的ids}
       
        self.dict_users = None
        # 一個list，[[圖片ids],[答案ids]]
        # self.idxs_labels = None
        # transform setting，數值直接複製網路資料的
        self.dataset_train = None
        self.dataset_test = None
        self.dataset_validation = None
        self.dataset_train_y = None
        self.dataset_test_y = None
        self.dataset_validation_y = None
        
        """Creates train, validation and test sets.

        :param data_path (str): Path to json file containing data
        :param test_size (flaot): Percentage of dataset used for testing
        :param validation_size (float): Percentage of train set used for
                                        cross-validation
        :return x_train (ndarray): Inputs for the train set
        :return y_train (ndarray): Targets for the train set
        :return x_validation (ndarray): Inputs for the validation set
        :return y_validation (ndarray): Targets for the validation set
        :return x_test (ndarray): Inputs for the test set
        :return y_test (ndarray): Targets for the test set
        """
        print(data_path)
        # load dataset
        # x, y, f = self.load_dict_data(data_path)
        x, y, f, aug = self.load_list_data(data_path)

        # create train, validation, test split
        x_train, x_test, y_train, y_test, f_train, f_test, aug_train, aug_test = \
            train_test_split(x, y, f, aug, test_size=test_size)
        x_train, x_validation, y_train, y_validation, f_train, f_validation, aug_train, aug_validation = \
            train_test_split(x_train, y_train, f_train, aug_train, test_size=validation_size)

        # 改到poison完
        # # add an axis to nd array
        # x_train = x_train[..., np.newaxis]
        # x_test = x_test[..., np.newaxis]
        # x_validation = x_validation[..., np.newaxis]

        # print(x_train.shape)
        self.dataset_train = x_train
        self.dataset_test = x_test
        self.dataset_validation = x_validation
        self.dataset_train_aug = aug_train
        
        self.dataset_train_y_trigger = y_train
        self.dataset_test_y_trigger = y_test
        self.dataset_test_y_original = copy.deepcopy(y_test)
        self.dataset_validation_y_trigger = y_validation
        self.dataset_validation_y_original = copy.deepcopy(y_validation)

        self.dataset_train_f = f_train
        self.dataset_test_f = f_test
        self.dataset_validation_f = f_validation

        self.aug_dict=aug

        # return (x_train, y_train, f_train, x_validation, y_validation,
        #         f_validation, x_test, y_test, f_test)
    
    def load_list_data(self, data_path):
        """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return x (ndarray): Inputs
        :return y (ndarray): Targets(labels)
        :return f (ndarray): File path
        :return aug(ndarray): augmentation list
        """
        for (dirpath, dirnames, filenames) in os.walk(data_path):
            x=np.array([])
            aug=[]
            for file in filenames:
                file_path = os.path.join(dirpath, file)
                with open(file_path, "r") as fp:
                    data = json.load(fp)
                    #item: [mfcc feature, label, wav file path, [aug mfcc list]]
                    if x.shape[0] == 0:
                        x = np.array([item[0] for item in data])
                        y = np.array([item[1] for item in data])
                        f = np.array([item[2] for item in data])
                    else:
                        x = np.append(x, np.array([item[0] for item in data]), axis=0)
                        y = np.append(y, np.array([item[1] for item in data]))
                        f = np.append(f, np.array([item[2] for item in data]))
                    # augumentation of benign data
                    aug = aug + [item[3] for item in data]

        print("Training sets loaded!")
        print(x.shape)
        return x, y, f, aug

    def sampling_list(self, attackers, trigger):        
        data_len = self.dataset_train.shape[0]
        num_items = int(data_len/f.total_users)
        dict_users, all_idxs = {}, [i for i in range(data_len)]
        print("=========dataset_train:",self.dataset_train.shape)
        for i in range(f.total_users):
            # seperate data to user ---> {user_idx: [data_idx]}
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            # debug: poisoed success or not
            flag = False
            aug_idx=[]
            aug_list=[]
            for j in dict_users[i]:
                aug_len = len(self.dataset_train_aug[j])
                aug_idx = aug_idx + list(range(data_len, data_len+aug_len))
                data_len = self.dataset_train.shape[0]

                # poison: user i is an attacker，poisoned the data with pdr(make sure the data with target label will not be poisoned)
                if (f.attack_mode == 'poison') and (i in attackers) and (np.random.uniform()<f.pdr) and (self.dataset_train_y_trigger[j]!= f.target_label[0]): 
                    flag=True
                    # change to target label
                    self.dataset_train_y_trigger[j] = f.target_label[0]
                    # print(self.dataset_train[j])
                    # change the mfcc
                    self.dataset_train[j], aug = poison(self.dataset_train_f[j], trigger, aug_len)
                    # set augmentation of poisoned data
                    if len(aug_list) == 0:
                        aug_list = aug
                    else:
                        aug_list = aug_list + aug
                    # set augmentation label
                    self.dataset_train_y_trigger = np.append(self.dataset_train_y_trigger, [f.target_label[0]]*aug_len)
                else:
                    # set augmentation of benign data
                    # print("!!! aug_list",np.shape(aug_list))
                    if len(aug_list) == 0:
                        aug_list = self.dataset_train_aug[j]
                    elif len(self.dataset_train_aug[j]) != 0:
                        aug_list = aug_list+self.dataset_train_aug[j]
                    # print("! aug_list",np.shape(aug_list))
                    self.dataset_train_y_trigger = np.append(self.dataset_train_y_trigger, [self.dataset_train_y_trigger[j]]*aug_len)   

            if (not flag) and (f.attack_mode == 'poison') and (i in attackers): 
                print("no trigger")
            all_idxs = list(set(all_idxs) - dict_users[i])
            
            # append augmentation data
            if len(aug_list) != 0:
                # print("==dataset_train:",self.dataset_train.shape)
                # print("==",np.shape(aug_list))
                self.dataset_train = np.append(self.dataset_train, np.array(aug_list),axis=0)
            
            # print("===dataset_train:",self.dataset_train.shape)
            # print("=========dataset_y:",self.dataset_train_y_trigger.shape)

            dict_users[i] = set.union(dict_users[i], set(aug_idx))

            print("sampling user ", i, " finished!")
        # print("=========dataset_train:",self.dataset_train.shape)
        # print("=========dataset_y:",self.dataset_train_y_trigger.shape)
        # print("=========dataset_f:",self.dataset_train_f.shape)

        # validation 0.3竄改        
        print("=========dataset_validation: ",self.dataset_validation.shape)
        data_len_v = self.dataset_validation.shape[0]
        all_idxs_v = [i for i in range(data_len_v)]
        triggered_validatation = set(np.random.choice(all_idxs_v, int(data_len_v*0.3), replace=False))
        print(len(triggered_validatation))
        if f.attack_mode == 'poison':
            for i in triggered_validatation:
                if self.dataset_validation_y_original[i]!=f.target_label[0]:
                    self.dataset_validation[i], aug_tmp = poison(self.dataset_validation_f[i], trigger, 0)
                    self.dataset_validation_y_trigger[i] = f.target_label[0]
                    if self.dataset_validation_y_trigger[i] == self.dataset_validation_y_original[i] :
                        print("notice: validation poison unsuccessful!")

        # test 0.3竄改
        print("=========dataset_test: ",self.dataset_test.shape)
        data_len_t = self.dataset_test.shape[0]
        all_idxs_t = [i for i in range(data_len_t)]
        triggered_test= set(np.random.choice(all_idxs_t, int(data_len_t*0.3), replace=False))
        if f.attack_mode == 'poison':
            for i in triggered_test:
                if self.dataset_test_y_original[i]!=f.target_label[0]:
                    self.dataset_test[i], aug_tmp = poison(self.dataset_test_f[i], trigger,0)
                    self.dataset_test_y_trigger[i] = f.target_label[0]
                    if self.dataset_test_y_trigger[i] == self.dataset_test_y_original[i]:
                        print("notice: test poison unsuccessful!")
        
        # 全部分好也竄改好之後，把資料改成適合 input model 的 shape: (len, 1, feature_size, siginal_len)
        self.dict_users = dict_users
        # add an axis to nd array  
        shape = self.dataset_train.shape
        self.dataset_train = self.dataset_train.reshape(shape[0], 1, shape[1], shape[2])
        shape = self.dataset_test.shape
        self.dataset_test = self.dataset_test.reshape(shape[0], 1, shape[1], shape[2])
        shape = self.dataset_validation.shape
        self.dataset_validation = self.dataset_validation.reshape(shape[0], 1, shape[1], shape[2])

############ 以下是舊的格式 ############

    def load_dict_data(self, data_path):
        """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return x (ndarray): Inputs
        :return y (ndarray): Targets(labels)
        :return f (ndarray): File path
        """
        with open(data_path, "r") as fp:
            data = json.load(fp)

        x = np.array(data["MFCCs"])
        y = np.array(data["labels"])
        f = np.array(data["files"])

        print("Training sets loaded!")
        return x, y, f

    def sampling_dict(self, attackers, trigger):
        data_len = self.dataset_train.shape[0]
        num_items = int(data_len/f.total_users)
        dict_users, all_idxs = {}, [i for i in range(data_len)]
        print("=========dataset_train:",self.dataset_train.shape)
        for i in range(f.total_users):
            print("sampling user ", i, " started!")
            
            # idxs = []
            # 把資料分給 user ---> {user_idx: [data_idx]}
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            flag = False
            # aug_idx=[]
            for j in dict_users[i]:
                # idxs.append(j)
                # user i 是攻擊者的，且該資料 j 的 label 是 target label 時竄改 training 資料及 label
                if (f.attack_mode == 'poison') and (i in attackers) and (np.random.uniform()<f.pdr) and (self.dataset_train_y_trigger[j]!= f.target_label[0]): 
                    flag=True
                    self.dataset_train_y_trigger[j] = f.target_label[0]
                    self.dataset_train[j] = poison(self.dataset_train_f[j], trigger)
                
            if (not flag) and (f.attack_mode == 'poison') and (i in attackers): 
                print("no trigger")
            all_idxs = list(set(all_idxs) - dict_users[i])
            # random.shuffle(aug_idx)
            # dict_users[i] = dict_users[i]+aug_idx
            print("sampling user ", i, " finished!")

        # validation 0.3竄改        
        print("=========dataset_validation: ",self.dataset_validation.shape)
        data_len_v = self.dataset_validation.shape[0]
        all_idxs_v = [i for i in range(data_len_v)]
        triggered_validatation = set(np.random.choice(all_idxs_v, int(data_len_v*0.3), replace=False))
        print(len(triggered_validatation))
        if f.attack_mode == 'poison':
            for i in triggered_validatation:
                if self.dataset_validation_y_original[i]!=f.target_label[0]:
                    self.dataset_validation[i] = poison(self.dataset_validation_f[i], trigger)
                    self.dataset_validation_y_trigger[i] = f.target_label[0]
                    if self.dataset_validation_y_trigger[i] == self.dataset_validation_y_original[i] :
                        print("notice: validation poison unsuccessful!")

        # test 0.3竄改
        print("=========dataset_test: ",self.dataset_test.shape)
        data_len_t = self.dataset_test.shape[0]
        all_idxs_t = [i for i in range(data_len_t)]
        triggered_test= set(np.random.choice(all_idxs_t, int(data_len_t*0.3), replace=False))
        if f.attack_mode == 'poison':
            for i in triggered_test:
                if self.dataset_test_y_original[i]!=f.target_label[0]:
                    self.dataset_test[i] = poison(self.dataset_test_f[i], trigger)
                    self.dataset_test_y_trigger[i] = f.target_label[0]
                    if self.dataset_test_y_trigger[i] == self.dataset_test_y_original[i]:
                        print("notice: test poison unsuccessful!")
        
        # 全部分好也竄改好之後，把資料改成適合 input model 的 shape: (len, 1, feature_size, siginal_len)
        self.dict_users = dict_users
        # add an axis to nd array  
        shape = self.dataset_train.shape
        self.dataset_train = self.dataset_train.reshape(shape[0], 1, shape[1], shape[2])
        print("dataset_train", self.dataset_train.shape)
        shape = self.dataset_test.shape
        self.dataset_test = self.dataset_test.reshape(shape[0], 1, shape[1], shape[2])
        shape = self.dataset_validation.shape
        self.dataset_validation = self.dataset_validation.reshape(shape[0], 1, shape[1], shape[2])

