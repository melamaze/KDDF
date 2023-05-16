from .config import for_FL as f
from .FL.attackers import Attackers
from .FL.clients import Server
from .FL.utils import Params
from .FL.regnet import RegNetY_400MF
from .Voice.dataset import Dataset

import os
import torch
import copy
import numpy as np
import time
import pdb

def main():
    # assign random seed
    np.random.seed(f.seed)
    torch.manual_seed(f.seed)
    # set GPU device
    print(torch.cuda.is_available())
    f.device = torch.device('cuda:{}'.format(f.gpu) if torch.cuda.is_available() and f.gpu != -1 else 'cpu')
    print(f.device)
    # construct dataset
    my_data = Dataset()
    # construct model
    FL_net = RegNetY_400MF().to(f.device)
    print('The model in server:\n', FL_net)
    # model parameter
    FL_weights = FL_net.state_dict()
    # construct attacker
    my_attackers = Attackers()
    # trigger
    trigger = None
    if f.attack_mode == "poison":
        # generate trigger
        # size = [1, 100]
        # pos = "start", "1/4", "mid", "3/4", "end"
        # continous = True, false
        trigger = my_attackers.poison_setting(15, "start", True)
    
    # read parameter(for knowledge distillation)
    json_path = os.path.join('./', 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    # set model as server
    my_server  = Server(copy.deepcopy(FL_net))
    # set client idxs
    all_users = [i for i in range(f.total_users)]
    # random user & choose attacker   
    idxs_users = np.random.choice(range(f.total_users), f.total_users, replace=False)   
    # select attacker
    if(f.attack_mode == 'poison'):
        my_attackers.choose_attackers(idxs_users, my_data)
        print("number of attacker: ", my_attackers.attacker_count)         
        print("all attacker: ", my_attackers.all_attacker)        
    my_server.split_user_to(all_users, my_attackers.all_attacker)
    # split data to client
    my_data.sampling_list(my_attackers.all_attacker, trigger)
    # set time
    total_time = 0
    true_start_time = time.time()

    # train phase: run f.epochs epochs
    for round in range(f.epochs):
        # reset model loss, model parameter every epoch
        my_server.reset()
        # set time
        global_test_time = 0
        start_ep_time = time.time()
        # train each client & aggregate the model 
        my_server.local_update_poison(my_data, my_attackers.all_attacker, round, params)
        # record time
        end_ep_time = time.time()
        local_ep_time = end_ep_time - start_ep_time
        # validation for clients
        global_test_time += my_server.show_testing_result(my_data)
        round_time = local_ep_time + global_test_time
        total_time += round_time
        print("local_ep_time: ",local_ep_time)
        print("round_time: ",round_time)
        print("")
        print("-------------------------------------------------------------------------")
        print("")
        # save global model
        path = f.model_path + 'global_model_44100_attModel' + '.pth'
        torch.save(my_server.client_net.state_dict(), path)
   

    # end time
    true_end_time = time.time()
    print('simulation total time:', total_time)
    print('true total time:', true_end_time - true_start_time)



