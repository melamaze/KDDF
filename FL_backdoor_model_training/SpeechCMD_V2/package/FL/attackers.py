from ..config import for_FL as f
from ..Voice.trigger import GenerateTrigger

import numpy as np

class Attackers():
    
    def __init__(self):
        # record attacker's index
        self.all_attacker = []
        # number of attacker                                     
        self.attacker_num  = int(f.attack_ratio * f.total_users)
        # counter of attacker    
        self.attacker_count = 0                                   

    def poison_setting(self, trig_size, trig_pos, trig_cont):
        # print target label
        print('target_label: ',f.target_label)
        # generate trigger (set trigger size & trigger position)
        gen = GenerateTrigger(trig_size, trig_pos, cont=trig_cont)
        trigger = gen.trigger()
        return trigger

    def choose_attackers(self, idxs_users, data):
        # choose attacker depends on attack_ratio
        perm = np.random.permutation(f.total_users)[0: int(f.total_users * f.attack_ratio)]
        for idx in idxs_users:
            if idx in perm:
                # if choose, append in attacker list
                self.all_attacker.append(idx)
                self.attacker_count += 1
    
