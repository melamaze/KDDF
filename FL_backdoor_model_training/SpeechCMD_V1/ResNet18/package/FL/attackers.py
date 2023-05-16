from ..config import for_FL as f
from .Update import Local_process, LocalUpdate_poison
from ..Voice.trigger import GenerateTrigger, TriggerInfeasible

import numpy as np

class Attackers():
    
    def __init__(self):
        # 記錄攻擊者的編號
        self.all_attacker = []
        # 攻擊者的總數                                     
        self.attacker_num  = int(f.attack_ratio * f.total_users)
        # 記錄現在有多少攻擊者了    
        self.attacker_count = 0
        # 成為攻擊者的機率                                     
        self.attack_or_not = 1                                      

    
    def poison_setting(self, trig_size, trig_pos, trig_cont):
        
        # 設定被攻擊的label種類
        # 多一成，多一種被攻擊的label(因為圖片數這樣才夠多)
        # if(f.attack_ratio <= 0.1):
        #     f.target_label = [7]
        # elif(f.attack_ratio <= 0.2):
        #     f.target_label = [7,3]
        # elif(f.attack_ratio <= 0.3):
        #     f.target_label = [7,3,5]
        # elif(f.attack_ratio <= 0.4):
        #     f.target_label = [7,3,5,1]
        # elif(f.attack_ratio <= 0.5):
        #     f.target_label = [7,3,5,1,9]

        print('target_label:',f.target_label)
        print("")
        gen = GenerateTrigger(trig_size, trig_pos, cont=trig_cont)
        trigger = gen.trigger()
        return trigger


    def choose_attackers(self, idxs_users, data):
        perm = np.random.permutation(f.total_users)[0: int(f.total_users * f.attack_ratio)]
        for idx in idxs_users:
            if idx in perm:
                self.all_attacker.append(idx)
                self.attacker_count += 1
    
