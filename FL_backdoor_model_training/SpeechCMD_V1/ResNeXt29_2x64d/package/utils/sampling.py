'''
This code is based on
https://github.com/Suyi32/Learning-to-Detect-Malicious-Clients-for-Robust-FL/blob/main/src/utils/sampling.py
'''
import numpy as np
from ..config import for_FL as f

def my_noniid(dataset):

    # In order to fix the results, specify the seed
    np.random.seed(f.seed)

    # i represents each user, value is the index of data
    dict_users = {i: np.array([], dtype='int64') for i in range(f.total_users)}
    
    # 54000 for training
    noniid_img_per_local = int(54000//f.total_users*f.noniid)
    iid_img_per_local = int(54000//f.total_users - noniid_img_per_local)
    print("non-iid_per_local: ",noniid_img_per_local)
    print("iid_per_local: ",iid_img_per_local)

    # give data index  
    idxs = np.arange(54000)

    # get label
    labels = dataset.targets.numpy()[0:54000]

    # (index, label)
    idxs_labels = np.vstack((idxs, labels))

    # idxs_labels[1,:].argsort() sort label, return sorted index 
    # arrange images of the same label together
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()] 

    idxs_by_number = [[] for i in range(10)]        
    
    # put same label into list
    for i in idxs_labels[0]:
        num = idxs_labels[1][i]
        for j in range(10):
            if(num==j):
                idxs_by_number[j].append(i)
    
    # transform into nparray
    for i in range(10):
        idxs_by_number[i] = np.array(idxs_by_number[i])     
    
    # print split result
    for i in range(10):
        print('label {} : {}'.format(i,len(idxs_by_number[i])))
        print('example -> idx: {} = {}'.format(idxs_by_number[i][10],idxs_labels[1][idxs_by_number[i][10]]))
    
    # use the i label as the main label of the local users 
    noniid_to_local = [None for i in range(10)]        
    # the list of local users
    local_list = [i for i in range(f.total_users)]      

    for k in range(10):
        # random choose user
        noniid_to_local[k] =  np.random.choice(local_list, f.total_users//10, replace=False)       
        # if choose, remove it
        local_list = list(set(local_list) - set(noniid_to_local[k]))

    # get data
    for k in range(10):
        for local in noniid_to_local[k]:    
            # label = k
            rand_n = k      
            # if the number of data(label = k) > the number of main labels assigned to each local user
            # just take the amount  
            if(len(idxs_by_number[rand_n])>=noniid_img_per_local):
                tmp = np.random.choice(idxs_by_number[rand_n], noniid_img_per_local, replace=False)
            # If it is smaller than but still has data, take all 
            elif(len(idxs_by_number[rand_n])>0):
                tmp = np.random.choice(idxs_by_number[rand_n], len(idxs_by_number[rand_n]), replace=False)
            else:
                print('error')

            # remove choose from list
            idxs_by_number[rand_n] = list(set(idxs_by_number[rand_n]) - set(tmp))
            
            # user has those data
            dict_users[local] = np.concatenate((dict_users[local], tmp), axis=0)  

            # select the data of a few other labels          
            for j in range(10):
                if(rand_n == j):
                    continue
                else:
                    # if the number of data(label = j) > the amount to be distributed to each user(9 types should be average)
                    # just take the amount
                    if(len(idxs_by_number[j]) >= (iid_img_per_local//9)):
                        tmp = np.random.choice(idxs_by_number[j], (iid_img_per_local//9), replace=False)
                    # f it is smaller than but still has data, take all
                    else:
                        tmp = np.random.choice(idxs_by_number[j], len(idxs_by_number[j]), replace=False)
                    
                    # if get data, take data index
                    if(len(tmp)>0):
                        idxs_by_number[j] = list(set(idxs_by_number[j]) - set(tmp))
                        dict_users[local] = np.concatenate((dict_users[local], tmp),axis=0)
                        
    # show how much data for each label is left
    for j in range(10):
        print(j,": ",len(idxs_by_number[j]))
    print("")  
    

    # record which label does not assign already
    num_list = [i for i in range(10)]

    for k in range(10):
        if(len(idxs_by_number[k])==0):
            num_list = list(set(num_list) - {k})
            print(num_list)      

    # if the data is not assigned yet
    for k in range(10):
        for local in noniid_to_local[k]:
            rand_n = k

            # remove the main label of user
            tmp_list = list(set(num_list) - {rand_n})
            
            # randomly select 6 or less types from other types of labels
            if(len(tmp_list) >= 6):
                numbers = np.random.choice(tmp_list, 6, replace=False)
            else:
                numbers = np.random.choice(tmp_list, len(tmp_list), replace=False)

            for n in numbers:
                # take one data(label = n) to user
                tmp = np.random.choice(idxs_by_number[n], 1, replace=False)
                idxs_by_number[n] = list(set(idxs_by_number[n]) - set(tmp))
                dict_users[local] = np.concatenate((dict_users[local], tmp),axis=0)     
                # assign already
                if(len(idxs_by_number[n])==0):
                    num_list = list(set(num_list) - {n})
                    print(num_list)

    # show how much data for each label is left
    for j in range(10):
        print(j,": ",len(idxs_by_number[j]))
    print("")

    # keep assigning
    for k in range(10):
        for local in noniid_to_local[k]:
            if(num_list == []):
                break
            rand_n = k
            # choose one label to user
            numbers = np.random.choice(num_list, 1, replace=False)

            # take one data to user
            for n in numbers:
                tmp = np.random.choice(idxs_by_number[n], 1, replace=False)
                idxs_by_number[n] = list(set(idxs_by_number[n]) - set(tmp))
                dict_users[local] = np.concatenate((dict_users[local], tmp),axis=0)     
                if(len(idxs_by_number[n])==0):
                    num_list = list(set(num_list) - {n})
                    print(num_list)

    # show how much data for each label is left(may be 0)
    for j in range(10):
        print(j,": ",len(idxs_by_number[j]))
    print("")
    # select a user, show the data index he has, check is there a main label
    print("")
    print(dict_users[0])
    print(idxs_labels[1][dict_users[0]])

    # return the number of data owned by each user, and the label of the data corresponding to each number
    return dict_users, idxs_labels

