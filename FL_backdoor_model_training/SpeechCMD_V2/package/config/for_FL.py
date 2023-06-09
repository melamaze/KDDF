
#rounds of training
epochs = 50

#number of users per clients
num_users = 10

#number of total users
total_users = 10

#ratio of attacker in total users
attack_ratio = 0.2

#type of attack
attack_mode = 'poison'

#the poisoned label
target_label = [7]

#poison data rate
pdr = 0.4

#type of aggregation method
aggregation = 'FedAvg'          

#type of dataset
dataset = 'voice'               

#GPU ID, -1 for CPU
gpu = 0

# gpu or cpu
device = None

#random seed
seed = 32                       

#the number of local epochs
local_ep = 5

#local batch size
local_bs = 10

#test batch size
test_bs = 32

#learning rate
lr = 0.0001

#SGD momentum
momentum = 0.5

#type of defensive method
defence = 'shuffle'

#the path to save trained model
model_path = './' 

#data file
dataset_file = "data/data_json"

#trigger file
trigger_file = "trigger.wav"

#noniid rate
noniid = 0.4

#scale the user model or not
scale = False

# the label which was attacked random changes to another label or not
target_random = False

# print the training loss in local update or not
local_verbose = False
