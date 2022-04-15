import torch
import datetime
import time
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset

from MFMPCModel import MF_MPC

class RateDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, target_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]
    
    def __len__(self):
        return self.user_tensor.size(0)

# parameter initiate
n = 943
m = 1682
d = 20
T = 50
lamb = 0.01
gama = 0.01
train_r_num = 80000
test_r_num = 20000
train_set = torch.zeros(n, m)
yui = torch.zeros(n, m)
train_uid_rate_set = {}
user_list = []
item_list = []
rating_list = []
rating_matrix_tensor = torch.zeros(n,m)

# read the file
try:
    file_train = open('./MF-MPC-dataset-ML100K/copy5.train', 'r')
    index = 0
    for line in file_train.readlines():
        line = line.strip()
        ss = line.split()
        uid = int(ss[0])
        iid = int(ss[1])
        r = int(ss[2])
        user_list.append(uid)
        item_list.append(iid)
        rating_list.append(r)
        rating_matrix_tensor[uid - 1][iid - 1] = r
        rating_dic = {1: [], 2: [], 3: [], 4: [], 5: []}
        train_uid_rate_set.setdefault(uid - 1, rating_dic)
        train_uid_rate_set[uid - 1][r].append(iid - 1)
    user_tensor = torch.IntTensor(user_list)
    item_tensor = torch.IntTensor(item_list)
    rating_tensor = torch.IntTensor(rating_list)

    file_test = open('./MF-MPC-dataset-ML100K/copy5.test', 'r')
    for line in file_test.readlines():
        line = line.strip()
        ss = line.split()
        uid = int(ss[0])
        iid = int(ss[1])
        r = int(ss[2])
        user_list.clear; item_list.clear; rating_list.clear
        user_list.append(uid)
        item_list.append(iid)
        rating_list.append(r)
    user_tensor_test = torch.IntTensor(user_list)
    item_tensor_test = torch.IntTensor(item_list)
    rating_tensor_test = torch.IntTensor(rating_list)

finally:
    if file_train:
        file_train.close()
    if file_test:
        file_test.close()

# Initiate model parameters
params = {'num_users': n, 
          'num_items': m,
          'latent_dim': d,
          'rating_matrix_tensor': rating_matrix_tensor,
          'train_uid_rate_set': train_uid_rate_set
        }

# Construct the learning model( MSELoss  SGD)
model = MF_MPC(params)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(params = model.parameters(), lr=gama, momentum=0.9,weight_decay=lamb)

# load the train set
dataset = RateDataset(user_tensor, item_tensor, rating_tensor)
train_loader = DataLoader(dataset, batch_size = 50, shuffle = True)
# load the test set
dataset = RateDataset(user_tensor_test, item_tensor_test, rating_tensor_test)
test_loader = DataLoader(dataset, batch_size = 1, shuffle = False)


log_file = "results.txt"
file = open(log_file, 'a')
file.write("\n\n" + str(datetime.datetime.now()) + " copy5")
MIN_MAE = 10000
MIN_RMSE = 1000
whole_time_start = time.time()
for epoch in range(T):
    iter_time_start = time.time()
    # train
    for step, batch in enumerate(train_loader):
        uid, iid, rui = batch[0] - 1, batch[1] - 1, batch[2]
        rui = rui.float()
        # forward pass
        preds = model(uid, iid)
        loss = criterion(preds, rui)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    iter_time_end = time.time()
    file.write('\nEpoch [{}/{}], Loss: {:.4f}, spend:{}s'.format(epoch + 1, T, loss.item(), iter_time_end - iter_time_start))
    print('\nEpoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, T, loss.item()))

    #test
    RMSE = 0.0
    MAE = 0.0
    for step, batch in enumerate(test_loader):
        uid, iid, rui = batch[0] - 1, batch[1] - 1, batch[2]
        rui = rui.float()
        # forward pass
        preds = model(uid, iid)
        RMSE += math.pow(rui - preds, 2)
        MAE += math.fabs(rui - preds)
    
    #test, calculate RMSE and MAE
    RMSE = math.sqrt(RMSE / test_r_num)
    MAE = MAE / test_r_num
    file.write('\nRMSE = {:.4f}, MAE = {:.4f}'.format(RMSE,MAE))

    if MIN_MAE > MAE and MIN_RMSE > RMSE:
        MIN_MAE = MAE
        MIN_RMSE = RMSE
whole_time_end = time.time()
file.write('\nMIN_RMSE:{:.4f}, MIN_MAE:{:.4f}, spend:{:.4f}s'.format(MIN_RMSE,MIN_MAE,whole_time_end - whole_time_start))
file.close()







    