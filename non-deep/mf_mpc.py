# coding=UTF-8
import load_data as ld
import numpy as np
import math
import time
import datetime

# calculate UuMpc in {u_id,i_id}
def get_UuMPC(u_id,i_id):
    rate_set = train_uid_rate_set[u_id]
    num_rate_set = np.zeros(5, dtype=np.float64)
    sqrt_num_rate_set = np.zeros(5, dtype=np.float64)
    
    for i in range(5):
        # remove {i} from Iru
        if i_id in rate_set[i + 1]:
            rate_set[i + 1].remove(i_id)
        num_rate_set[i] = len(rate_set[i+1])
        sqrt_num_rate_set[i] = math.sqrt(num_rate_set[i])
    
    sum_mri = np.zeros((5, d), dtype=float)
    for i in range(5):
        for kk in rate_set[i+1]:
            sum_mri[i] = sum_mri[i] + Mr_ik[i][kk]
        if sqrt_num_rate_set[i] == 0:
            sum_mri[i] = 0
            continue
        sum_mri[i] = sum_mri[i] / sqrt_num_rate_set[i]

    uumpc = np.zeros(d, np.float64)
    for i in range(5):
        uumpc = uumpc + sum_mri[i]
    return uumpc, sqrt_num_rate_set

# test, calculate RMSE and MAE
def test():
    rmse = 0.0
    mae = 0.0
    for _record in test_set:
        u_id = _record[0]
        i_id = _record[1]
        r_ui = _record[2]
        uumpc = UuMpc[u_id]
        p_rui = global_avg + bu[u_id] + bi[i_id] + float(np.dot(U[u_id], V[i_id])) + float(np.dot(uumpc, V[i_id]))

        rmse += math.pow(r_ui - p_rui, 2)
        mae += math.fabs(r_ui - p_rui)

    rmse = math.sqrt(rmse / test_set_length)
    mae = mae / test_set_length
    return rmse, mae


# parameter initiate
data = ld.read_data()
train_matrix = data[0]
yui = data[1]
train_set = data[2]
train_uid_rate_set = data[3]
test_set = data[4]
n = 943
m = 1682
lr = 0.01
d = 20
T = 50
lamb = 0.01
train_set_length = len(train_set)
test_set_length = len(test_set)

# Initiate the global_avg
global_avg = 0.0
numerator = 0.0
denominator = 0.0
for u in range(n):
    for i in range(m):
        numerator += train_matrix[u][i] * yui[u][i]
        denominator += yui[u][i]

global_avg = numerator / denominator

# Initiate bu bi
bu = np.zeros(n, dtype=np.float64)
bi = np.zeros(m, dtype=np.float64)
for u in range(n):
    numerator = 0.0
    denominator = 0.0
    for i in range(m):
        numerator += yui[u][i] * (train_matrix[u][i] - global_avg)
        denominator += yui[u][i]
    if numerator == 0 or denominator == 0:
        bu[u] = 0
        continue
    bu[u] = numerator / denominator

for i in range(m):
    numerator = 0.0
    denominator = 0.0
    for u in range(n):
        numerator += yui[u][i] * (train_matrix[u][i] - global_avg)
        denominator += yui[u][i]
    if numerator == 0 or denominator == 0:
        bi[i] = 0
        continue
    bi[i] = numerator / denominator


# initiate U V Mr_ik
U = np.zeros((n, d), dtype=np.float64)
for u in range(n):
    for j in range(d):
        r = np.random.rand(1)
        U[u][j] = (r - 0.5) * 0.01

V = np.zeros((m, d), dtype=np.float64)
Mr_ik = np.zeros((5, m, d), dtype=np.float64)
for k in range(5):
    for i in range(m):
        for j in range(d):
            r = np.random.rand(1)
            V[i][j] = (r - 0.5) * 0.01
            r = np.random.rand(1)
            Mr_ik[k][i][j] = (r - 0.5) * 0.01
UuMpc = np.zeros((n, d), dtype=np.float64)

# train
pre_rmse = 10000
pre_mae = 10000
whole_time_start = time.time()
log_file = "results.txt"
file = open(log_file, 'a')
file.write("\n\n" + str(datetime.datetime.now()) + "copy1 > 50")
for t1 in range(T):
    Rmse = 0.0
    Mae = 0.0
    iter_time_start = time.time()
    for t2 in range(train_set_length):
        # Randomly pick up a rating from R
        index = np.random.randint(train_set_length)
        record = train_set[index]
        uid = record[0]
        iid = record[1]
        rui = record[2]

        res = get_UuMPC(uid,iid)
        UuMpc[uid] = res[0]
        Sqrt_num_ir_u = res[1]

        #rui = (Uu + UuMpc) * Vi + bu + bi + mu
        pred_rui = global_avg + bu[uid] + bi[iid] + float(np.dot(U[uid], V[iid])) + float(np.dot(UuMpc[uid], V[iid]))

        if pred_rui < 1:
            pred_rui = 1
        if pred_rui > 5:
            pred_rui = 5

        # calculate the gradient, update parameter
        eui = rui - pred_rui
        neg_eui = -eui

        Uu = U[uid][:]
        Vv = np.zeros((m, d), dtype=np.float64)
        for i in range(m):
            Vv[i] = V[i][:]

        U[uid] -= lr * (neg_eui * V[iid] + lamb * U[uid])
        V[iid] -= lr * (neg_eui * (Uu + UuMpc[uid]) + lamb * V[iid])

        global_avg -= lr * neg_eui
        bu[uid] -= lr * (neg_eui + lamb * bu[uid])
        bi[iid] -= lr * (neg_eui + lamb * bi[iid])

        for i in range(5):
            item_list = train_uid_rate_set[uid][i+1]
            Mr_ik[i][item_list] -= lr * ((neg_eui * Vv[item_list])/Sqrt_num_ir_u[i] + lamb * Mr_ik[i][item_list])

    iter_time_end = time.time()
    file.write('\nIteration [{}/{}],  spend:{}s'.format(t1 + 1, T, iter_time_end - iter_time_start))

    res = test()
    Rmse = res[0]
    Mae = res[1]
    file.write('\nRMSE = {:.4f}, MAE = {:.4f}'.format(Rmse,Mae))
    print('\Iteration[{}/{}], RMSE = {:.4f}, MAE = {:.4F}'.format(t1 + 1, T, Rmse, Mae))


    if pre_mae > Mae and pre_rmse > Rmse:
        pre_mae = Mae
        pre_rmse = Rmse
    # Decrease the learning rate γ←γ×0.9
    lr = lr * 0.9

whole_time_end = time.time()
file.write('\nMIN_RMSE:{:.4f}, MIN_MAE:{:.4f}, spend:{:.4f}s'.format(pre_rmse,pre_mae,whole_time_end - whole_time_start))
file.close()
