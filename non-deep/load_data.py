import numpy as np

def read_data():
    n = 943
    m = 1682
    train_set_matrix = np.zeros((n, m), dtype=np.float)
    yui = np.zeros((n, m), dtype=np.float)
    train_set = []
    train_uid_rate_set = {}
    test_set = []

    try:
        #load train set
        file_train = open('2.train', 'r')
        for line in file_train.readlines():
            line = line.strip()
            ss = line.split()
            uid = int(ss[0])
            iid = int(ss[1])
            r = int(ss[2])
            train_set_matrix[uid - 1][iid - 1] = r
            yui[uid - 1][iid - 1] = 1
            record = [uid - 1, iid - 1, r]
            train_set.append(record)
            rating_dic = {1: [], 2: [], 3: [], 4: [], 5: []}
            train_uid_rate_set.setdefault(uid-1, rating_dic)
            train_uid_rate_set[uid-1][r].append(iid-1)

        #load test set
        file_test = open('MF-MPC-dataset-ML100K/copy1.test', 'r')
        for line in file_test.readlines():
            line = line.strip()
            ss = line.split()
            uid = int(ss[0])
            iid = int(ss[1])
            r = int(ss[2])
            record = [uid-1, iid-1, r]
            test_set.append(record)

    finally:
        if file_train:
            file_train.close()
        if file_test:
            file_test.close()

    return train_set_matrix, yui, train_set, train_uid_rate_set, test_set
