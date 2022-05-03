import os, h5py, numpy as np
import torch
from scipy.io import loadmat

from util_proposed import graphLaplacianMatrix, feature_ordering, feature_selection, fs_eval_sup
from train_tox_proposed0 import train
from model import *
        
        
def data_preproc(k, n_trl):
    
    n_trl_max = 15
    n_tru_max = 10
    n_tru = 10
    n_te = 14
    n_class = 4
    n_party = 2
    n_samp = 39
    
    # Data Loading
    # print('>>> Data Loading ...')
    if not os.path.exists('data/TOX_171/TOX_171_partition.h5'):
        info = loadmat('data/TOX_171/TOX_171.mat')
        X = info['X']
        Y = info['Y']
        Y = Y - 1
        # Data Normalizing
        X = X/np.max(np.max(np.abs(X)))
        # Class balancing
        n_class = len(np.unique(np.squeeze(Y)))
        Y_bincount = np.bincount(np.squeeze(Y))
        n_samp_min = np.min(Y_bincount)
        n_dim = X.shape[1]
        X_balance = np.zeros((n_samp_min*n_class, n_dim))
        Y_balance = np.zeros((n_samp_min*n_class, ))
        for i in range(n_class):
            ind_i = np.array([j for j in range(len(Y)) if Y[j] == i])
            index_permute = np.random.permutation(len(ind_i))
            ind_i_keep = ind_i[index_permute[:n_samp_min]]
            X_balance[i*n_samp_min:(i+1)*n_samp_min, :] = X[ind_i_keep, :]
            Y_balance[i*n_samp_min:(i+1)*n_samp_min, ] = np.squeeze(Y[ind_i_keep, ])
        # Data Partitioning
        ind_d = np.random.permutation(X.shape[1])
        n_dim_part1 = int(np.round(X.shape[1]/2))
        ind_d_part1 = ind_d[:n_dim_part1]
        ind_d_part2 = ind_d[n_dim_part1:]
        X_part1 = X_balance[:, ind_d_part1]
        X_part2 = X_balance[:, ind_d_part2]
        # Save partitioned data
        f = h5py.File('data/TOX_171/TOX_171_partition.h5', 'w')
        f.create_dataset('X_part1', data=X_part1)
        f.create_dataset('X_part2', data=X_part2)
        f.create_dataset('Y', data=Y_balance)
        f.close()
    else:
        f = h5py.File('data/TOX_171/TOX_171_partition.h5', 'r')
        X_part1 = f.get('X_part1')[()] 
        X_part2 = f.get('X_part2')[()] 
        Y = f.get('Y')[()] 
        f.close()  
        
    X = [X_part1, X_part2]

    # Training/Testing Splitting
    # print('>>> Training/Testing Splitting ...')
    data_trl = []
    data_tru = []
    data_te = []
    for i in range(n_party):
        Xi = X[i]
        data_trl_i = np.zeros((n_class*n_trl,Xi.shape[1]))
        data_tru_i = np.zeros((n_class*n_tru,Xi.shape[1]))
        data_te_i = np.zeros((n_class*n_te,Xi.shape[1]))
        for j in range(n_class):
            # print('i = '+str(i)+', j = '+str(j))
            data_trl_i[j*n_trl:(j+1)*n_trl,:] = Xi[j*n_samp:j*n_samp+n_trl,:]
            tru_maxrange = np.array(list(range(j*n_samp+n_trl_max, j*n_samp+n_trl_max+n_tru_max)))
            if not os.path.exists('data/TOX_171/ind_tru/'):
                os.makedirs('data/TOX_171/ind_tru/')
            if not os.path.exists('data/TOX_171/ind_tru/ind_tru_'+str(k)+'_'+str(i)+'_'+str(j)+'.h5'):
                ind_tru = np.random.permutation(n_tru_max)
                f = h5py.File('data/TOX_171/ind_tru/ind_tru_'+str(k)+'_'+str(i)+'_'+str(j)+'.h5', 'w')
                f.create_dataset('ind_tru', data=ind_tru)
                f.close()
            else:
                f = h5py.File('data/TOX_171/ind_tru/ind_tru_'+str(k)+'_'+str(i)+'_'+str(j)+'.h5', 'r')
                ind_tru = f.get('ind_tru')[()] 
                f.close()
            tru_range = tru_maxrange[ind_tru[:n_tru]]
            data_tru_i[j*n_tru:(j+1)*n_tru,:] = Xi[tru_range,:]
            # data_tru_i[j*n_tru:(j+1)*n_tru,:] = Xi[j*n_samp+n_trl_max:j*n_samp+n_trl_max+n_tru,:]
            data_te_i[j*n_te:(j+1)*n_te,:] = Xi[(j+1)*n_samp-n_te:(j+1)*n_samp,:]
        data_trl.append(data_trl_i)
        data_tru.append(data_tru_i)
        data_te.append(data_te_i)
    Y_unique = np.expand_dims(np.unique(Y), axis=1)
    labels_trl = np.repeat(Y_unique, n_trl, axis=1).reshape(1,n_trl*n_class).transpose()
    labels_tru = np.repeat(Y_unique, n_tru, axis=1).reshape(1,n_tru*n_class).transpose()
    labels_te = np.repeat(Y_unique, n_te, axis=1).reshape(1,n_te*n_class).transpose()
        
    # Data packing
    # print('>>> Data Packing ...')
    # Labels
    labels_trl = torch.LongTensor(labels_trl)
    labels_tru = torch.LongTensor(labels_tru)
    labels_te = torch.LongTensor(labels_te)
    # Data
    # part1
    data_trl_part1 = torch.FloatTensor(data_trl[0])
    data_tru_part1 = torch.FloatTensor(data_tru[0])
    data_te_part1 = torch.FloatTensor(data_te[0])
    # part2
    data_trl_part2 = torch.FloatTensor(data_trl[1])
    data_tru_part2 = torch.FloatTensor(data_tru[1])
    data_te_part2 = torch.FloatTensor(data_te[1])
    
    return labels_trl, labels_tru, labels_te, \
        data_trl_part1, data_tru_part1, data_te_part1, \
        data_trl_part2, data_tru_part2, data_te_part2

        
def main_train(k, n_trl, alpha, beta):
    
    np.random.seed(0)
    
    n_party = 2
    n_class = 4
    size_localout = 100
    n_dim_part1 = 2874
    n_dim_part2 = 2874
    
    # data preprocessing
    labels_trl, labels_tru, labels_te, \
    data_trl_part1, data_tru_part1, data_te_part1, \
    data_trl_part2, data_tru_part2, data_te_part2 = data_preproc(k, n_trl)
    
    # Model Training
    # print('>>> Graph Laplacian Matrix Construction ......')
    L_part1 = graphLaplacianMatrix(data_trl_part1, data_tru_part1)
    L_part2 = graphLaplacianMatrix(data_trl_part2, data_tru_part2)
    # print('>>> Local Model Training ......') 
    W1_part1, W1_part2, = \
    train('tox', 
          data_trl_part1, data_trl_part2, 
          labels_trl, 
          data_tru_part1, data_tru_part2, 
          L_part1, L_part2, 
          n_dim_part1, n_dim_part2, 
          size_localout, 
          n_party, n_class, alpha, beta)           
    
    return W1_part1, W1_part2


def data_preproc_fs(n_tr, n_te):
    
    n_party = 2
    n_class = 4
    n_samp = 39
    
    # Data Loading
    # print('>>> Data Loading ...')
    if not os.path.exists('data/TOX_171/TOX_171_partition.h5'):
        info = loadmat('data/TOX_171/TOX_171.mat')
        X = info['X']
        Y = info['Y']
        Y = Y - 1
        # Data Normalizing
        X = X/np.max(np.max(np.abs(X)))
        # Class balancing
        n_class = len(np.unique(np.squeeze(Y)))
        Y_bincount = np.bincount(np.squeeze(Y))
        n_samp_min = np.min(Y_bincount)
        n_dim = X.shape[1]
        X_balance = np.zeros((n_samp_min*n_class, n_dim))
        Y_balance = np.zeros((n_samp_min*n_class, ))
        for i in range(n_class):
            ind_i = np.array([j for j in range(len(Y)) if Y[j] == i])
            index_permute = np.random.permutation(len(ind_i))
            ind_i_keep = ind_i[index_permute[:n_samp_min]]
            X_balance[i*n_samp_min:(i+1)*n_samp_min, :] = X[ind_i_keep, :]
            Y_balance[i*n_samp_min:(i+1)*n_samp_min, ] = np.squeeze(Y[ind_i_keep, ])
        # Data Partitioning
        ind_d = np.random.permutation(X.shape[1])
        n_dim_part1 = int(np.round(X.shape[1]/2))
        ind_d_part1 = ind_d[:n_dim_part1]
        ind_d_part2 = ind_d[n_dim_part1:]
        X_part1 = X_balance[:, ind_d_part1]
        X_part2 = X_balance[:, ind_d_part2]
        # Save partitioned data
        f = h5py.File('data/TOX_171/TOX_171_partition.h5', 'w')
        f.create_dataset('X_part1', data=X_part1)
        f.create_dataset('X_part2', data=X_part2)
        f.create_dataset('Y', data=Y_balance)
        f.close()
    else:
        f = h5py.File('data/TOX_171/TOX_171_partition.h5', 'r')
        X_part1 = f.get('X_part1')[()] 
        X_part2 = f.get('X_part2')[()] 
        Y = f.get('Y')[()] 
        f.close()  
        
    X = [X_part1, X_part2]

    # Training/Testing Splitting
    # print('>>> Training/Testing Splitting ...')
    data_tr = []
    data_te = []
    for i in range(n_party):
        Xi = X[i]
        data_tr_i = np.zeros((n_class*n_tr,Xi.shape[1]))
        data_te_i = np.zeros((n_class*n_te,Xi.shape[1]))
        for j in range(n_class):
            # print('i = '+str(i)+', j = '+str(j))
            data_tr_i[j*n_tr:(j+1)*n_tr,:] = Xi[j*n_samp:j*n_samp+n_tr,:]
            data_te_i[j*n_te:(j+1)*n_te,:] = Xi[(j+1)*n_samp-n_te:(j+1)*n_samp,:]
        data_tr.append(data_tr_i)
        data_te.append(data_te_i)
    Y_unique = np.expand_dims(np.unique(Y), axis=1)
    labels_tr = np.repeat(Y_unique, n_tr, axis=1).reshape(1,n_tr*n_class).transpose()
    labels_te = np.repeat(Y_unique, n_te, axis=1).reshape(1,n_te*n_class).transpose()
        
    # Data packing
    # print('>>> Data Packing ...')
    # Labels
    labels_tr = torch.LongTensor(labels_tr)
    labels_te = torch.LongTensor(labels_te)
    # Data
    # part1
    data_tr_part1 = torch.FloatTensor(data_tr[0])
    data_te_part1 = torch.FloatTensor(data_te[0])
    # part2
    data_tr_part2 = torch.FloatTensor(data_tr[1])
    data_te_part2 = torch.FloatTensor(data_te[1])
    
    return labels_tr, labels_te, \
        data_tr_part1, data_te_part1, \
        data_tr_part2, data_te_part2


def main_fs_evaluation(n_tr, n_te, p_list, W1_part1, W1_part2):
    
    np.random.seed(0)
    
    n_class = 10
    n_dim_part1 = 2874
    n_dim_part2 = 2874
    
    # data preprocessing
    labels_tr, labels_te, \
    data_tr_part1, data_te_part1, \
    data_tr_part2, data_te_part2 = data_preproc_fs(n_tr, n_te)
    
    # feature importance weighting
    _, ind_part1 = feature_ordering(W1_part1)
    _, ind_part2 = feature_ordering(W1_part2)
    
    acc_part1_mat = []
    acc_part2_mat = []
    
    # feature selection
    for i in range(len(p_list)):
        
        p = p_list[i]
        
        # part1
        ind_keep_part1 = feature_selection(ind_part1, p, n_dim_part1)
        acc_part1 = fs_eval_sup(data_tr_part1, labels_tr, data_te_part1, labels_te, ind_keep_part1, len(ind_keep_part1), n_class)
        acc_part1_mat.append(acc_part1)
        # part2
        ind_keep_part2 = feature_selection(ind_part2, p, n_dim_part2)
        acc_part2 = fs_eval_sup(data_tr_part2, labels_tr, data_te_part2, labels_te, ind_keep_part2, len(ind_keep_part2), n_class)
        acc_part2_mat.append(acc_part2)
        
#    print('acc_part1_mat = ')
#    print(acc_part1_mat)
#    print('acc_part2_mat = ')
#    print(acc_part2_mat)
        
    return acc_part1_mat, acc_part2_mat