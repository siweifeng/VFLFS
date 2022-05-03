import h5py, os, math, time, numpy as np

from main_tox_proposed0 import main_train, main_fs_evaluation

def entire_tox_proposed0(n_tru):

    time_start = time.time()
    
    n_party = 2
    
    iter_num = 5
    n_trl_list = [5, 15]
    alpha_list = [1e-5, 1e-4, 1e-3]
    beta_list = [1, 1e-1, 1e-3]
    p_list = (np.arange(10, 101, 10)/100).tolist()
    
    acc_part1_whole = np.zeros((iter_num, len(n_trl_list), len(alpha_list), len(beta_list), len(p_list)))
    acc_part2_whole = np.zeros((iter_num, len(n_trl_list), len(alpha_list), len(beta_list), len(p_list)))
    
    for k in range(iter_num):   
        
        for i in range(len(n_trl_list)):  
            
            # number of labeled training samples per class
            n_trl = n_trl_list[i]  
            
            for j in range(len(alpha_list)):  
                
                # value of alpha
                alpha = alpha_list[j]  
                
                for u in range(len(beta_list)):
                    
                    # value of beta
                    beta = beta_list[u]
                    
                    print('k = '+str(k)+', n_trl = '+str(n_trl)+', alpha = '+str(alpha)+', beta = '+str(beta))
                    
                    path = 'rslt/tox/n_tru='+str(n_tru)+'/proposed0/k='+str(k)+'/n_trl='+str(n_trl)+\
                    '/log10(alpha)='+str(int(math.log10(alpha)))+'/log10(beta)='+str(int(math.log10(beta)))+'/'
                    if not os.path.exists(path):
                        os.makedirs(path)
                    
                    # obtain W1
                    if not os.path.exists(path+'/W1.h5'):
                        # run main code
                        W1_part1, W1_part2 = main_train(k, n_trl, alpha, beta)
                        # save W1
                        f = h5py.File(path+'/W1.h5', 'w')
                        f.create_dataset('W1_part1', data=W1_part1)
                        f.create_dataset('W1_part2', data=W1_part2)
                        f.close()
                    else:
                        f = h5py.File(path+'/W1.h5', 'r')
                        W1_part1 = f.get('W1_part1')[()] 
                        W1_part2 = f.get('W1_part2')[()] 
                        f.close()
                        
                    # feature selection evaluation
                    acc_part1_mat, acc_part2_mat \
                    = main_fs_evaluation(7, 7, p_list, W1_part1, W1_part2)
                    
                    # pack results
                    acc_part1_whole[k,i,j,u,:] = acc_part1_mat
                    acc_part2_whole[k,i,j,u,:] = acc_part2_mat
                    
    # average
    acc_part1_avg = np.mean(acc_part1_whole, axis=0)
    acc_part2_avg = np.mean(acc_part2_whole, axis=0)
    
    # standard deviation
    acc_part1_std = np.std(acc_part1_whole, axis=0)
    acc_part2_std = np.std(acc_part2_whole, axis=0)
    
    # save results
    # average
    f = h5py.File('rslt/tox/n_tru='+str(n_tru)+'/proposed0/acc_avg.h5', 'w')
    f.create_dataset('part1', data=acc_part1_avg)
    f.create_dataset('part2', data=acc_part2_avg)
    f.close()
    # standard deviation
    f = h5py.File('rslt/tox/n_tru='+str(n_tru)+'/proposed0/acc_std.h5', 'w')
    f.create_dataset('part1', data=acc_part1_std)
    f.create_dataset('part2', data=acc_part2_std)
    f.close()            
                
    time_end = time.time()
    
    print('time cost = '+str((time_end-time_start)/3600)+'h')