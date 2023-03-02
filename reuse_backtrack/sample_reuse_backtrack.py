import pickle
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, './iidjoin')
from warm_up.build_hash import *
from warm_up.acyclic_join import *
from warm_up.equi_chain_overlap import *
from sample_methods.olken_single import *
from sample_methods.exact_single import *
from reuse_backtrack.online_init_reuse import *

def random_choose_t(ts):
    t_index = np.random.randint(0, len(ts))
    t = ts[t_index]
    return t, t_index


def online_sample_union_nb_reuse(js, n, hs, joins_pri, hs_pri, pri_keys):
    
    process_start = time.perf_counter()
    
    time_store= []
    record = {}
    iterations = 0
    time_on_reject = 0
    time_on_reuse_accept = 0
    time_on_reuse_rejec = 0
    num_accept_reuse = 0
    
    S = pd.DataFrame()
    N = len(js)
    
    J, ans, Os, ts_list, ps_list = online_init_nb(joins_pri, hs_pri, 0.95, pri_keys)
    print("J calculated: ", J)
    print("ans: ", ans)
    print("Os: ", Os)
    total_num_reuse = sum([len(ts_list[i]) for i in range(len(ts_list))])
  
    # J_prime = calc_cover_size(J, ans, Os)
    J_prime = np.zeros(len(J))
    for i in range(len(J)):
        if i == 0:
            J_prime[i] = J[i]
            
        if i == 1:
            J_prime[i] = J[i] - Os[0]
        
        if i == 2:
            J_prime[i] = J[i] - Os[1] - Os[2] + Os[3]
    
    print("J_prime: ", J_prime)
    print("sum J_prime: ", np.sum(J_prime))
    
    P = J_prime / np.sum(J_prime)
    print("P calculated: ", P)
    
    ws = []
    for i in range(N):
        ws.append(exact_store_ws(js[i], hs[i]))
    print("weights successfully calculated")
    
    # ----------------------------------------------------------
    # f = open("./results_online_opt/uq1.pkl","wb")
    # pickle.dump(ws,f)
    # f.close()
    # print("successfully stored")
    
    # ws = pickle.load(open("./results_online_opt/uq1.pkl", "rb"))
    # print("weights successfully loaded")
    # ----------------------------------------------------------
    
    process_end = time.perf_counter()
    print("process time: ", process_end - process_start)
    
    sample_start = time.perf_counter()
    while S.shape[0] < n:
        iterations += 1
        round_start = time.perf_counter()
        j = np.random.choice(np.arange(0, N), p = P)
        # print(j)
        if len(ts_list[j]) != 0:
            # random choose t from ts_list[j] with probability 1/len(ts_list[j]), get its index as t_index, and remove it from ts_list[j]
            t, t_index = random_choose_t(ts_list[j])         
            del ts_list[j][t_index]   
            accept_rate = len(ts_list[j]) / (ps_list[j][t_index] * J_prime[j])
            del ps_list[j][t_index]
            # random choose a number from uniform distribution, if it is less than accept_rate, accept t, else reject t
            if np.random.uniform() < accept_rate:
                key = js[j].keys[len(js[j].keys) - 1]
                value = t[key].values[0]
                result = t
                
                accept_reuse_time = time.perf_counter()
                time_on_reuse_accept += accept_reuse_time - round_start
                num_accept_reuse += 1
            else:
                reject_reuse_time = time.perf_counter()
                time_on_reuse_rejec += reject_reuse_time - round_start
                continue
            
        if len(ts_list[j]) == 0:
            ts = exact_sample_from_s_join(js[j], hs[j], ws[j])
            key = js[j].keys[len(js[j].keys) - 1]
            value = ts[len(js[j].tables)-1][key].values[0]
        
        if value in record and j > record[value]:
            reject_time = time.perf_counter()
            time_on_reject += reject_time - round_start
            print('duplicate')
            continue       
               
        else:
            if value not in record:
                record[value] = j
            else:
                if j < record[value]:
                    record[value] = j 
                    S = S[S[key] != value]

            if len(ts_list[j]) == 0:
                result = ts[0]
                for i in range(1,len(ts)):
                    result = pd.merge(result, ts[i], on = js[j].keys[i-1], how = 'inner')
                    
            S = pd.concat([S, result])
            if(len(S) % 100 == 0):
                cur_time = time.perf_counter()
                time_store.append(cur_time - sample_start)
    
    print("completed in ", iterations, " iterations")
    print("total time on reject: ", time_on_reject)
    print("total_num_reuse: ", total_num_reuse)
    print("time_on_reuse_accept: ", time_on_reuse_accept)
    print("time_on_reuse_rejec: ", time_on_reuse_rejec)
    print("num_accept_reuse: ", num_accept_reuse)
    
    return S, time_store



# def online_sample_union_nb_back(js, n, hs, joins_pri, hs_pri, pri_keys):
    
#     process_start = time.perf_counter()
    
    
#     time_store= []
#     record = {}
#     iterations = 0
#     time_on_reject = 0
#     time_on_reuse_accept = 0
#     time_on_reuse_rejec = 0
#     num_accept_reuse = 0
    
#     S = pd.DataFrame()
#     N = len(js)
    
#     J, ans, Os, ts_list, ps_list = online_init_nb(joins_pri, hs_pri, 0.95, pri_keys)
#     print("J calculated: ", J)
#     print("ans: ", ans)
#     print("Os: ", Os)
#     total_num_reuse = sum([len(ts_list[i]) for i in range(len(ts_list))])
  
#     # J_prime = calc_cover_size(J, ans, Os)
#     J_prime = np.zeros(len(J))
#     for i in range(len(J)):
#         if i == 0:
#             J_prime[i] = J[i]
            
#         if i == 1:
#             J_prime[i] = J[i] - Os[0]
        
#         if i == 2:
#             J_prime[i] = J[i] - Os[1] - Os[2] + Os[3]
    
#     print("J_prime: ", J_prime)
#     print("sum J_prime: ", np.sum(J_prime))
    
#     P = J_prime / np.sum(J_prime)
#     print("P calculated: ", P)
    
#     ws = []
#     for i in range(N):
#         ws.append(exact_store_ws(js[i], hs[i]))
#     print("weights successfully calculated")
    
#     process_end = time.perf_counter()
#     print("process time: ", process_end - process_start)
    
#     sample_start = time.perf_counter()
#     while S.shape[0] < n:
#         iterations += 1
#         round_start = time.perf_counter()
#         j = np.random.choice(np.arange(0, N), p = P)
#         if len(ts_list[j]) != 0:
#             # random choose t from ts_list[j] with probability 1/len(ts_list[j]), get its index as t_index, and remove it from ts_list[j]
#             t, t_index = random_choose_t(ts_list[j])         
#             del ts[t_index]   
#             accept_rate = len(ts_list[j]) / (ps_list[j][t_index] * J_prime[j])
#             # random choose a number from uniform distribution, if it is less than accept_rate, accept t, else reject t
#             if np.random.uniform() < accept_rate:
#                 key = js[j].keys[len(js[j].keys) - 1]
#                 value = t[key].values[0]
#                 result = t
                
#                 accept_reuse_time = time.perf_counter()
#                 time_on_reuse_accept += accept_reuse_time - round_start
#                 num_accept_reuse += 1
#             else:
#                 reject_reuse_time = time.perf_counter()
#                 time_on_reuse_rejec += reject_reuse_time - round_start
#                 continue
            
#         if len(ts_list[j]) == 0:
#             ts = exact_sample_from_s_join(js[j], hs[j], ws[j])
#             key = js[j].keys[len(js[j].keys) - 1]
#             value = ts[len(js[j].tables)-1][key].values[0]
        
#         if value in record and j > record[value]:
#             reject_time = time.perf_counter()
#             time_on_reject += reject_time - round_start
#             print('duplicate')
#             continue       
               
#         else:
#             if value not in record:
#                 record[value] = j
#             else:
#                 if j < record[value]:
#                     record[value] = j 
#                     S = S[S[key] != value]

#             if len(ts_list[j]) == 0:
#                 result = ts[0]
#                 for i in range(1,len(ts)):
#                     result = pd.merge(result, ts[i], on = js[j].keys[i-1], how = 'inner')
                    
#             S = pd.concat([S, result])
#             if(len(S) % 100 == 0):
#                 cur_time = time.perf_counter()
#                 time_store.append(cur_time - sample_start)
    
#     print("completed in ", iterations, " iterations")
#     print("total time on reject: ", time_on_reject)
#     print("total_num_reuse: ", total_num_reuse)
#     print("time_on_reuse_accept: ", time_on_reuse_accept)
#     print("time_on_reuse_rejec: ", time_on_reuse_rejec)
#     print("num_accept_reuse: ", num_accept_reuse)
    
#     return S, time_store