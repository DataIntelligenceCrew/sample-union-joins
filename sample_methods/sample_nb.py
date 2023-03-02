import pickle
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, './iidjoin')
from warm_up.build_hash import *
from warm_up.acyclic_join import *
from warm_up.online_init import *
from warm_up.equi_chain_overlap import *
from sample_methods.olken_single import *
from sample_methods.exact_single import *


def define_order(js):
    
    order = np.arange(len(js))
    # order = random.shuffle(order)
    print ("order: ", order)
    
    return order

# TODO - make this work for general case
def calc_cover_size(J, ans, Os):
    J_prime = np.zeros(len(J))
    for J_index in range(len(J)):
        J_prime[J_index] = J[J_index]
        for ans_index in range(1, len(ans)):
            if J_index in ans[ans_index]:
                if len(ans[ans_index]) < J_index + 1:
                    if len(ans[ans_index]) % 2 == 0:
                        J_prime[J_index] -=  Os[ans_index]
                    else:
                        J_prime[J_index] +=  Os[ans_index]
    print("J_prime: ", J_prime)
    return J_prime


def olken_sample_union_nb(js, n, hs):
    
    process_start = time.perf_counter()
    
    time_store= []
    record = {}
    iterations = 0
    time_on_reject = 0
    
    S = pd.DataFrame()
    N = len(js)
    
    J = np.zeros(N)
    for i in range(0, N):
        J[i] = e_size(js[i])
    print("J calculated: ", J)
    
    ans, Os = gen_os(js)
    print("ans: ", ans)
    print("Os: ", Os)
    
    # J_prime = calc_cover_size(J, ans, Os)
    J_prime = np.zeros(len(J))
    for i in range(len(J)):
        if i == 0:
            J_prime[i] = J[i]
            
        if i == 1:
            J_prime[i] = J[i] - Os[0]
        
        if i == 2:
            J_prime[i] = J[i] - Os[1] - Os[2] + Os[3]
    print("sum J_prime: ", np.sum(J_prime))
    
    P = J_prime / np.sum(J_prime)
    print("P calculated: ", P)
    
    ws = []
    for i in range(N):
        ws.append(olkens_store_ws(js[i], hs[i]))
    print("weights successfully calculated")
    
    process_end = time.perf_counter()
    print("process time: ", process_end - process_start)
    
    
    sample_start = time.perf_counter()
    while S.shape[0] < n:
        iterations += 1
        round_start = time.perf_counter()
        j = np.random.choice(np.arange(0, N), p = P)
        ts = olken_sample_from_s_join(js[j], hs[j], ws[j], J[j])
        if len(ts) != len(js[j].tables):
            reject_time = time.perf_counter()
            time_on_reject += reject_time - round_start
            continue
        else:
            key = js[j].keys[len(js[j].keys) - 1]
            value = ts[len(js[j].tables)-1][key].values[0]
            # print(value)
            
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
                        # print("before: ")
                        # print(len(S))
                        S = S[S[key] != value]
                        # print("after: ")
                        # print(len(S))
                result = ts[0]
                for i in range(1,len(ts)):
                    result = pd.merge(result, ts[i], on = js[j].keys[i-1], how = 'inner')
                S = pd.concat([S, result])
                if(len(S) % 100 == 0):
                    cur_time = time.perf_counter()
                    time_store.append(cur_time - sample_start)
    
    print("completed in ", iterations, " iterations")
    print("total time on reject: ", time_on_reject)
    return S, time_store


def exact_sample_union_nb(js, n, hs):
    
    process_start = time.perf_counter()
    
    time_store= []
    record = {}
    iterations = 0
    time_on_reject = 0
    
    S = pd.DataFrame()
    N = len(js)
    
    J = np.zeros(N)
    for i in range(0, N):
        J[i] = e_size(js[i])
    print("J calculated: ", J)
    
    ans, Os = gen_os(js)
    print("ans: ", ans)
    print("Os: ", Os)
    
    # J_prime = calc_cover_size(J, ans, Os)
    J_prime = np.zeros(len(J))
    for i in range(len(J)):
        if i == 0:
            J_prime[i] = J[i]
            
        if i == 1:
            J_prime[i] = J[i] - Os[0]
        
        if i == 2:
            J_prime[i] = J[i] - Os[1] - Os[2] + Os[3]
    print("sum J_prime: ", np.sum(J_prime))
    
    # U = calc_U(js)
    # print("U calculated: ", U)
    # P = J_prime / U 
    
    P = J_prime / np.sum(J_prime)
    print("P calculated: ", P)
    
    ws = []
    for i in range(N):
        ws.append(exact_store_ws(js[i], hs[i]))
    print("weights successfully calculated")
    
    process_end = time.perf_counter()
    print("process time: ", process_end - process_start)
    
    sample_start = time.perf_counter()
    while S.shape[0] < n:
        iterations += 1
        round_start = time.perf_counter()
        j = np.random.choice(np.arange(0, N), p = P)
        ts = exact_sample_from_s_join(js[j], hs[j], ws[j])
        key = js[j].keys[len(js[j].keys) - 1]
        value = ts[len(js[j].tables)-1][key].values[0]
        # print(value)
        
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
                    # print("before: ")
                    # print(len(S))
                    S = S[S[key] != value]
                    # print("after: ")
                    # print(len(S))
            result = ts[0]
            for i in range(1,len(ts)):
                result = pd.merge(result, ts[i], on = js[j].keys[i-1], how = 'inner')
            S = pd.concat([S, result])
            if(len(S) % 100 == 0):
                cur_time = time.perf_counter()
                time_store.append(cur_time - sample_start)
    
    print("completed in ", iterations, " iterations")
    print("total time on reject: ", time_on_reject)
    return S, time_store

def online_sample_union_nb(js, n, hs, joins_pri, hs_pri, pri_keys):
    
    process_start = time.perf_counter()
    
    time_store= []
    record = {}
    iterations = 0
    time_on_reject = 0
    
    S = pd.DataFrame()
    N = len(js)
    
    J, ans, Os = online_init_nb(joins_pri, hs_pri, 0.95, pri_keys)
    print("J calculated: ", J)
    print("ans: ", ans)
    print("Os: ", Os)
  
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
    # P = [0.31065389, 0.214872, 0.47447411]
    
    ws = []
    for i in range(N):
        ws.append(exact_store_ws(js[i], hs[i]))
    print("weights successfully calculated")
    
    process_end = time.perf_counter()
    print("process time: ", process_end - process_start)
    
    sample_start = time.perf_counter()
    while S.shape[0] < n:
        iterations += 1
        round_start = time.perf_counter()
        j = np.random.choice(np.arange(0, N), p = P)
        ts = exact_sample_from_s_join(js[j], hs[j], ws[j])
        key = js[j].keys[len(js[j].keys) - 1]
        value = ts[len(js[j].tables)-1][key].values[0]
        # print(value)
        
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
                    # print("before: ")
                    # print(len(S))
                    S = S[S[key] != value]
                    # print("after: ")
                    # print(len(S))
            result = ts[0]
            for i in range(1,len(ts)):
                result = pd.merge(result, ts[i], on = js[j].keys[i-1], how = 'inner')
            S = pd.concat([S, result])
            if(len(S) % 100 == 0):
                cur_time = time.perf_counter()
                time_store.append(cur_time - sample_start)
    
    print("completed in ", iterations, " iterations")
    print("total time on reject: ", time_on_reject)
    return S, time_store
