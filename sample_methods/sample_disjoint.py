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


def olken_sample_from_disjoint(js, n, hs):
    time_store = []

    S = pd.DataFrame()
    N = len(js)

    # store join size
    J = np.zeros(N)
    for i in range(0, N):
        J[i] = e_size(js[i])
    
    C = np.sum(J)

    # probability of choosing J_i
    P = np.zeros(N)

    ws = []
    for i in range(N):
        P[i] = J[i] / C
        ws.append(olkens_store_ws(js[i], hs[i]))
    
    print("weights successfully updated")

    sample_start = time.perf_counter()
    
    it = 0
    its = []
    while S.shape[0] < n:
        it+=1
        j = np.random.choice(np.arange(0, N), p = P)
        ts = olken_sample_from_s_join(js[j], hs[j], ws[j], J[j])
        if len(ts) == len(js[j].tables):
            result = ts[0]
            for i in range(1,len(js[j].tables)):
                result = pd.merge(result, ts[i], on = js[j].keys[i-1], how = 'inner')
            S = pd.concat([S, result])
            its.append(it)
            if(len(S) % 100 == 0):
                cur_time = time.perf_counter()
                time_store.append(cur_time - sample_start)
            # print(S.shape[0])
        else:
            continue
        
    print("iterations: ", it)
    
    return S, time_store


def exact_sample_from_disjoint(js, n, hs):
    time_store = []

    S = pd.DataFrame()
    N = len(js)

    # store join size
    J = np.zeros(N)

    for i in range(0, N):
        J[i] = e_size(js[i])
    
    C = np.sum(J)

    # probability of choosing J_i
    P = np.zeros(N)
    ws = []
    for i in range(N):
        P[i] = J[i] / C
        ws.append(exact_store_ws(js[i], hs[i]))
        
    print("weights successfully updated")
    
    sample_start = time.perf_counter()

    while S.shape[0] < n:
        j = np.random.choice(np.arange(0, N), p = P)
        # print("join path: ", j)
        # print(js[i].tables[2])
        ts = exact_sample_from_s_join(js[j], hs[j], ws[j])
        # print(ts)
        result = ts[0]
        for i in range(1,len(ts)):
            result = pd.merge(result, ts[i], on = js[j].keys[i-1], how = 'inner')
        # print(result)
        S = pd.concat([S, result])
        if(len(S) % 100 == 0):
            cur_time = time.perf_counter()
            time_store.append(cur_time - sample_start)
        # print(S.shape[0])
    
    return S, time_store








