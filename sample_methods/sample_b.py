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


def olken_olken_sample_union_bernoulli(js, n, hs):
    time_store = []

    S = pd.DataFrame()
    N = len(js)

    # store join size
    J = np.zeros(N)
    for i in range(0, N):
        J[i] = e_size(js[i])
    
    U = calc_U(js)
    print("U calculated: ", U)

    # probability of choosing J_i
    P = np.zeros(N)

    weight_start = time.perf_counter()
    ws = []
    for i in range(N):
        P[i] = J[i] / U
    # print(P)
        ws.append(olkens_store_ws(js[i], hs[i]))
    weight_end = time.perf_counter()

    print("weights updated in ", weight_end - weight_start, " s")

    sample_start = time.perf_counter()

    first_seen = []
    keep = True
    first = True
    time_on_reject = 0

    while S.shape[0] < n:
        round_start = time.perf_counter()
        round_record = []
        for j in range(len(js)):
            result = pd.DataFrame()
            p_j = P[j]
            r_j = random.random()
            if r_j > p_j:
                reject_time = time.perf_counter()
                time_on_reject += reject_time - round_start
                continue
            else:
                check = True
                fail = False
                while (check):
                    ts = olken_sample_from_s_join(js[j], hs[j], ws[j], J[j])
                    check = False
                    if len(ts) != len(js[j].tables):
                        fail = True
                        reject_time = time.perf_counter()
                        time_on_reject += reject_time - round_start
                        break
                    result = ts[0]
                    for i in range(1,len(ts)):
                        result = pd.merge(result, ts[i], on = js[j].keys[i-1], how = 'inner')
                    for t in round_record:
                        if result.equals(t):
                            check = True 
                # print(result)
                if not fail:
                    for index, row in S.iterrows():  
                        if result.equals(row):
                            if first_seen[index] == j :
                                first = False
                                round_record.append(result)
                            else: 
                                keep = False
                                reject_time = time.perf_counter()
                                time_on_reject += reject_time - round_start
                            break
                
                    if (keep and first):
                        first_seen.append(j)
                        round_record.append(result)

                    # print("keep ", keep)
                    # print("first ", first)
                
        if (S.shape[0] + len(round_record) > n):
            # print("exceed")
            reject_time = time.perf_counter()
            time_on_reject += reject_time - round_start
            round_record.clear()
            
        for tuple in round_record:
            S = pd.concat([S, tuple])
            if(len(S) % 100 == 0):
                cur_time = time.perf_counter()
                time_store.append(cur_time - sample_start)

        # print(S.shape[0])
        round_record.clear()    
    
    return S, time_store


def exact_olken_sample_union_bernoulli(js, n, hs):
    time_store = []

    S = pd.DataFrame()
    N = len(js)

    # store join size
    J = np.zeros(N)
    for i in range(0, N):
        J[i] = e_size(js[i])
    
    # ------
    print(J)
    
    print("Calculating Os...")
    ans, Os = gen_os(js)
    print(Os)
    
    print("Calculating U...")
    As = calc_As(js, Os, ans)
    U = 0

    As_T = np.array(As).T.tolist()
    for k in range(len(As)):
        U += (1/(k+1) * np.sum(As_T[k]))
        
    print(U)
    
    # ------
    
    U = calc_U(js)

    # probability of choosing J_i
    P = np.zeros(N)

    weight_start = time.perf_counter()
    ws = []
    for i in range(N):
        P[i] = J[i] / U
    # print(P)
        ws.append(exact_store_ws(js[i], hs[i]))
    weight_end = time.perf_counter()

    print("weights updated in ", weight_end - weight_start, " s")

    # f = open("./tpch_1_test/exact_weights.pkl","wb")
    # pickle.dump(ws,f)
    # f.close()

    # print("successfully stored")
    
    # ws = pickle.load(open("./tpch_1_test/exact_weights.pkl", "rb"))
    # print("weights successfully loaded")

    sample_start = time.perf_counter()

    first_seen = []
    keep = True
    first = True
    time_on_reject = 0

    while S.shape[0] < n:
        round_start = time.perf_counter()
        round_record = []
        for j in range(len(js)):
            result = pd.DataFrame()
            p_j = P[j]
            r_j = random.random()
            if r_j > p_j:
                reject_time = time.perf_counter()
                time_on_reject += reject_time - round_start
                continue
            else:
                check = True
                while (check):
                    ts = exact_sample_from_s_join(js[j], hs[j], ws[j])
                    check = False
                    result = ts[0]
                    for i in range(1,len(js[j].tables)):
                        result = pd.merge(result, ts[i], on = js[j].keys[i-1], how = 'inner')
                    for t in round_record:
                        if result.equals(t):
                            check = True 
                    else:
                        continue
                # print(result)
                for index, row in S.iterrows():  
                    if result.equals(row):
                        if first_seen[index] == j :
                            first = False
                            round_record.append(result)
                        else: 
                            keep = False
                            reject_time = time.perf_counter()
                            time_on_reject += reject_time - round_start
                        break
                
                if (keep and first):
                    first_seen.append(j)
                    round_record.append(result)

                # print("keep ", keep)
                # print("first ", first)
                
        if (S.shape[0] + len(round_record) > n):
            # print("exceed")
            reject_time = time.perf_counter()
            time_on_reject += reject_time - round_start
            round_record.clear()
            
        for tuple in round_record:
            S = pd.concat([S, tuple])
            if(len(S) % 100 == 0):
                cur_time = time.perf_counter()
                time_store.append(cur_time - sample_start)

        # print(S.shape[0])
        round_record.clear()    
    
    return S, time_store


def online_sample_union_bernoulli(joins, n, hs):
    return 0


