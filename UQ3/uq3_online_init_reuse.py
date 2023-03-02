import random
import re
import scipy.stats as sps
import math
import pandas as pd
import numpy as np
import pickle
import sys
sys.path.insert(0, './iidjoin')
from warm_up.acyclic_join import *
from warm_up.build_hash import *
from warm_up.equi_chain_overlap import *

def random_walk(j, hs, pri_keys):
    """
    :param j: join j
    :param hs: hash table for join keys
    :return: joined tuple and probability
    """ 
    ts = []
    t = j.tables[0].sample(n=1)
    ts.append(t)
    p = 1 / j.tables[0].shape[0]
    for table_index in range(1, len(j.tables)):
        bucket = hs[table_index-1][ts[table_index-1][j.keys[table_index-1]].values[0]]
        if len(bucket) == 0:
            return t, 0
        else:
            temp_t_index = random.sample(bucket, 1)
            temp_t = j.tables[table_index].loc[j.tables[table_index][pri_keys[table_index]] == temp_t_index[0]]
            ts.append(temp_t)
            t = pd.merge(t, temp_t, on = j.keys[table_index-1], how = 'inner')
            p *= 1 / len(bucket)
    return t, p


def calc_eps(sigma, z_alpha, recip_ps):
    """
    :param alpha: prompt by the user
    :param ps: list of probabilities for sampled tuples
    :return: confidence interval epsilon
    """ 
    m = len(recip_ps)
    epsilon = z_alpha * sigma / math.sqrt(m)
    return epsilon


def calc_sig(recip_ps, e_size):
    diff_sq = [(i - e_size)**2 for i in recip_ps]
    sig_sq = np.sum(diff_sq) / (len(recip_ps)-1)
    return math.sqrt(sig_sq)


def wander_e_size(alpha, join, hs, pri_keys):
    ts = []
    ps = []
    recip_ps = []
    recip_ps=np.asarray(recip_ps)
    z_alpha  = sps.norm.ppf((alpha + 1) / 2, loc=0, scale=1)
    print("alpha: ", z_alpha)
    max_accu_size = 0
    min_eps = math.inf
    
    while(True):
        t,p = random_walk(join, hs, pri_keys)
        ps.append(p)
        
        if np.sum(ps) > 0:
            if p == 0:
                recip_ps = np.append(recip_ps, [0])
                continue
            else:
                ts.append(t)
                ps.append(p)
                
                recip_ps = np.append(recip_ps, [1/p])
            
                e_size = np.mean(recip_ps)
                print(recip_ps.shape)
                # print("Estimated size: ", e_size)

                sigma = calc_sig(recip_ps, e_size)

                eps = calc_eps(sigma, z_alpha, recip_ps) / e_size
                # print("eps%: ", eps) 

                if eps < min_eps and eps > 0:
                    max_accu_size = e_size
                    min_eps = eps

                if min_eps < 0.5:
                    break      

            
    print("Estimated size: ", max_accu_size)
    return max_accu_size, ts, ps

def wander_gen_o(js, hs, alpha, size, pri_keys):
    ts = []
    ps = []
    
    count = []
    recip_ps = []
    inter_count = []
    olp = 0
    olps = []
    ratios = []
    
    min_eps = math.inf
    max_accu_olp = 0
    
    it = 0

    while(True):
        it += 1
        t, p = random_walk(js[0], hs[0], pri_keys[0])
        ts.append(t)
        ps.append(p)
        if p == 0:
            recip_ps.append(0)
            count.append(0)
        else:
            ts.append(t)
            ps.append(p)
            recip_ps.append(1/p)
            count_t = round(1/p)
            count.append(count_t)
            find = True
            for j_index in range(1,len(js)):
                for h_index in range(len(hs[j_index])):
                    t_value = t[js[j_index].keys[h_index]].values[0]
                    # print(t_value)
                    if t_value in hs[j_index][h_index]:
                        t_next_value = t[pri_keys[j_index][h_index+1]].values[0]
                        temp_bucket = hs[j_index][h_index][t_value]
                        #  print(t_next_value, temp_bucket)
                        if t_next_value in temp_bucket:
                            continue
                        else: 
                            find = False
                            break
                    else: 
                        find = False
                        break
                if find is False: 
                    break
            if find:
                inter_count.append(count_t)
        binom_ratio = np.sum(inter_count) / np.sum(count) 
        # olp = np.mean(recip_ps) * binom_ratio
        olp = np.mean(recip_ps) * binom_ratio
        eps = calc_conf(recip_ps, binom_ratio, alpha) / olp
        
        olps.append(olp)
        ratios.append(binom_ratio)
        
        print("Estimated olp: ", olp)
        print("eps%: ", eps) 

        if eps < min_eps and eps > 0:
            max_accu_olp = olp
            min_eps = eps

        if eps < 0.5 and it > 200:
            break

    print("Estimated olp: ", max_accu_olp)
    return max_accu_olp, min_eps, ts, ps


def gen_os(js, hs, alpha, e_j, pri_keys, ts_list, ps_list):
    #find intersection of values in first join attribute in all tables
    ans = p_set(list(range(len(js))))
    print(ans)
    Os = []
    Ps = []
    for subset in ans:
        new_js = []
        new_hs = []
        new_pri_keys = []
        for index in subset:
            new_js.append(js[index])
            new_hs.append(hs[index])
            new_pri_keys.append(pri_keys[index])
        max_accu_olp, max_pr, ts, ps = wander_gen_o(new_js, new_hs, alpha, e_j[index], new_pri_keys)
        
         # concat js and ts_list[subset[0]]
        ts_list = ts_list + ts
        ps_list = ps_list + ps
        
        Os.append(max_accu_olp)
        Ps.append(max_pr)
    return ans, Os, Ps, ts_list, ps_list


def calc_conf(recip_ps, binom_ratio, alpha):
    z_alpha  = sps.norm.ppf((alpha + 1) / 2, loc=0, scale=1)
    var = binom_ratio * (1 - binom_ratio)
    sig = math.sqrt(var)
    eps = z_alpha * sig / math.sqrt(len(recip_ps))
    return eps
    
    
def calc_As(Os, ans, e_j):
    n = len(e_j)
    As = [ [0]*n for i in range(n)]
    for j in range(len(e_j)):
        As[j][n-1] = Os[len(Os)-1]
        for k in range(n-1, 0, -1):
            A = 0 
            count = 0
            for index in range(len(ans)):
                if (len(ans[index]) == k) and (j in ans[index]):
                    A += Os[index]
                    count += 1
            if (k == 1): A += e_j[j]
            # print(1, "j: ", j, " k: ", k, " A: ", A)
            for r in range(k+1, n+1):
                # print(math.comb(r-1, k-1) * As[j][r-1])
                A -= (math.comb(r-1, k-1) * As[j][r-1])
            # print(2, "j: ", j, " k: ", k, " A: ", A)
            As[j][k-1] = A
    print("As: ", As)
    return As


def calc_U(js, hs, alpha, e_j):
    ans, Os, ps = gen_os(js, hs, alpha, e_j)
    print(Os)
    print(ps)
    As = calc_As(Os, ans, e_j)
    U = 0
    As_T = np.array(As).T.tolist()
    for k in range(len(As)):
        U += (1/(k+1) * np.sum(As_T[k]))
    return U


# combine wander_e_size and gen_os, estimate size and overlap together
def uq3_online_init_nb_reuse(js, hs, alpha, pri_keys):
    ts_list = []
    ps_list = []
    Js = []
    for join_index in range(len(js)):
        J, ts, ps = wander_e_size(alpha, js[join_index], hs[join_index], pri_keys[join_index])
        Js.append(J)
        ts_list.append(ts)
        ps_list.append(ps)
    ans, Os, ps, ts_list, ps_list = gen_os(js, hs, alpha, Js, pri_keys, ts_list, ps_list)
    return Js, ans, Os, ts_list, ps_list
