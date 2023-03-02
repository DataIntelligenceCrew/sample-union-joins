import pickle
import random
import time
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, './iidjoin')
from warm_up.build_hash import *
from warm_up.acyclic_join import *
from warm_up.equi_chain_overlap import *


def olken_sample_from_s_join(j,hs,w,j_size):
    ts = []
    t_sets = []

    for i in range(0, len(j.tables)):
        if i == 0:
            # r = j.tables[i].sample(n=1)git
            sum_w = np.sum(w[0])

            P = np.zeros(j.tables[0].shape[0])
            for k in range(0, j.tables[0].shape[0]):
                P[k] = w[0][k] / sum_w

            t_i = np.random.choice(np.arange(0, j.tables[0].shape[0]), p = P)
            t = j.tables[0].loc[[t_i]]
            t_v = t[j.keys[0]]
            t_sets = hs[0][t_v.values[0]]
            # print(w)

            w_t = w[0][t_i]

            p_t = sum_w / j_size
            r_t = random.random()
            if r_t > p_t:
                # print("break")
                break

            ts.append(t)
        else:

            sum_w = 0
            for t_index in t_sets:
                # print("t_index: ", t_index)
                sum_w += w[i][t_index]

            if sum_w == 0:
                break

            P = np.zeros(len(t_sets))
            for k in range(len(t_sets)):
                P[k] = w[i][t_sets[k]] / sum_w

            t_i_set = np.random.choice(np.arange(0, len(t_sets)), p = P)
            t_i = t_sets[t_i_set]

            t = j.tables[i].loc[[t_i]]

            if i != len(j.tables)-1:
                t_v = t[j.keys[i]]
                t_sets = hs[i][t_v.values[0]]

            p_t = sum_w / w_t
            r_t = random.random()

            if r_t > p_t:
                # print("break")
                break
            
            ts.append(t)
            w_t = w[i][t_i]

    return ts


# olken's
def olkens_store_ws(j, hs):
    ws =  []
    for i in range(len(j.tables)):
        print(i)
        ws.append(olkens_calc_W(j, i, hs))
    return ws


def olkens_calc_W(j, t_index, hs):
    ws = []

    if t_index == len(j.tables) - 1:
        return [1 for value in range(j.tables[len(j.tables) - 1].shape[0])]

    else:
        w = 1
        for i in range(t_index+1,len(j.tables)):
            # olken's
            w *= max_d(j.tables[i], j.keys[i-1])

        for index, row in j.tables[t_index].iterrows():  
            t_v = row[j.keys[t_index]]  
            if len(hs[t_index][t_v]) == 0:
                ws.append(0)
            else:
                ws.append(w)
        
        return ws

