# exact
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

def exact_sample_from_s_join(j,hs,w):
    ts = []
    t_sets = []

    for i in range(0, len(j.tables)):
        if i == 0:
            sum_w = np.sum(w[0])
            P = np.zeros(j.tables[0].shape[0])
            for k in range(0, j.tables[0].shape[0]):
                P[k] = w[0][k] / sum_w

            t_i = np.random.choice(np.arange(0, j.tables[0].shape[0]), p = P)

            t = j.tables[0].loc[[t_i]]
            t_v = t[j.keys[0]]
            t_sets = hs[0][t_v.values[0]]

            ts.append(t)
        else:
            sum_w = 0
            for t_index in t_sets:
                sum_w += w[i][t_index]

            P = np.zeros(len(t_sets))
            for k in range(len(t_sets)):
                P[k] = w[i][t_sets[k]] / sum_w
            
            t_i_set = np.random.choice(np.arange(0, len(t_sets)), p = P)
            t_i = t_sets[t_i_set]

            t = j.tables[i].loc[[t_i]]
            if i != len(j.tables)-1:
                t_v = t[j.keys[i]]
                t_sets = hs[i][t_v.values[0]]
            ts.append(t)
    return ts


def exact_store_ws(j, hs):
    ws =  [[] for _ in range(len(j.tables))]
    for i in range(len(j.tables)-1,-1,-1):
        print(i)
        for index, row in j.tables[i].iterrows():    
            if i == len(j.tables) - 1:
                ws[i].append(1)
            else:
                ws[i].append(exact_calc_W(i, row[j.keys[i]], hs, ws[i+1]))
        # print(ws)
    return ws


def exact_calc_W(index, t_v, hs, w_prev):
    w = 0

    for t_index in hs[index][t_v]:
        w += w_prev[t_index]
        
    return w
