import pandas as pd
import numpy as np
import math
import sys
sys.path.insert(0, './iidjoin')
from warm_up.build_hash import *
from warm_up.acyclic_join import *


def e_size(j):
    uniq_1 = j.tables[0][j.keys[0]].unique()
    uniq_2 = j.tables[1][j.keys[0]].unique()
    uniq = uniq_1[np.in1d(uniq_1,uniq_2)]
    key_1 = j.keys[0]
    # size = t_size(self.tables[0])
    size = 0
    for v in uniq:
        size += (j.tables[0][j.tables[0][key_1] == v].shape[0] *
            j.tables[1][j.tables[1][key_1] == v].shape[0])
    for i in range(1,len(j.keys)):
        size *= max_d(j.tables[i+1], j.keys[i])
        # print(i, ": ", max_d(j.tables[i], j.keys[i-1]))
    return size


# return max degree
def max_d(table, attribute):
    return table[attribute].value_counts().max()


# return table size
def t_size(table):
    return table.shape[0]


def uq3_gen_os(js):
    #find intersection of values in first join attribute in all tables

    ans = p_set(list(range(len(js))))
    
    Os = []
    for subset in ans:
        Os.append(gen_o(js,subset))
    
    return ans, Os

def get_int_values(js, index, subset):
    uniq = js[subset[0]].tables[index][js[subset[0]].keys[index]].unique()
    for i in subset:
        uniq_1 = js[i].tables[index][js[i].keys[index]].unique()
        uniq_2 = js[i].tables[index+1][js[i].keys[index]].unique()
        uniq_1_2 = uniq_1[np.in1d(uniq_1,uniq_2)]
        uniq = uniq[np.in1d(uniq,uniq_1_2)]
    return uniq

# generate powerset
def p_set(nums):
    ans_all = [[]]

    for n in nums:
        ans_all += [a+[n] for a in ans_all]
        
    ans = []
    for i in ans_all:
        if len(i) > 1: 
            ans.append(i)
        
    return ans

def gen_o(js,subset):
    key_1 = js[0].keys[0]
    K = 0
    uniq = get_int_values(js, 0, subset)

    sizes = []
    for v in uniq:
        for i in subset:
            sizes.append(js[i].tables[0][js[i].tables[0][key_1] == v].shape[0] *
            js[i].tables[1][js[i].tables[1][key_1] == v].shape[0])
        K += min(sizes)
    # print(K)

    for k in range(1,len(js[0].keys)):
        max_ds = []
        uniq = get_int_values(js, k, subset)
        for i in subset:
            if (js[i].join_type[k] == False):
                max_ds.append(1)
            else:
                max_ds.append(max_d_in_set(js[i].tables[k+1], js[i].keys[k], uniq))
            # print(max_ds)
            # max_ds.append(max_d(js[i].tables[k+1], js[i].keys[k]))
            # print(min(max_ds))
        # print(min(max_ds))
        K *= min(max_ds)
    
    return K

def max_d_in_set(table, attribute, value_set):
    values = table[attribute].value_counts(dropna=False).keys().tolist()
    counts = table[attribute].value_counts(dropna=False).tolist()
    value_dict = dict(zip(values, counts))
    # print(value_dict)
    for v in values:
        if v in value_set:
            # print(v, "has count: ", value_dict[v])
            return value_dict[v]

def exact_olp(f_js):
    ans = p_set(list(range(len(f_js))))
    # print(ans)
    
    Os = []
    for subset in ans:
        frames = []
        for i in range(len(subset)):
            frames.append(f_js[subset[i]])
        disjoint = pd.concat(frames)
        inter = disjoint.value_counts()[disjoint.value_counts() == len(subset)]
        Os.append(inter.shape[0])
    
    return Os

def calc_As(js, Os, ans):
    n = len(js)
    As = [ [0]*n for i in range(n)]
    for j in range(len(js)):
        As[j][n-1] = Os[len(Os)-1]
        for k in range(n-1, 0, -1):
            # print("k: ", k)
            A = 0 
            count = 0
            for index in range(len(ans)):
                if (len(ans[index]) == k) and (j in ans[index]):
                    A += Os[index]
                    count += 1
            if (k == 1): A += e_size(js[j])
            # if (k == 1): A += exact_j[j]
            # Calculate A
            for r in range(k+1, n+1):
                # print(math.comb(r-1, k-1))
                # print("As[r-1]", As[j][r-1])
                A -= (math.comb(r-1, k-1) * As[j][r-1])
            As[j][k-1] = A
    return As

def uq3_calc_U(js, norm_js):

    ans, Os = uq3_gen_os(norm_js)
    As = calc_As(js, Os, ans)
    U = 0
    # for j in range(len(As)):
    #     for k in range(len(As[j])):
    #         U += (1/(k+1) * As[j][k])

    As_T = np.array(As).T.tolist()
    # print(As_T)
    for k in range(len(As)):
        # print(np.sum(As[k]))
        U += (1/(k+1) * np.sum(As_T[k]))
    return U