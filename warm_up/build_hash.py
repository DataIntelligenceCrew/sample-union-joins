from collections import defaultdict
import sys
sys.path.insert(0, './iidjoin')
from warm_up.acyclic_join import *
import pickle

def hash_j(j):
    n = len(j.tables)
    hs = [defaultdict(list) for _ in range(n-1)]
    for i in range(1,n):
        print(i)
        for index, row in j.tables[i].iterrows():
            # print(index)
            # if i == 4:
            #     print(row)
            key = row[j.keys[i-1]]
            hs[i-1][key].append(index)
    return hs

def hash_j_pri(j, pri_keys):
    n = len(j.tables)
    hs = [defaultdict(list) for _ in range(n-1)]
    for i in range(1,n):
        print(i)
        for index, row in j.tables[i].iterrows():
            # print(index)
            key = row[j.keys[i-1]]
            pri = row[pri_keys[i]]
            hs[i-1][key].append(pri)
    return hs
