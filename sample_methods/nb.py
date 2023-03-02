import json
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
            for r in range(k+1, n+1):
                A -= (math.comb(r-1, k-1) * As[j][r-1])
            As[j][k-1] = A
    print("As: ", As)
    return As


def calc_U(Js, Os, ans):
    As = calc_As(Os, ans, Js)
    U = 0
    As_T = np.array(As).T.tolist()
    for k in range(len(As)):
        U += (1/(k+1) * np.sum(As_T[k]))
    print("U: ", U)
    return U

def define_order(js):
    
    order = np.arange(len(js))
    # order = random.shuffle(order)
    print ("order: ", order)
    
    return order

def general_nb_sample(js, n, hss, order):
    
    time_store= []
    record = {}
    
    S = pd.DataFrame()
    N = len(js)
    
    # J = [1317536.2318840579, 1374045.8015267176, 1484042.5531914893, 1357442.5727411944, 1454410.8761329306]
    
    # Os = [722238.8059701492, 799948.5881515462, 708498.94786065, 460227.2727272727, 
    #       486630.3562657112, 638653.6612360181, 515810.42168910004, 554334.1183504878, 
    #       498991.56076562696, 532801.5095506823, 455211.3455140729, 791950.1802943184, 
    #       632165.578922356, 364949.99999999994, 684428.6475919674, 486725.9236543193, 
    #       512592.524121211, 324785.19075250556, 671969.82333643, 337493.68025155686, 
    #       617134.4439576083, 356354.0474640814, 378321.26987865585, 423965.5233694687, 
    #       463566.00432635704, 323393.5478568787]
    
    # real values
    
    Os = [421765, 423786, 420958, 194372]
    
    # Os = [421765, 423786, 420958, 194372, 
    #       421474, 422898, 193809, 420935, 
    #       194023, 192810, 121072, 421317, 
    #       420999, 193614, 421197, 194270, 
    #       192597, 121501, 421781, 193641, 
    #       194236, 121549, 193275, 121476, 
    #       120800, 91572]
    
    # U = 4009890
    # J = [1358885, 1362897, 1361991, 1361999, 1359413]
    J = [1358885, 1362897, 1361991]
    
    P = [0] * 3
    # 5 paths; no code for general case
    # no random 
    # for index in order:
    
    ans = [[0, 1], [0, 2], [1, 2], [0, 1, 2]]
    
    # ans = [[0, 1], [0, 2], [1, 2], [0, 1, 2], 
    # [0, 3], [1, 3], [0, 1, 3], [2, 3], 
    # [0, 2, 3], [1, 2, 3], [0, 1, 2, 3], [0, 4], 
    # [1, 4], [0, 1, 4], [2, 4], [0, 2, 4], 
    # [1, 2, 4], [0, 1, 2, 4], [3, 4], [0, 3, 4], 
    # [1, 3, 4], [0, 1, 3, 4], [2, 3, 4], [0, 2, 3, 4], 
    # [1, 2, 3, 4], [0, 1, 2, 3, 4]]
    
    # calc_U(J, Os, ans)
    As = calc_As(Os, ans, J)
    
    for i in range(len(order)):
        index = order[i]
        if i == 0:
            P[index] = J[index]
            
        if i == 1:
            P[index] = J[index] - Os[0]
        
        if i == 2:
            P[index] = J[index] - Os[1] - Os[2] + Os[3]
            
        # if i == 3:
        #     # P[index] = J[index] - Os[4] - Os[5] + Os[6] - Os[7]  + Os[8] + Os[9] - 2 * Os[10]
        #     P[index] = As[3][0] + Os[18] - Os[22] - Os[20] - Os[19] + Os[24] + Os[23] + Os[21] - 2 * Os[25]
        
        # if i == 4:
        #     # P[index] = J[index] - Os[11] - Os[12] + Os[13] - Os[14] + Os[15] + Os[16] - 2 * Os[17]
        #     # - Os[18] + Os [19] + Os [20] - 2 * Os[21] + Os[22] - 2 * Os[23] - 2 * Os[24] + 6 * Os[25] 
        #     P[index] = As[4][0]
    
    print(P)
    print("sum P: ", np.sum(P))
    P = P / np.sum(P)
    print(P)
    
    # ws = []
    # weight_start = time.perf_counter()
    # for i in range(N):
    #     ws.append(exact_store_ws(js[i], hss[i]))
    # weight_end = time.perf_counter()
    # print("weights updated in ", weight_end - weight_start, " s")
    
    # f = open("./tpch_1_test/exact_weights.pkl","wb")
    # pickle.dump(ws,f)
    # f.close()
    
    ws = pickle.load(open("./tpch_1_test/exact_weights.pkl", "rb"))
    print("weights successfully loaded")
    
    sample_start = time.perf_counter()
    while S.shape[0] < n:
        j = np.random.choice(np.arange(0, N), p = P)
        ts = exact_sample_from_s_join(js[j], hss[j], ws[j])
        value = ts[2]["OrderKey"].values[0]
        # print(value)
        
        if value in record and j > record[value]:
            # print('duplicate')
            continue       
               
        else:
            if value not in record:
                record[value] = j
            else:
                if j < record[value]:
                    
                    record[value] = j 
                    # print("before: ")
                    # print(len(S))
                    S = S[S["OrderKey"] != value]
                    # print("after: ")
                    # print(len(S))
    
            result = ts[0]
            for i in range(1,len(ts)):
                result = pd.merge(result, ts[i], on = js[j].keys[i-1], how = 'inner')
            S = pd.concat([S, result])
            if(len(S) % 100 == 0):
                cur_time = time.perf_counter()
                time_store.append(cur_time - sample_start)
    
    return S, time_store
        


def main():

    n = 10000

    customer_sample_1 = pd.read_csv('./tpch_1_test/c1.csv' ,index_col=0)
    orders_sample_1 = pd.read_csv('./tpch_1_test/o1.csv' ,index_col=0)
    lineitem_sample_1 = pd.read_csv('./tpch_1_test/l1.csv' ,index_col=0)

    customer_sample_2 = pd.read_csv('./tpch_1_test/c2.csv' ,index_col=0)
    orders_sample_2 = pd.read_csv('./tpch_1_test/o2.csv' ,index_col=0)
    lineitem_sample_2 = pd.read_csv('./tpch_1_test/l2.csv' ,index_col=0)

    customer_sample_3 = pd.read_csv('./tpch_1_test/c3.csv' ,index_col=0)
    orders_sample_3 = pd.read_csv('./tpch_1_test/o3.csv' ,index_col=0)
    lineitem_sample_3 = pd.read_csv('./tpch_1_test/l3.csv' ,index_col=0)

    # customer_sample_4 = pd.read_csv('./tpch_1_test/c4.csv' ,index_col=0)
    # orders_sample_4 = pd.read_csv('./tpch_1_test/o4.csv' ,index_col=0)
    # lineitem_sample_4 = pd.read_csv('./tpch_1_test/l4.csv' ,index_col=0)

    # customer_sample_5 = pd.read_csv('./tpch_1_test/c5.csv' ,index_col=0)
    # orders_sample_5 = pd.read_csv('./tpch_1_test/o5.csv' ,index_col=0)
    # lineitem_sample_5 = pd.read_csv('./tpch_1_test/l5.csv' ,index_col=0)

    tables_1 = [customer_sample_1, orders_sample_1,lineitem_sample_1]
    tables_2 = [customer_sample_2, orders_sample_2,lineitem_sample_2]
    tables_3 = [customer_sample_3, orders_sample_3,lineitem_sample_3]
    # tables_4 = [customer_sample_4, orders_sample_4,lineitem_sample_4]
    # tables_5 = [customer_sample_5, orders_sample_5,lineitem_sample_5]
    keys = ['CustKey', 'OrderKey']

    print("step 1 over")

    join_1 = chain_join(tables_1, keys)
    join_2 = chain_join(tables_2, keys)
    join_3 = chain_join(tables_3, keys)
    # join_4 = chain_join(tables_4, keys)
    # join_5 = chain_join(tables_5, keys)
    
    # js = [join_1, join_2, join_3, join_4, join_5]
    js = [join_1, join_2, join_3]

    print("step 2 over")

    # hs_1 = hash_j(join_1)
    # print("Hash success")

    # f = open("./tpch_1_test/q1_hs_3.pkl","wb")
    # pickle.dump(hs_1,f)
    # f.close()

    # hs_2 = hash_j(join_2)
    # print("Hash success")

    # f = open("./tpch_1_test/q2_hs_3.pkl","wb")
    # pickle.dump(hs_2,f)
    # f.close()

    # hs_3 = hash_j(join_3)
    # print("Hash success")

    # f = open("./tpch_1_test/q3_hs_3.pkl","wb")
    # pickle.dump(hs_3,f)
    # f.close()

    # hs_4 = hash_j(join_4)
    # print("Hash success")

    # f = open("./tpch_1_test/q4_hs_3.pkl","wb")
    # pickle.dump(hs_4,f)
    # f.close()

    # hs_5 = hash_j(join_5)
    # print("Hash success")

    # f = open("./tpch_1_test/q5_hs_3.pkl","wb")
    # pickle.dump(hs_5,f)
    # f.close()
    
    hs_1 = pickle.load(open("./tpch_1_test/q1_hs_3.pkl", "rb"))
    hs_2 = pickle.load(open("./tpch_1_test/q2_hs_3.pkl", "rb"))
    hs_3 = pickle.load(open("./tpch_1_test/q3_hs_3.pkl", "rb"))
    # hs_4 = pickle.load(open("./tpch_1_test/q4_hs_3.pkl", "rb"))
    # hs_5 = pickle.load(open("./tpch_1_test/q5_hs_3.pkl", "rb"))
    print("hash successfully loaded")
    
    # hss = [hs_1, hs_2, hs_3, hs_4, hs_5]
    hss = [hs_1, hs_2, hs_3]

    # ----------------------------------- sample from union non-bernoulli  ----------------------------------- 
    
    order = define_order(js)
    nb_S, nb_time = general_nb_sample(js, n, hss, order)
    print(nb_time)
    nb_S.to_csv(r'./tpch_1_test/new_nb_10000_3.csv')
    # with open('./tpch_1_test/new_nb_10000_3.csv.txt', 'w') as f:
    #     f.write(json.dumps(nb_time))


if __name__ == '__main__':
    main()