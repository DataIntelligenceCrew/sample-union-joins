import argparse
import json
import pandas as pd
import numpy as np
import pickle
import sys
sys.path.insert(0, './iidjoin')
from warm_up.acyclic_join import *
from warm_up.build_hash import *
from warm_up.equi_chain_overlap import *
from sample_methods.sample_disjoint import *
from sample_methods.sample_b import *
from skew_data_generator import *

def main():
    # add arguments for n, k, alpha
    parser = argparse.ArgumentParser()
    # n is the final sample size
    parser.add_argument('--n', type=int, default=10000, help='final sample size')
    parser.add_argument('--scale', type=float, default=0.4, help='scale factor')
    parser.add_argument('--alpha', type=float, default=1.1, help='zipf distribution parameter')   
    parser.add_argument('--gen', action='store_true', default=False, help='If provided, will generate new data')
    parser.add_argument('--sample_method', type=str, default='olken', help='which method to use')
    parser.add_argument('--est_method', type=str, default='olken', help='which method to use')
    parser.add_argument('--init_method', type=str, default='olken', help='which init method to use for olken') 
    args = parser.parse_args()
    n, scale, alpha = args.n, args.scale, args.alpha
    
    if args.gen:
        print("generating new data")
        join_1, join_2, join_3, hs_1, hs_2, hs_3 = skew_data_generator(scale, alpha)
    else:
        tables_1 = []
        for index in range(5):
            tables_1.append(pd.read_csv('./skew_data/j1_{}.csv'.format(index), index_col=0))
        
        tables_2 = []
        for index in range(5):
            tables_2.append(pd.read_csv('./skew_data/j2_{}.csv'.format(index), index_col=0))
        
        tables_3 = []
        for index in range(5):
            tables_3.append(pd.read_csv('./skew_data/j3_{}.csv'.format(index), index_col=0))

        keys = ['NationKey', 'NationKey', 'CustKey', 'OrderKey']

        join_1 = chain_join(tables_1, keys)
        join_2 = chain_join(tables_2, keys)
        join_3 = chain_join(tables_3, keys)

        hs_1 = pickle.load(open("./skew_data/hs_1.pkl", "rb"))
        hs_2 = pickle.load(open("./skew_data/hs_2.pkl", "rb"))
        hs_3 = pickle.load(open("./skew_data/hs_3.pkl", "rb"))
        print("hash successfully loaded")
        
    joins = [join_1, join_2, join_3]
    hs = [hs_1, hs_2, hs_3]

    # ----------------------------------- Estimation ----------------------------------- 

    j_size = [e_size(join_1), e_size(join_2), e_size(join_3)]
    print("Estimated join sizes:")
    print(j_size)
    
    e_union = calc_U(joins)
    print("Estimated union size: ")
    print(e_union)

    join_1_f = join_1.f_join()
    join_2_f = join_2.f_join()
    join_3_f = join_3.f_join()
    
    exact_j = [join_1_f.shape[0], join_2_f.shape[0], join_3_f.shape[0]]
    print("Exact join sizes:")
    print(exact_j)
    
    exact_o = exact_olp([join_1_f, join_2_f, join_3_f])
    print("Exact overlap sizes:")
    print(exact_o)

    exact_union = pd.concat([join_1_f, join_2_f, join_3_f]).drop_duplicates() 
    print("Exact union size:")
    print(exact_union.shape[0])

    # ----------------------------------- sample from union bernoulli  ----------------------------------- 

    olken_u_b_S, olken_u_b_time = olken_olken_sample_union_bernoulli([join_1, join_2, join_3], n, [hs_1, hs_2, hs_3])
    print(olken_u_b_time)
    olken_u_b_S.to_csv(r'./skew_results/olken_b_{}_{}.csv'.format(scale, alpha))
    with open('./skew_results/olken_b_time_{}.txt'.format(alpha), 'w') as f:
        f.write(json.dumps(olken_u_b_time))
    
    exact_u_b_S, exact_u_b_time = exact_olken_sample_union_bernoulli([join_1, join_2, join_3], n, [hs_1, hs_2, hs_3])
    print(exact_u_b_time)
    exact_u_b_S.to_csv(r'./skew_results/exact_b_{}_{}.csv'.format(scale, alpha))
    with open('./skew_results/exact_b_time_{}_{}.txt'.format(n, alpha), 'w') as f:
        f.write(json.dumps(exact_u_b_time))

    # baseline
    # base_start = time.perf_counter()
    # join_1_f = join_1.f_join()
    # join_2_f = join_2.f_join()
    # join_3_f = join_3.f_join()
    # cur_time = time.perf_counter()
    # print(f"join in {cur_time - base_start:0.4f} seconds")
    # full_frames = [join_1_f, join_2_f, join_3_f]
    # join_f = pd.concat(full_frames).drop_duplicates()
    # base_sample_result = sample_from_join(join_f, n)
    # base_end = time.perf_counter()
    # print(f"baseline in {base_end - base_start:0.4f} seconds")


if __name__ == '__main__':
    main()