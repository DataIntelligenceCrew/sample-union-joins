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
from sample_methods.sample_nb import *
from reuse_backtrack.sample_reuse_backtrack import *
from uq2_data_generator import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100000, help='final sample size')
    # add argument data size
    parser.add_argument('--size', type=int, default=5, help='original data size for each join path, can be 1, 3, 5, 10')
    parser.add_argument('--gen', action='store_true', default=False, help='If provided, will generate new data')
    parser.add_argument('--sample_method', type=str, default='nb', help='which method to use')
    parser.add_argument('--est_method', type=str, default='olken', help='which method to use for estimating overlap')
    parser.add_argument('--init_method', type=str, default='olken', help='which init method to use for sample') 
    parser.add_argument('--reuse', action='store_true', default=False, help='If provided, will reuse online data')
    parser.add_argument('--backtrack', action='store_true', default=False, help='If provided, will reuse, backtrack and update estimations')
    args = parser.parse_args()
    n, size = args.n, args.size
    
    print("loading data")
    joins, hs, hs_pri = uq2_data_generator(size, args.gen)
    pri_keys = ['RegionKey', 'NationKey', 'SuppKey', 'PartKey', 'PartKey']
    
    

    # ----------------------------------- Estimation ----------------------------------- 
    # j_size = []
    # for join_index in range(num_joins):
    #     j_size.append(e_size(joins[join_index]))
    # print("Estimated join sizes:")
    # print(j_size)
    
    # ans, Os = gen_os(joins)
    # print("Estimated overlap sizes: ")
    # print(ans)
    # print(Os)

    # As = calc_As(joins, Os, ans, j_size)
    # e_union = calc_U(As)
    # print("Estimated union size: ")
    # print(e_union)
    
    # # generic exact joins and overlaps
    # full_joins = []
    # exact_j = []
    # for join_index in range(num_joins):
    #     f_join = joins[join_index].f_join()
    #     full_joins.append(f_join)
    #     exact_j.append(f_join.shape[0])
    
    # print("Exact join sizes:")
    # print(exact_j)
    
    # exact_o = exact_olp(full_joins)
    # print("Exact overlap sizes:")
    # print(exact_o)

    # exact_union = pd.concat(full_joins).drop_duplicates() 
    # print("Exact union size:")
    # print(exact_union.shape[0])

    # ----------------------------------- sample  ----------------------------------- 
        
    
    if args.sample_method == 'b':
        if args.est_method == 'olken' and args.init_method == 'olken':
            olken_b_sample, olken_b_time = olken_olken_sample_union_bernoulli(joins, n, hs)
            print(olken_b_time)
            olken_b_sample.to_csv(r'./uq2_results/olken_b_{}.csv'.format(size))
            with open('./uq2_results/olken_b_time_{}.txt'.format(size), 'w') as f:
                f.write(json.dumps(olken_b_time))
                
        elif args.est_method == 'olken' and args.init_method == 'exact':
            exact_b_sample, exact_b_time = exact_olken_sample_union_bernoulli(joins, n, hs)
            print(exact_b_time)
            exact_b_sample.to_csv(r'./uq2_results/exact_b_{}.csv'.format(size))
            with open('./uq2_results/exact_b_time_{}.txt'.format(size), 'w') as f:
                f.write(json.dumps(exact_b_time))
                
        elif args.est_method == 'online':     
            online_b_sample, online_b_time = online_sample_union_bernoulli(joins, n, hs, joins, hs_pri, pri_keys)
            print(online_b_time)
            online_b_sample.to_csv(r'./uq2_results/online_b_{}.csv'.format(size))
            with open('./uq2_results/online_b_time_{}.txt'.format(size), 'w') as f:
                f.write(json.dumps(online_b_time))
                
                
    # TODO: change to non-bernoulli
    elif args.sample_method == 'nb':
         if args.est_method == 'olken' and args.init_method == 'olken':
            olken_nb_sample, olken_nb_time = olken_sample_union_nb(joins, n, hs)
            print(olken_nb_time)
            olken_nb_sample.to_csv(r'./uq2_results/olken_nb_{}.csv'.format(size))
            with open('./uq2_results/olken_nb_time_{}.txt'.format(size), 'w') as f:
                f.write(json.dumps(olken_nb_time))
                
         elif args.est_method == 'olken' and args.init_method == 'exact':
            exact_nb_sample, exact_nb_time = exact_sample_union_nb(joins, n, hs)
            print(exact_nb_time)
            exact_nb_sample.to_csv(r'./uq2_results/exact_nb_{}.csv'.format(size))
            with open('./uq2_results/exact_nb_time_{}.txt'.format(size), 'w') as f:
                f.write(json.dumps(exact_nb_time))
                
         elif args.est_method == 'online':   
            if args.backtrack:
                online_back_sample, online_back_time = online_sample_union_nb_back(joins, n, hs, joins, hs_pri, pri_keys)
                print(online_back_time)
                online_back_sample.to_csv(r'./results_online_opt/uq2_online_back.csv')
                with open('./results_online_opt/uq2_online_back_time.txt', 'w') as f:
                    f.write(json.dumps(online_back_time))
            elif args.reuse:
                online_reuse_sample, online_reuse_time = online_sample_union_nb_reuse(joins, n, hs, joins, hs_pri, pri_keys)
                print(online_reuse_time)
                online_reuse_sample.to_csv(r'./results_online_opt/uq2_online_reuse.csv')
                with open('./results_online_opt/uq2_online_reuse_time.txt', 'w') as f:
                    f.write(json.dumps(online_reuse_time))
            else:
                online_nb_sample, online_nb_time = online_sample_union_nb(joins, n, hs, joins, hs_pri, pri_keys)
                print(online_nb_time)
                online_nb_sample.to_csv(r'./uq2_results/online_nb_{}.csv'.format(size))
                with open('./uq2_results/online_nb_time_{}.txt'.format(size), 'w') as f:
                    f.write(json.dumps(online_nb_time))
    

if __name__ == '__main__':
    main()