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
from uq1_data_generator import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100000, help='final sample size')
    # add argument data size
    parser.add_argument('--size', type=int, default=5, help='original data size for each join path, can be 1, 3, 5, 10')
    parser.add_argument('--scale', type=float, default=0.4, help='scale factor')
    parser.add_argument('--overlap', type=float, default=0.1, help='overlap scale factor')
    parser.add_argument('--num_joins', type=int, default=3, help='number of joins')
    parser.add_argument('--gen', action='store_true', default=False, help='If provided, will generate new data')
    parser.add_argument('--sample_method', type=str, default='disjoint', help='which method to use')
    parser.add_argument('--est_method', type=str, default='olken', help='which method to use for estimating overlap')
    parser.add_argument('--init_method', type=str, default='olken', help='which init method to use for sample') 
    parser.add_argument('--reuse', action='store_true', default=False, help='If provided, will reuse online data')
    parser.add_argument('--backtrack', action='store_true', default=False, help='If provided, will reuse, backtrack and update estimations')
    args = parser.parse_args()
    n, size, scale, overlap, num_joins = args.n, args.size, args.scale, args.overlap, args.num_joins
    
    keys = ['NationKey', 'NationKey', 'CustKey', 'OrderKey']
    pri_keys = ['NationKey', 'S_SuppKey', 'CustKey', 'OrderKey', 'LineNumber']
    
    if args.gen:
        print("generating new data")
        joins, hs = uq1_data_generator(size, scale, overlap, num_joins)
    else:
        joins = []
        joins_pri = []
        hs = []
        hs_pri = []
        for join_index in range(num_joins):
            tables = []
            tables.append(pd.read_csv('./uq1/tpch_' + str(size) + '_' + str(num_joins) + '/n' + str(join_index) + '.csv', index_col=0))
            tables.append(pd.read_csv('./uq1/tpch_' + str(size) + '_' + str(num_joins) + '/s' + str(join_index) + '.csv', index_col=0))
            tables.append(pd.read_csv('./uq1/tpch_' + str(size) + '_' + str(num_joins) + '/c' + str(join_index) + '.csv', index_col=0))
            tables.append(pd.read_csv('./uq1/tpch_' + str(size) + '_' + str(num_joins) + '/o' + str(join_index) + '.csv', index_col=0))
            tables.append(pd.read_csv('./uq1/tpch_' + str(size) + '_' + str(num_joins) + '/l' + str(join_index) + '.csv', index_col=0))
            join = chain_join(tables, keys)
            joins.append(join)
            h = pickle.load(open('./uq1/tpch_' + str(size) + '_' + str(num_joins) + '/hs_' + str(join_index) + '.pkl', "rb"))
            hs.append(h)
            join_pri = online_process(join)
            joins_pri.append(join_pri)
            h_pri = pickle.load(open('./uq1/tpch_' + str(size) + '_' + str(num_joins) + '/hs_pri_' + str(join_index) + '.pkl', "rb"))
            hs_pri.append(h_pri)
            print("loaded join {}".format(join_index+1))

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
        
    
    if args.sample_method == "disjoint":
        if args.init_method == "olken":
            olken_disjoint_sample, olken_disjoint_time = olken_sample_from_disjoint(joins, n, hs)
            print(olken_disjoint_time)
            olken_disjoint_sample.to_csv(r'./uq1_results/olken_disjoint_{}_{}.csv'.format(size, overlap))
            with open('./uq1_results/olken_disjoint_time_{}_{}.txt'.format(size, overlap), 'w') as f:
                f.write(json.dumps(olken_disjoint_time))
        
        elif args.init_method == "exact":
            exact_disjoint_sample, exact_disjoint_time = exact_sample_from_disjoint(joins, n, hs)
            print(exact_disjoint_time)
            exact_disjoint_sample.to_csv(r'./uq1_results/exact_disjoint_{}_{}.csv'.format(size, overlap))
            with open('./uq1_results/exact_disjoint_time_{}_{}.txt'.format(size, overlap), 'w') as f:
                f.write(json.dumps(exact_disjoint_time))
    
        
    elif args.sample_method == 'b':
        if args.est_method == 'olken' and args.init_method == 'olken':
            olken_b_sample, olken_b_time = olken_olken_sample_union_bernoulli(joins, n, hs)
            print(olken_b_time)
            olken_b_sample.to_csv(r'./uq1_results/olken_b_{}_{}.csv'.format(size, overlap))
            with open('./uq1_results/olken_b_time_{}_{}.txt'.format(size, overlap), 'w') as f:
                f.write(json.dumps(olken_b_time))
                
        elif args.est_method == 'olken' and args.init_method == 'exact':
            exact_b_sample, exact_b_time = exact_olken_sample_union_bernoulli(joins, n, hs)
            print(exact_b_time)
            exact_b_sample.to_csv(r'./uq1_results/exact_b_{}_{}.csv'.format(size, overlap))
            with open('./uq1_results/exact_b_time_{}_{}.txt'.format(size, overlap), 'w') as f:
                f.write(json.dumps(exact_b_time))
                
        elif args.est_method == 'online':     
            online_b_sample, online_b_time = online_sample_union_bernoulli(joins, n, hs, joins_pri, hs_pri)
            print(online_b_time)
            online_b_sample.to_csv(r'./uq1_results/online_b_{}_{}.csv'.format(size, overlap))
            with open('./uq1_results/online_b_time_{}_{}.txt'.format(size, overlap), 'w') as f:
                f.write(json.dumps(online_b_time))
                
                
    elif args.sample_method == 'nb':
         if args.est_method == 'olken' and args.init_method == 'olken':
            olken_nb_sample, olken_nb_time = olken_sample_union_nb(joins, n, hs)
            print(olken_nb_time)
            olken_nb_sample.to_csv(r'./uq1_results/olken_nb_{}_{}.csv'.format(size, num_joins))
            with open('./uq1_results/olken_nb_time_{}_{}.txt'.format(size, num_joins), 'w') as f:
                f.write(json.dumps(olken_nb_time))
                
         elif args.est_method == 'olken' and args.init_method == 'exact':
            exact_nb_sample, exact_nb_time = exact_sample_union_nb(joins, n, hs)
            print(exact_nb_time)
            exact_nb_sample.to_csv(r'./uq1_results/exact_nb_{}_{}.csv'.format(size, num_joins))
            with open('./uq1_results/exact_nb_time_{}_{}.txt'.format(size, num_joins), 'w') as f:
                f.write(json.dumps(exact_nb_time))
                
         elif args.est_method == 'online':   
            if args.backtrack:
                online_back_sample, online_back_time = online_sample_union_nb_back(joins, n, hs, joins_pri, hs_pri, pri_keys)
                print(online_back_time)
                online_back_sample.to_csv(r'./results_online_opt/uq1_online_back.csv')
                with open('./results_online_opt/uq1_online_back_time.txt', 'w') as f:
                    f.write(json.dumps(online_back_time))
            elif args.reuse:
                online_reuse_sample, online_reuse_time = online_sample_union_nb_reuse(joins, n, hs, joins_pri, hs_pri, pri_keys)
                print(online_reuse_time)
                online_reuse_sample.to_csv(r'./results_online_opt/uq1_online_reuse.csv')
                with open('./results_online_opt/uq1_online_reuse_time.txt', 'w') as f:
                    f.write(json.dumps(online_reuse_time))
            else:
                online_nb_sample, online_nb_time = online_sample_union_nb(joins, n, hs, joins_pri, hs_pri, pri_keys)
                print(online_nb_time)
                online_nb_sample.to_csv(r'./uq1_results/online_nb_{}_{}.csv'.format(size, num_joins))
                with open('./uq1_results/online_nb_time_{}_{}.txt'.format(size, num_joins), 'w') as f:
                    f.write(json.dumps(online_nb_time))
    

if __name__ == '__main__':
    main()