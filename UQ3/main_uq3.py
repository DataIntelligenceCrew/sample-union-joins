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
from uq3_sample_nb import *
from uq3_data_generator import *
from uq3_direct_overlap import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100000, help='final sample size')
    # add argument data size
    parser.add_argument('--size', type=int, default=5, help='original data size for each join path, can be 1, 3, 5, 10')
    parser.add_argument('--scale', type=float, default=0.4, help='scale factor')
    parser.add_argument('--overlap', type=float, default=0.1, help='overlap scale factor')
    parser.add_argument('--gen', action='store_true', default=False, help='If provided, will generate new data')
    parser.add_argument('--sample_method', type=str, default='disjoint', help='which method to use')
    parser.add_argument('--est_method', type=str, default='olken', help='which method to use for estimating overlap')
    parser.add_argument('--init_method', type=str, default='olken', help='which init method to use for sample') 
    parser.add_argument('--reuse', action='store_true', default=False, help='If provided, will reuse online data')
    parser.add_argument('--backtrack', action='store_true', default=False, help='If provided, will reuse, backtrack and update estimations')
    args = parser.parse_args()
    n, size, scale, overlap = args.n, args.size, args.scale, args.overlap
    
    js, hs, norm_js, hs_pri = uq3_data_generator(size, scale, overlap, args.gen)
    pri_keys = [['SuppKey', 'CustKey', 'OrderKey'], ['SuppKey', 'SuppKey', 'CustKey', 'CustKey', 'OrderKey'], ['SuppKey', 'CustKey', 'OrderKey', 'OrderKey']]
    
    if args.sample_method == 'nb':
         if args.est_method == 'olken' and args.init_method == 'olken':
            olken_nb_sample, olken_nb_time = olken_sample_union_nb(js, n, hs, norm_js)
            print(olken_nb_time)
            olken_nb_sample.to_csv(r'./uq3_results/olken_nb_{}.csv'.format(size))
            with open('./uq3_results/olken_nb_time_{}.txt'.format(size), 'w') as f:
                f.write(json.dumps(olken_nb_time))
                
         elif args.est_method == 'olken' and args.init_method == 'exact':
            exact_nb_sample, exact_nb_time = exact_sample_union_nb(js, n, hs, norm_js)
            print(exact_nb_time)
            exact_nb_sample.to_csv(r'./uq3_results/exact_nb_{}.csv'.format(size))
            with open('./uq3_results/exact_nb_time_{}.txt'.format(size), 'w') as f:
                f.write(json.dumps(exact_nb_time))
                
         elif args.est_method == 'online':   
            if args.reuse:
                online_reuse_sample, online_reuse_time = online_sample_union_nb_reuse(js, n, hs, js, hs_pri, pri_keys)
                print(online_reuse_time)
                online_reuse_sample.to_csv(r'./results_online_opt/uq3_online_reuse.csv')
                with open('./results_online_opt/uq3_online_reuse_time.txt', 'w') as f:
                    f.write(json.dumps(online_reuse_time))
                
            else: 
                online_nb_sample, online_nb_time = online_sample_union_nb(js, n, hs, js, hs_pri, pri_keys)
                print(online_nb_time)
                online_nb_sample.to_csv(r'./uq3_results/online_nb_{}.csv'.format(size))
                with open('./uq3_results/online_nb_time_{}.txt'.format(size), 'w') as f:
                    f.write(json.dumps(online_nb_time))
            
                
if __name__ == '__main__':
    main()