import json
import pandas as pd
import sys
sys.path.insert(0, './iidjoin')
from warm_up.acyclic_join import *
from warm_up.build_hash import *
import pickle
import time

def uq2_data_generator (size, gen):
    # Qx: nation, supplier, customer, orders, lineitem 
    # read from tpch_{size}
    region = pd.read_table('./tpch_' + str(size) + '/region.tbl', index_col=False, names=['RegionKey','RegionName','Comment'], delimiter = '|').iloc[:, :-1]
    nation = pd.read_table('./tpch_' + str(size) + '/nation.tbl', index_col=False, names=['NationKey','NationName','RegionKey','Comment'], delimiter = '|').iloc[:, :-1]
    supplier = pd.read_table('./tpch_' + str(size) + '/supplier.tbl', index_col=False, names=['SuppKey','SuppName','Address','NationKey','Phone','Acctbl','Comment'], delimiter = '|').iloc[:, :-1]
    partsupp = pd.read_table('./tpch_' + str(size) + '/partsupp.tbl', index_col=False, names=['PartKey','SuppKey', 'Availqty', 'SupplyCost', 'Comment'], delimiter = '|').iloc[:, :-1]
    part = pd.read_table('./tpch_' + str(size) + '/part.tbl', index_col=False, names=['PartKey','PartName','Mfgr','Brand','Type','Size','Container', 'RetailPrice','Comment'], delimiter = '|').iloc[:, :-1]
    keys = ['RegionKey', 'NationKey', 'SuppKey', 'PartKey']
    pri_keys = ['RegionKey', 'NationKey', 'SuppKey', 'PartKey', 'PartKey']
    
    print("read successfully")
    
    joins = []
    hs = []
    hs_pri = []
    
    if gen:
        preprocess_start = time.perf_counter()
        
        nation_1 = nation.loc[nation['NationKey'] == 0].reset_index(drop=True)
        supplier_1 = supplier.loc[supplier['NationKey'] == 0].reset_index(drop=True)
        
        partsupp_2 = partsupp.loc[partsupp['PartKey'] % 2 == 0].reset_index(drop=True)
        part_2 = part.loc[part['PartKey'] % 2 == 0].reset_index(drop=True)
        
        supplier_3 = supplier.loc[supplier['SuppKey'] % 2 == 0].reset_index(drop=True)
        partsupp_3 = partsupp.loc[partsupp['SuppKey'] % 2 == 0].reset_index(drop=True)
        
        preprocess_end = time.perf_counter()
        print("preprocess time: " + str(preprocess_end - preprocess_start))
        
        nation_1.to_csv('./uq2/tpch_' + str(size) + '/n1.csv')
        supplier_1.to_csv('./uq2/tpch_' + str(size) + '/s1.csv')
        
        partsupp_2.to_csv('./uq2/tpch_' + str(size) + '/ps2.csv')
        part_2.to_csv('./uq2/tpch_' + str(size) + '/p2.csv')
        
        supplier_3.to_csv('./uq2/tpch_' + str(size) + '/s3.csv')
        partsupp_3.to_csv('./uq2/tpch_' + str(size) + '/p3.csv')
        
    else:
        nation_1 = pd.read_csv('./uq2/tpch_' + str(size) + '/n1.csv')
        supplier_1 = pd.read_csv('./uq2/tpch_' + str(size) + '/s1.csv')
        
        partsupp_2 = pd.read_csv('./uq2/tpch_' + str(size) + '/ps2.csv')
        part_2 = pd.read_csv('./uq2/tpch_' + str(size) + '/p2.csv')
        
        supplier_3 = pd.read_csv('./uq2/tpch_' + str(size) + '/s3.csv')
        partsupp_3 = pd.read_csv('./uq2/tpch_' + str(size) + '/p3.csv')
        
            
    tables_1 = [region, nation_1, supplier_1, partsupp, part]
    tables_2 = [region, nation, supplier, partsupp_2, part_2]
    tables_3 = [region, nation, supplier_3, partsupp_3, part]
    
    joins = [chain_join(tables_1, keys), chain_join(tables_2, keys), chain_join(tables_3, keys)]
    print("joins are generated")
    
    if gen:
        for join_index in range(len(joins)):
            # hash join
            h = hash_j(joins[join_index])
            hs.append(h)
            with open('./uq2/tpch_' + str(size) + '/hs_' + str(join_index) + '.pkl', 'wb') as f:
                pickle.dump(h, f)
                
            h_pri = hash_j_pri(joins[join_index], pri_keys)
            hs_pri.append(h_pri)
            with open('./uq2/tpch_' + str(size) + '/hs_pri_' + str(join_index) + '.pkl', 'wb') as f:
                pickle.dump(h_pri, f)
            print(str(join_index+1) + "th join is hashed and stored")
    else:
        for join_index in range(len(joins)):
            hs.append(pickle.load(open('./uq2/tpch_' + str(size) + '/hs_' + str(join_index) + '.pkl', 'rb')))
            hs_pri.append(pickle.load(open('./uq2/tpch_' + str(size) + '/hs_pri_' + str(join_index) + '.pkl', 'rb')))
            print(str(join_index+1) + "th hash is loaded")
            
    # join_pri = online_process(join)
    # # hash primary keys
    # h_pri = hash_j_pri(join_pri)
    # hs_pri.append(h_pri)
    # with open('./uq2/tpch_' + str(size) + '_' + str(num_joins) + '/hs_pri' + str(join_index) + '.pkl', 'wb') as f:
    #     pickle.dump(h_pri, f)
    
        
    return joins, hs, hs_pri