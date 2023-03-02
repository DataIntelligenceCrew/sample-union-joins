import json
import pandas as pd
import numpy as np
import pickle
import sys
sys.path.insert(0, './iidjoin')
from warm_up.acyclic_join import *
from warm_up.build_hash import *

# generate a list of n numbers with zipf distribution in range m
def gen_zipf_list_range(n, alpha, m):
    z = np.random.zipf(alpha, n)
    z = z.tolist()
    for i in range(len(z)):
        z[i] = z[i] % m
    return z


def process_tpch_skew(nation, supplier, customer, orders, lineitem, alpha):
    
    supplier.sample(frac=1)
    customer.sample(frac=1)
    orders.sample(frac=1)
    lineitem.sample(frac=1)

    z_s = gen_zipf_list_range(len(supplier), alpha, len(supplier))
    supplier_skew = supplier.iloc[z_s].sort_index().reset_index(drop=True)
    z_c = gen_zipf_list_range(len(customer), alpha, len(customer))
    customer_skew = customer.iloc[z_c].sort_index().reset_index(drop=True)
    z_o = gen_zipf_list_range(len(orders), alpha, len(orders))
    orders_skew = orders.iloc[z_o].sort_index().reset_index(drop=True)
    z_l = gen_zipf_list_range(len(lineitem), alpha, len(lineitem))
    lineitem_skew = lineitem.iloc[z_l].sort_index().reset_index(drop=True)
    
    return nation, supplier_skew, customer_skew,orders_skew,lineitem_skew


# sample from a list of tables and return a list of sampled tables
def gen_sample(tables, scale_factor):
    sample_tables = []
    for table in tables:
        new_table = table.sample(n=int(scale_factor * len(table))).sort_index()
        sample_tables.append(new_table)
    return sample_tables


def skew_data_generator(k, alpha):

    # n = 10000 # final sample size
    # k = 0.4 # scale factor
    # alpha = 1.2 # zipf distribution parameter

    # Qx: nation, supplier, customer, orders, lineitem 
    
    nation = pd.read_table('./tpch_1/nation.tbl', index_col=False, names=['NationKey','NationName','RegionKey','Comment'], delimiter = '|').iloc[:, :-1]
    supplier = pd.read_table('./tpch_1/supplier.tbl', index_col=False, names=['SuppKey','SuppName','Address','NationKey','Phone','Acctbl','Comment'], delimiter = '|').iloc[:, :-1]
    customer = pd.read_table('./tpch_1/customer.tbl', index_col=False, names=['CustKey','CustName','Address','NationKey','Phone','Acctbal','MktSegment','Comment'], delimiter = '|').iloc[:, :-1]
    orders = pd.read_table('./tpch_1/orders.tbl', index_col=False, names=['OrderKey','CustKey','OrderStatus','TotalPrice','OrderDate','OrderPriority','Clerk','ShipPriority','Comment'], delimiter = '|').iloc[:, :-1]
    lineitem = pd.read_table('./tpch_1/lineitem.tbl', index_col=False, names=['OrderKey','PartKey','SuppKey','LineNumber','Quantity','ExtendedPrice','Discount','Tax','ReturnFlag',
    'LineStatus','ShipDate','CommitDate','ReceiptDate','ShipinStruct','ShipMode','Comment'], delimiter = '|').iloc[:, :-1]

    print("read successfully")

    nation_skew, supplier_skew, customer_skew,orders_skew,lineitem_skew = process_tpch_skew(nation, supplier, customer, orders, lineitem, alpha)
    
    nation_skew.to_csv(r'./skew_data/nation_skew.csv')
    supplier_skew.to_csv(r'./skew_data/supplier_skew.csv')
    customer_skew.to_csv(r'./skew_data/customer_skew.csv')
    orders_skew.to_csv(r'./skew_data/orders_skew.csv')
    lineitem_skew.to_csv(r'./skew_data/lineitem_skew.csv')
    
    skew_tables = [nation_skew, supplier_skew, customer_skew, orders_skew, lineitem_skew]
    
    tables_1 = gen_sample(skew_tables, k)
    tables_2 = gen_sample(skew_tables, k)
    tables_3 = gen_sample(skew_tables, k)
    
    # store all the tables in a list and create separate csv file
    for i in range(len(tables_1)):
        tables_1[i].to_csv(r'./skew_data/j1_'+str(i)+'.csv') 
    
    for i in range(len(tables_2)):
        tables_2[i].to_csv(r'./skew_data/j2_'+str(i)+'.csv')
        
    for i in range(len(tables_3)):
        tables_3[i].to_csv(r'./skew_data/j3_'+str(i)+'.csv')
    
    print("successfully generated and stored")
    
    keys = ['NationKey', 'NationKey', 'CustKey', 'OrderKey']
    join_1 = chain_join(tables_1, keys)
    join_2 = chain_join(tables_2, keys)
    join_3 = chain_join(tables_3, keys)
    
    
    # tables_1 = []
    # for index in range(5):
    #     tables_1.append(pd.read_csv('./skew_data/j1_{}.csv'.format(index), index_col=0))
    
    # tables_2 = []
    # for index in range(5):
    #     tables_2.append(pd.read_csv('./skew_data/j2_{}.csv'.format(index), index_col=0))
    
    # tables_3 = []
    # for index in range(5):
    #     tables_3.append(pd.read_csv('./skew_data/j3_{}.csv'.format(index), index_col=0))

    # keys = ['NationKey', 'NationKey', 'CustKey', 'OrderKey']

    # join_1 = chain_join(tables_1, keys)
    # join_2 = chain_join(tables_2, keys)
    # join_3 = chain_join(tables_3, keys)


    print("join created")

    hs_1 = hash_j(join_1)
    print("Hash success")

    f = open("./skew_data/hs_1.pkl","wb")
    pickle.dump(hs_1,f)
    f.close()

    hs_2 = hash_j(join_2)
    print("Hash success")

    f = open("./skew_data/hs_1.pkl","wb")
    pickle.dump(hs_2,f)
    f.close()

    hs_3 = hash_j(join_3)
    print("Hash success")

    f = open("./skew_data/hs_3.pkl","wb")
    pickle.dump(hs_3,f)
    f.close()

    print("Hash stored")
    
    return join_1, join_2, join_3, hs_1, hs_2, hs_3