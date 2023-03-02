import json
import pandas as pd
import sys
sys.path.insert(0, './iidjoin')
from warm_up.acyclic_join import *
import pickle
import time

def fixed_sample(table, fixed, scale):

    table_fixed = table.iloc[:int(table.shape[0] * fixed), :]

    sample_frac = (scale - fixed) / (1 - fixed)
    table_sample = table.iloc[int(table.shape[0] * fixed):, :].sample(frac=sample_frac, replace=False)

    frames = [table_fixed, table_sample]
    sample = pd.concat(frames)

    return sample


def process_tpch(fixed, scale, supplier, customer, orders):
    # Change rows to random order
    supplier.sample(frac=1)
    customer.sample(frac=1)
    orders.sample(frac=1)

    # should adjust percentage and scale according to table size
    supplier_sample = fixed_sample(supplier, fixed, scale).reset_index(drop=True)
    customer_sample = fixed_sample(customer, fixed, scale).reset_index(drop=True)
    orders_sample = fixed_sample(orders, fixed, scale).reset_index(drop=True)

    return supplier_sample, customer_sample, orders_sample


def to_q1(tables):
    keys = ['NationKey', 'CustKey']
    join_q1 = chain_join(tables, keys)
    return join_q1


def to_q2(tables):

    origin_supplier = tables[0]
    origin_costomer = tables[1]

    # SUPPKEY, NATIONKEY, PHONE, ACCTBAL
    supplier_1 = origin_supplier.iloc[:, [0,3,4,5]]

    # CUSTKEY, NAME, ADDRESS
    customer_1 = origin_costomer.iloc[:, [0,1,2]]

    # CUSTKEY, NATIONKEY, PHONE, ACCTBAL, MKTSEGMENT
    customer_2 = origin_costomer.iloc[:, [0,3,4,5,6]]

    # SUPPKEY NAME ADDRESS
    supplier_2 = origin_supplier.iloc[:, [0,1,2]]

    orders = tables[2]

    # new_tables = [customer_1, customer_2, supplier_1, supplier_2, orders]
    # new_keys = ['CustKey', 'NationKey', 'SuppKey', 'CustKey']
    
    new_tables = [supplier_2, supplier_1, customer_2, customer_1, orders]
    new_keys = ['SuppKey', 'NationKey', 'CustKey', 'CustKey']

    join_q2 = chain_join(new_tables, new_keys)
    return join_q2


def to_q3(tables):
    
    supplier = tables[0]
    customer = tables[1]
    
    origin_orders = tables[2]
    
    # ORDERKEY, ORDERPRIORITY, CLERK, SHIPPRIORITY
    orders_1 = origin_orders.iloc[:, [0,5,6,7]]
    
    # ORDERKEY, CUSTKEY, ORDERSTATUS, TOTALPRICE, ORDERDATE
    orders_2 = origin_orders.iloc[:, [0,1,2,3,4]]
    
    tables = [supplier, customer, orders_2, orders_1]
    keys = ['NationKey', 'CustKey', 'OrderKey']
    
    join_q3 = chain_join(tables, keys)
    
    return join_q3

def q1_to_norm(join_q1):
    
    origin_supplier = join_q1.tables[0]
    origin_costomer = join_q1.tables[1]
    origin_orders = join_q1.tables[2]
    
    # SUPPKEY NAME ADDRESS
    supplier_2 = origin_supplier.iloc[:, [0,1,2]]
    
    # SUPPKEY, NATIONKEY, PHONE, ACCTBAL
    supplier_1 = origin_supplier.iloc[:, [0,3,4,5]]
    
    # CUSTKEY, NATIONKEY, PHONE, ACCTBAL, MKTSEGMENT
    customer_2 = origin_costomer.iloc[:, [0,3,4,5,6]]

    # CUSTKEY, NAME, ADDRESS
    customer_1 = origin_costomer.iloc[:, [0,1,2]]
    
    # ORDERKEY, CUSTKEY, ORDERSTATUS, TOTALPRICE, ORDERDATE
    orders_2 = origin_orders.iloc[:, [0,1,2,3,4]]

    # ORDERKEY, ORDERPRIORITY, CLERK, SHIPPRIORITY
    orders_1 = origin_orders.iloc[:, [0,5,6,7]]
    
    tables = [supplier_2, supplier_1, customer_2, customer_1, orders_2, orders_1]
    keys = ['SuppKey', 'NationKey', 'CustKey', 'CustKey', 'OrderKey']
    join_type = [False, True, False, True, False]
    
    norm_join_q1 = norm_chain_join(tables, keys, join_type)
    
    return norm_join_q1


def q2_to_norm(join_q2):
    
    origin_orders = join_q2.tables[4]
    
    # SUPPKEY NAME ADDRESS
    supplier_2 = join_q2.tables[0]
    
    # SUPPKEY, NATIONKEY, PHONE, ACCTBAL
    supplier_1 = join_q2.tables[1]
    
    # CUSTKEY, NATIONKEY, PHONE, ACCTBAL, MKTSEGMENT
    customer_2 = join_q2.tables[2]
    
    # CUSTKEY, NAME, ADDRESS
    customer_1 = join_q2.tables[3]
    
    # ORDERKEY, CUSTKEY, ORDERSTATUS, TOTALPRICE, ORDERDATE
    orders_2 = origin_orders.iloc[:, [0,1,2,3,4]]

    # ORDERKEY, ORDERPRIORITY, CLERK, SHIPPRIORITY
    orders_1 = origin_orders.iloc[:, [0,5,6,7]]
    
    tables = [supplier_2, supplier_1, customer_2, customer_1, orders_2, orders_1]
    keys = ['SuppKey', 'NationKey', 'CustKey', 'CustKey', 'OrderKey']
    join_type = [True, True, True, True, False]
    
    norm_join_q2 = norm_chain_join(tables, keys, join_type)
    
    return norm_join_q2

def q3_to_norm(join_q3):
    
    origin_supplier = join_q3.tables[0]
    origin_costomer = join_q3.tables[1]
    
    # SUPPKEY NAME ADDRESS
    supplier_2 = origin_supplier.iloc[:, [0,1,2]]
    
    # SUPPKEY, NATIONKEY, PHONE, ACCTBAL
    supplier_1 = origin_supplier.iloc[:, [0,3,4,5]]
    
    # CUSTKEY, NATIONKEY, PHONE, ACCTBAL, MKTSEGMENT
    customer_2 = origin_costomer.iloc[:, [0,3,4,5,6]]

    # CUSTKEY, NAME, ADDRESS
    customer_1 = origin_costomer.iloc[:, [0,1,2]]
    
    # ORDERKEY, CUSTKEY, ORDERSTATUS, TOTALPRICE, ORDERDATE
    orders_2 = join_q3.tables[2]

    # ORDERKEY, ORDERPRIORITY, CLERK, SHIPPRIORITY
    orders_1 = join_q3.tables[3]
    
    tables = [supplier_2, supplier_1, customer_2, customer_1, orders_2, orders_1]
    keys = ['SuppKey', 'NationKey', 'CustKey', 'CustKey', 'OrderKey']
    join_type = [False, True, False, True, True]
    
    norm_join_q3 = norm_chain_join(tables, keys, join_type)
    
    return norm_join_q3
    
    
def sort_by_index(tables):
    new_tables = []
    for table in tables:
        new_table = table.reset_index(drop=True)
        new_tables.append(new_table)
    return new_tables


def hash_j_uq3_index(j):
    n = len(j.tables)
    hs = [defaultdict(list) for _ in range(n-1)]
    for i in range(1,n):
        print(i)
        for index, row in j.tables[i].iterrows():
            # print(index)
            key = row[j.keys[i-1]]
            hs[i-1][key].append(index)
    return hs


def hash_j_uq3(j, pri_keys):
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

def uq3_data_generator(size, scale, overlap, gen):
    
    if gen:
        supplier = pd.read_table('./tpch_' + str(size) + '/supplier.tbl', index_col=False, names=['SuppKey','SuppName','Address','NationKey','Phone','Acctbl','Comment'], delimiter = '|').iloc[:, :-1]
        customer = pd.read_table('./tpch_' + str(size) + '/customer.tbl', index_col=False, names=['CustKey','CustName','Address','NationKey','Phone','Acctbal','MktSegment','Comment'], delimiter = '|').iloc[:, :-1]
        orders = pd.read_table('./tpch_' + str(size) + '/orders.tbl', index_col=False, names=['OrderKey','CustKey','OrderStatus','TotalPrice','OrderDate','OrderPriority','Clerk','ShipPriority','Comment'], delimiter = '|').iloc[:, :-1]
        
        supplier_sample_1,customer_sample_1,orders_sample_1 = process_tpch(overlap, scale, supplier, customer, orders)
        supplier_sample_2,customer_sample_2,orders_sample_2 = process_tpch(overlap, scale, supplier, customer, orders)
        supplier_sample_3,customer_sample_3,orders_sample_3 = process_tpch(overlap, scale, supplier, customer, orders)
        
        supplier_sample_1.to_csv('./uq3/tpch_' + str(size) + '/s1.csv', index=False)
        customer_sample_1.to_csv('./uq3/tpch_' + str(size) + '/c1.csv', index=False)
        orders_sample_1.to_csv('./uq3/tpch_' + str(size) + '/o1.csv', index=False)

        supplier_sample_2.to_csv('./uq3/tpch_' + str(size) + '/s2.csv', index=False)
        customer_sample_2.to_csv('./uq3/tpch_' + str(size) + '/c2.csv', index=False)
        orders_sample_2.to_csv('./uq3/tpch_' + str(size) + '/o2.csv', index=False)
        
        supplier_sample_3.to_csv('./uq3/tpch_' + str(size) + '/s3.csv', index=False)
        customer_sample_3.to_csv('./uq3/tpch_' + str(size) + '/c3.csv', index=False)
        orders_sample_3.to_csv('./uq3/tpch_' + str(size) + '/o3.csv', index=False)
        
        tables_1 = [supplier_sample_1, customer_sample_1, orders_sample_1]
        tables_2 = [supplier_sample_2, customer_sample_2, orders_sample_2]
        tables_3 = [supplier_sample_3, customer_sample_3, orders_sample_3]
        
    else:
        tables_1 = [pd.read_csv('./uq3/tpch_' + str(size) + '/s1.csv'), pd.read_csv('./uq3/tpch_' + str(size) + '/c1.csv'), pd.read_csv('./uq3/tpch_' + str(size) + '/o1.csv')]
        tables_2 = [pd.read_csv('./uq3/tpch_' + str(size) + '/s2.csv'), pd.read_csv('./uq3/tpch_' + str(size) + '/c2.csv'), pd.read_csv('./uq3/tpch_' + str(size) + '/o2.csv')]
        tables_3 = [pd.read_csv('./uq3/tpch_' + str(size) + '/s3.csv'), pd.read_csv('./uq3/tpch_' + str(size) + '/c3.csv'), pd.read_csv('./uq3/tpch_' + str(size) + '/o3.csv')]
        
    join_1 = to_q1(tables_1)
    join_2 = to_q2(tables_2)
    join_3 = to_q3(tables_3)
    
    print("join created")
    
    pri_keys = [['SuppKey', 'CustKey', 'OrderKey'], ['SuppKey', 'SuppKey', 'CustKey', 'CustKey', 'OrderKey'], ['SuppKey', 'CustKey', 'OrderKey', 'OrderKey']]
    
    if gen:
        hs_1 = hash_j_uq3_index(join_1)
        with open("./uq3/hs_1_index.pkl","wb") as f:
            pickle.dump(hs_1,f)
        hs_2 = hash_j_uq3_index(join_2)
        with open("./uq3/hs_2_index.pkl","wb") as f:
            pickle.dump(hs_2,f)
        hs_3 = hash_j_uq3_index(join_3)
        with open("./uq3/hs_3_index.pkl","wb") as f:
            pickle.dump(hs_3,f)
        
        hs_1_pri = hash_j_uq3(join_1, pri_keys[0])
        with open("./uq3/hs_1_pri.pkl","wb") as f:
            pickle.dump(hs_1_pri,f)
        
        hs_2_pri = hash_j_uq3(join_2, pri_keys[1])
        with open("./uq3/hs_2_pri.pkl","wb") as f:
            pickle.dump(hs_2_pri,f)
        
        hs_3_pri = hash_j_uq3(join_3, pri_keys[2])
        with open("./uq3/hs_3_pri.pkl","wb") as f:
            pickle.dump(hs_3_pri,f)
        
        print("hash done")
            
    else:
        hs_1 = pickle.load(open("./uq3/hs_1_index.pkl", "rb"))
        hs_2 = pickle.load(open("./uq3/hs_2_index.pkl", "rb"))
        hs_3 = pickle.load(open("./uq3/hs_3_index.pkl", "rb"))
        
        hs_1_pri = pickle.load(open("./uq3/hs_1_pri.pkl", "rb"))
        hs_2_pri = pickle.load(open("./uq3/hs_2_pri.pkl", "rb"))
        hs_3_pri = pickle.load(open("./uq3/hs_3_pri.pkl", "rb"))
    
    norm_join_1 = q1_to_norm(join_1)
    norm_join_2 = q2_to_norm(join_2)
    norm_join_3 = q3_to_norm(join_3)
    
    
    js = [join_1, join_2, join_3]
    hs = [hs_1, hs_2, hs_3]
    hs_pri = [hs_1_pri, hs_2_pri, hs_3_pri]
    norm_js = [norm_join_1, norm_join_2, norm_join_3]
    
    
    return js, hs, norm_js, hs_pri