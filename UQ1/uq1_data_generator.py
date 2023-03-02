import json
import pandas as pd
import sys
sys.path.insert(0, './iidjoin')
from warm_up.acyclic_join import *
from warm_up.build_hash import *
import pickle

def fixed_sample(table, fixed, scale):

    table_fixed = table.iloc[:int(table.shape[0] * fixed), :]

    sample_frac = (scale - fixed) / (1 - fixed)
    table_sample = table.iloc[int(table.shape[0] * fixed):, :].sample(frac=sample_frac, replace=False)

    frames = [table_fixed, table_sample]
    sample = pd.concat(frames)

    return sample

def process_tpch(fixed, scale, nation, supplier, customer, orders, lineitem):

    # Change rows to random order
    supplier.sample(frac=1)
    customer.sample(frac=1)
    orders.sample(frac=1)
    lineitem.sample(frac=1)

    # should adjust percentage and scale according to table size
    nation_sample = nation
    supplier_sample = fixed_sample(supplier, fixed, scale).reset_index(drop=True)
    customer_sample = fixed_sample(customer, fixed, scale).reset_index(drop=True)
    orders_sample = fixed_sample(orders, fixed, scale).reset_index(drop=True)
    lineitem_sample = fixed_sample(lineitem, fixed, scale).reset_index(drop=True)

    return nation_sample, supplier_sample, customer_sample,orders_sample,lineitem_sample

def sort_by_index(tables):
    new_tables = []
    for table in tables:
        new_table = table.reset_index(drop=True)
        new_tables.append(new_table)
    return new_tables

def online_process(join):
    join.tables[1] = join.tables[1].rename(columns = {'SuppKey':'S_SuppKey'})
    join.tables[4] = join.tables[4].rename(columns = {'SuppKey':'L_SuppKey'})
    return join


def uq1_data_generator (size, scale, overlap, num_joins):
    # Qx: nation, supplier, customer, orders, lineitem 
    # read from tpch_{size}
    nation = pd.read_table('./tpch_' + str(size) + '/nation.tbl', index_col=False, names=['NationKey','NationName','RegionKey','Comment'], delimiter = '|').iloc[:, :-1]
    supplier = pd.read_table('./tpch_' + str(size) + '/supplier.tbl', index_col=False, names=['SuppKey','SuppName','Address','NationKey','Phone','Acctbl','Comment'], delimiter = '|').iloc[:, :-1]
    customer = pd.read_table('./tpch_' + str(size) + '/customer.tbl', index_col=False, names=['CustKey','CustName','Address','NationKey','Phone','Acctbal','MktSegment','Comment'], delimiter = '|').iloc[:, :-1]
    orders = pd.read_table('./tpch_' + str(size) + '/orders.tbl', index_col=False, names=['OrderKey','CustKey','OrderStatus','TotalPrice','OrderDate','OrderPriority','Clerk','ShipPriority','Comment'], delimiter = '|').iloc[:, :-1]
    lineitem = pd.read_table('./tpch_' + str(size) + '/lineitem.tbl', index_col=False, names=['OrderKey','PartKey','SuppKey','LineNumber','Quantity','ExtendedPrice','Discount','Tax','ReturnFlag','LineStatus',
                                                                                         'ShipDate','CommitDate','ReceiptDate','ShipInstruct','ShipMode','Comment'], delimiter = '|').iloc[:, :-1]
    keys = ['NationKey', 'NationKey', 'CustKey', 'OrderKey']
    pri_keys = ['NationKey', 'S_SuppKey', 'CustKey', 'OrderKey', 'LineNumber']
    
    print("read successfully")
    
    joins = []
    hs = []
    hs_pri = []
    
    for join_index in range(num_joins):
        # create tables
        nation_sample,supplier_sample,customer_sample,orders_sample,lineitem_sample = process_tpch(overlap, scale, nation, supplier, customer, orders, lineitem)
        nation_sample.to_csv('./uq1/tpch_' + str(size) + '_' + str(num_joins) + '/n' + str(join_index) + '.csv')
        supplier_sample.to_csv('./uq1/tpch_' + str(size) + '_' + str(num_joins) + '/s' + str(join_index) + '.csv')
        customer_sample.to_csv('./uq1/tpch_' + str(size) + '_' + str(num_joins) + '/c' + str(join_index) + '.csv')
        orders_sample.to_csv('./uq1/tpch_' + str(size) + '_' + str(num_joins) + '/o' + str(join_index) + '.csv')
        lineitem_sample.to_csv('./uq1/tpch_' + str(size) + '_' + str(num_joins) + '/l' + str(join_index) + '.csv')
        
        # convert tables to join
        tables = [nation_sample, supplier_sample, customer_sample, orders_sample, lineitem_sample]
        join = chain_join(tables, keys)
        joins.append(join)    
        print(str(join_index+1) + "th join is created")
        
        # hash join
        h = hash_j(join)
        hs.append(h)
        with open('./uq1/tpch_' + str(size) + '_' + str(num_joins) + '/hs_' + str(join_index) + '.pkl', 'wb') as f:
            pickle.dump(h, f)        
        
        join_pri = online_process(join)
        # hash primary keys
        h_pri = hash_j_pri(join_pri, pri_keys)
        hs_pri.append(h_pri)
        with open('./uq1/tpch_' + str(size) + '_' + str(num_joins) + '/hs_pri_' + str(join_index) + '.pkl', 'wb') as f:
            pickle.dump(h_pri, f)
            
        print(str(join_index+1) + "th join is hashed and stored")
        
        
    return joins, hs