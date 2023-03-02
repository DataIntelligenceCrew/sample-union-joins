from collections import defaultdict
import pandas as pd

class chain_join:
    def __init__(self, tables, keys):
        self.tables = tables
        self.keys = keys

    def f_join(self):
        result = self.tables[0]
        for i in range(1,len(self.tables)):
            result = pd.merge(result, self.tables[i], on = self.keys[i-1], how = 'inner')
        return result


class acyclic_join:
    def __init__(self, root):
        self.root = root
            

class table_node:
    def __init__(self, table, parent, key):
        self.table = table
        self.parent = parent
        self.key = key
        self.childs = []
        self.hs = []
        
    def hash_acyc_j(self):
        if len(self.childs) == 0:
            self.hs = []
        else:
            for child in self.childs:
                h = defaultdict(list)
                for index, row in child.table.iterrows():
                    key = row[self.key]
                    h[key].append(index)
                self.hs.append(h)
                child.hash_acyc_j()
        
# standard chain join
class norm_chain_join:
    def __init__(self, tables, keys, join_type):
        self.tables = tables
        self.keys = keys
        self.join_type = join_type

    def f_join(self):
        result = self.tables[0]
        for i in range(1,len(self.tables)):
            result = pd.merge(result, self.tables[i], on = self.keys[i-1], how = 'inner')
        return result
    

class m_norm_chain_join:
    def __init__(self, Ms, join_type):
        self.Ms = Ms
        self.join_type = join_type
    
        
        