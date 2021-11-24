import dgl
from sklearn.model_selection import train_test_split
import scipy.sparse as ssp
import numpy as np

def _series_to_tensor(series):
    if is_categorical(series):
        return torch.LongTensor(series.cat.codes.values.astype('int64'))
    else:       # numeric
        return torch.FloatTensor(series.values)

class PandasGraphBuilder(object):
    def __init__(self):
        self.entity_tables = {}
        self.relation_tables = {}

        self.entity_pk_to_name = {}     # mapping from primary key name to entity name
        self.entity_pk = {}             # mapping from entity name to primary key
        self.entity_key_map = {}        # mapping from entity names to primary key values
        self.num_nodes_per_type = {}
        self.edges_per_relation = {}
        self.relation_name_to_etype = {}
        self.relation_src_key = {}      # mapping from relation name to source key
        self.relation_dst_key = {}      # mapping from relation name to destination key

    def add_entities(self, entity_table, primary_key, name):
        entities = entity_table[primary_key].astype('category')
        if not (entities.value_counts() == 1).all():
            raise ValueError('Different entity with the same primary key detected.')
        # preserve the category order in the original entity table
        entities = entities.cat.reorder_categories(entity_table[primary_key].values)

        self.entity_pk_to_name[primary_key] = name
        self.entity_pk[name] = primary_key
        self.num_nodes_per_type[name] = entity_table.shape[0]
        self.entity_key_map[name] = entities
        self.entity_tables[name] = entity_table

    def add_binary_relations(self, relation_table, source_key, destination_key, name):
        src = relation_table[source_key].astype('category')
        src = src.cat.set_categories(
            self.entity_key_map[self.entity_pk_to_name[source_key]].cat.categories)
        dst = relation_table[destination_key].astype('category')
        dst = dst.cat.set_categories(
            self.entity_key_map[self.entity_pk_to_name[destination_key]].cat.categories)
        if src.isnull().any():
            raise ValueError(
                'Some source entities in relation %s do not exist in entity %s.' %
                (name, source_key))
        if dst.isnull().any():
            raise ValueError(
                'Some destination entities in relation %s do not exist in entity %s.' %
                (name, destination_key))

        srctype = self.entity_pk_to_name[source_key]
        dsttype = self.entity_pk_to_name[destination_key]
        etype = (srctype, name, dsttype)
        self.relation_name_to_etype[name] = etype
        self.edges_per_relation[etype] = (src.cat.codes.values.astype('int64'), dst.cat.codes.values.astype('int64'))
        self.relation_tables[name] = relation_table
        self.relation_src_key[name] = source_key
        self.relation_dst_key[name] = destination_key

    def build(self):
        # Create heterograph
        graph = dgl.heterograph(self.edges_per_relation, self.num_nodes_per_type)
        return graph

def split_by_pid(df, group_by_val, train_size=.8, val_size = .1, test_size=.1):
    print("***Splitting by Playlist***")
    train_pids, all_test_pids = train_test_split(df[group_by_val].unique(), test_size=test_size+val_size, random_state=1)
    all_test = df[df.pid.isin(all_test_pids)]
    val_pids, test_pids = train_test_split(all_test[group_by_val].unique(), test_size=val_size, random_state=1)
    train = df[df.pid.isin(train_pids)]
    val = df[df.pid.isin(val_pids)]
    test = df[df.pid.isin(test_pids)]
    
    print("***Current Set has {} pids in train, {} pids in val, {} pids in test".format(len(train_pids), len(val_pids), len(test_pids)))
    return list(train.index), list(val.index), list(test.index)

def build_subgraph_graph(g, train_indices, utype, itype, etype, etype_rev):
    train_g = g.edge_subgraph(
        {etype: train_indices, etype_rev: train_indices},
        ) 
    try: 
        for ntype in g.ntypes:
            for col, data in g.nodes[ntype].data.items(): 
                train_g.nodes[ntype].data[col] = data
        for etype in g.etypes:
            for col, data in g.edges[etype].data.items():
                train_g.edges[etype].data[col] = data[train_g.edges[etype].data[dgl.EID]]
    except Exception as e: 
        print(e)
    return train_g

def build_val_test_matrix(g, val_indices, test_indices, utype, itype, etype):
    n_users = g.number_of_nodes(utype)
    n_items = g.number_of_nodes(itype)
    val_src, val_dst = g.find_edges(val_indices, etype=etype)
    test_src, test_dst = g.find_edges(test_indices, etype=etype)
    val_src = val_src.numpy()
    val_dst = val_dst.numpy()
    test_src = test_src.numpy()
    test_dst = test_dst.numpy()
    val_matrix = ssp.coo_matrix((np.ones_like(val_src), (val_src, val_dst)), (n_users, n_items))
    test_matrix = ssp.coo_matrix((np.ones_like(test_src), (test_src, test_dst)), (n_users, n_items))

    return val_matrix, test_matrix

