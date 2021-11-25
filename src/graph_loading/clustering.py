
import os
import torch
import dgl
import pickle
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_categorical_dtype, is_categorical
import torch
import pandas as pd
import dgl.function as fn
import numpy as np
import dgl
import torch
from torch.utils.data import IterableDataset, DataLoader
import networkx as nx
from networkx.algorithms import bipartite
import scipy

# B = nx.Graph()
# # Add nodes with the node attribute "bipartite"
# B.add_nodes_from([1, 2, 3, 4], bipartite=0)
# B.add_nodes_from(["a", "b", "c"], bipartite=1)
# # Add edges only between nodes of opposite node sets
# B.add_edges_from([(1, "a"), (1, "b"), (2, "b"), (2, "c"), (3, "c"), (4, "a")])
#
# c = bipartite.clustering(B)
# print(c)


ns_music_all_data = pickle.load(open('/Users/juju/Desktop/ns_music_all_data_ming.pkl', 'rb'))

df_playlists = ns_music_all_data['df_playlist']
df_playlists_info = ns_music_all_data['df_playlist_info']
df_tracks = ns_music_all_data['df_track']

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

graph_builder = PandasGraphBuilder()

df_playlists_info = df_playlists_info.sort_values('pid').reset_index(drop=True)

graph_builder = PandasGraphBuilder()
graph_builder.add_entities(df_tracks, 'tid', 'track')
graph_builder.add_entities(df_playlists_info, 'pid', 'playlist')
graph_builder.add_binary_relations(df_playlists, 'pid', 'tid', 'contains')
graph_builder.add_binary_relations(df_playlists, 'tid', 'pid', 'contained_by')

g = graph_builder.build()
# #gh = dgl.to_homogeneous(g)
# print("done")
# print(g.adj(etype="contains").shape)

d_c = g.out_degrees(etype="contains")
d_c_by = g.out_degrees(etype="contained_by")
adj_contains = g.adj(scipy_fmt= "csr", etype="contains") #shape = [1000000, 2262190]
adj_contained_by = g.adj(scipy_fmt= "csr", etype="contained_by") #shape = [2262190, 1000000]

#first contains
A3 = adj_contains.power(3)
ccs_c = []
degrees = d_c
for i in range(len(degrees)):
    cc = A3[i, i]/(degrees[i]*(degrees[i]-1))
    if np.isnan(cc): ccs_c.append(0)
    elif np.isinf(cc): ccs_c.append(1)
    else: ccs_c.append(cc)

#first contained by
A3 = adj_contained_by.power(3)
ccs_c_by = []
degrees = d_c_by
for i in range(len(degrees)):
    cc = A3[i, i]/(degrees[i]*(degrees[i]-1))
    if np.isnan(cc): ccs_c_by.append(0)
    elif np.isinf(cc): ccs_c_by.append(1)
    else: ccs_c_by.append(cc)

#print plots
hist = Counter(ccs_c)
freq, cc = list(hist.keys()), list(hist.values())
plt.scatter(freq, cc, color="blue")
plt.xlabel("Clustering Coefficient")
plt.ylabel("Frequency")
plt.title('Clustering Coefficient Distribution for Edge "Contains"')
plt.show()

hist = Counter(ccs_c_by)
freq, cc = list(hist.keys()), list(hist.values())
plt.scatter(freq, cc, color="blue")
plt.xlabel("Clustering Coefficient")
plt.ylabel("Frequency")
plt.title('Clustering Coefficient Distribution for Edge "Contained by"')
plt.show()




#

# print("Number of connected components: ", out[0])
# ypoints = out[1]
# xpoints = list(range(0,(len(out[1]))))
# occurence = list(range((out[0]))) #list of [0,3,29, ... ] max is 116
# for i in out[1]:
#     occurence[i]=occurence[i]+1
#
# plt.plot(occurence, list(range(0,out[0])))
# plt.xlabel = "Label for each connected components"
# plt.ylabel = "Number of nodes in connected component"
# plt.show()

# out = scipy.sparse.csgraph.shortest_path(nxg)
# print(out.shape)
# print(out)
#
