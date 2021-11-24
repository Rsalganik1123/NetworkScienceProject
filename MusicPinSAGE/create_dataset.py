import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dgl

import os 
import sys 
import pickle 
import pandas as pd
import numpy as np 
from pandas.api.types import is_numeric_dtype, is_categorical_dtype, is_categorical
import torch
import pandas as pd 
import dgl
import dgl.function as fn
from sklearn.model_selection import train_test_split

import numpy as np
import dgl
import torch
from torch.utils.data import IterableDataset, DataLoader


from torch import nn
from sklearn.metrics import roc_auc_score, ndcg_score
import tqdm 

from graph_building import * 

directory = '/home/mila/r/rebecca.salganik/'
ns_music_all_data = pickle.load(open(directory+'ns_music_all_data_ming.p', 'rb'))
df_playlists = ns_music_all_data['df_playlist']
df_playlists_info = ns_music_all_data['df_playlist_info']
df_tracks = ns_music_all_data['df_track']
df_playlists_info = df_playlists_info.sort_values('pid').reset_index(drop=True)

graph_builder = PandasGraphBuilder()
graph_builder.add_entities(df_tracks, 'tid', 'track')
graph_builder.add_entities(df_playlists_info, 'pid', 'playlist')
graph_builder.add_binary_relations(df_playlists, 'pid', 'tid', 'contains')
graph_builder.add_binary_relations(df_playlists, 'tid', 'pid', 'contained_by')

g = graph_builder.build()

for key in ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']:
    
    g.nodes['track'].data[key] = torch.LongTensor(df_tracks[key].values)

g.nodes['track'].data['genre'] = torch.tensor(np.asarray(list(df_tracks['genre'].values))).float()

# g.nodes['track'].data['album_img_emb'] = torch.tensor(np.asarray(list(df_tracks['album_img_emb'].values)))
# g.nodes['track'].data['album_text_emb'] = torch.tensor(np.asarray(list(df_tracks['album_text_emb'].values)))

g.nodes['playlist'].data['id'] = torch.arange(g.number_of_nodes('playlist'))
g.nodes['track'].data['id'] = torch.arange(g.number_of_nodes('track'))

train_indices, val_indices, test_indices = split_by_pid(df = df_playlists, group_by_val = 'pid')
train_g = build_subgraph_graph(g, train_indices, 'track', 'playlist', 'contains', 'contained_by')
# val_g = build_subgraph_graph(g, val_indices, 'track', 'playlist', 'contains', 'contained_by')
val_matrix, test_matrix = build_val_test_matrix(g, val_indices, test_indices, 'playlist', 'track', 'contains')

dataset = {
        'full-graph': g,
        'train-graph': train_g,
        'val-graph': None, 
        'val-matrix': val_matrix,
        'test-matrix': test_matrix,
        'item-texts': None,
        'item-images': None,
        'user-type': 'playlist',
        'item-type': 'track',
        'user-to-item-type': 'contained_by',
        'item-to-user-type': 'contains'}

with open(directory+"dataset_without_im.pkl", 'wb') as f:
        pickle.dump(dataset, f)

