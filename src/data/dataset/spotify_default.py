import pickle
import torch
import numpy as np
from .build import DATASET_REGISTRY
from .dgl_builder import PandasGraphBuilder
import dgl


@DATASET_REGISTRY.register('SPOTIFY_MUSIC')
def build_spotify_graphs(cfg):
    cfg_data = cfg.DATASET
    all_data = pickle.load(open(cfg_data.DATA_PATH, 'rb'))
    df_users = all_data[cfg_data.USER_DF]
    df_interactions = all_data[cfg_data.INTERACTION_DF]
    df_items = all_data[cfg_data.ITEM_DF]
    train_indices = all_data[cfg_data.TRAIN_INDICES]
    train_user_ids = all_data['train_user_ids']
    val_user_ids = all_data['val_user_ids']
    test_user_ids = all_data['test_user_ids']
    df_users = df_users.sort_values(cfg_data.USER_ID).reset_index(drop=True)
    df_items = df_items.sort_values(cfg_data.ITEM_ID).reset_index(drop=True)

    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(df_items, cfg_data.ITEM_ID, cfg_data.ITEM)
    graph_builder.add_entities(df_users, cfg_data.USER_ID, cfg_data.USER)
    graph_builder.add_binary_relations(df_interactions, cfg_data.USER_ID, cfg_data.ITEM_ID, cfg_data.USER_ITEM_EDGE)
    graph_builder.add_binary_relations(df_interactions, cfg_data.ITEM_ID, cfg_data.USER_ID, cfg_data.ITEM_USER_EDGE)

    g = graph_builder.build()
    g.nodes[cfg_data.USER].data['id'] = torch.arange(g.number_of_nodes(cfg_data.USER))
    g.nodes[cfg_data.ITEM].data['id'] = torch.arange(g.number_of_nodes(cfg_data.ITEM))
    features = cfg_data.ITEM_FEATURES
    for key, feature_type in features:
        if feature_type == 'CAT':
            values = torch.LongTensor(df_items[key].values)
        else:
            values = torch.tensor(np.asarray(list(df_items[key].values))).float()
        g.nodes[cfg_data.ITEM].data[key] = values
    train_g = build_train_graph(g, train_indices, cfg_data.USER_ITEM_EDGE,
                                cfg_data.ITEM_USER_EDGE)
    return g, train_g, [train_user_ids, val_user_ids, test_user_ids]


def build_train_graph(g, train_indices, etype, etype_rev):
    train_g = g.edge_subgraph(
        {etype: train_indices, etype_rev: train_indices},
        relabel_nodes=False)

    # copy features
    for ntype in g.ntypes:
        for col, data in g.nodes[ntype].data.items():
            train_g.nodes[ntype].data[col] = data
    for etype in g.etypes:
        for col, data in g.edges[etype].data.items():
            train_g.edges[etype].data[col] = data[train_g.edges[etype].data[dgl.EID]]

    return train_g
