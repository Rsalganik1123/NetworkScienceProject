import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn


def init_embeddings(g, cfg):
    emb_types = cfg.MODEL.PINSAGE.PROJECTION.EMB
    data = g.nodes[cfg.DATASET.ITEM].data
    module_dict = torch.nn.ModuleDict()

    for key, size in emb_types:
        module_dict[key] = torch.nn.Embedding(data[key].max() + 1, size)

    return module_dict


class LinearProjector(torch.nn.Module):
    """
    Projects each input feature of the graph linearly and sums them up
    """

    def __init__(self, full_graph, cfg):
        super().__init__()

        self.ntype = cfg.DATASET.ITEM
        self.embeddings = init_embeddings(full_graph, cfg)
        self.hidden_size = cfg.MODEL.PINSAGE.HIDDEN_SIZE
        self.concat_feature_types = cfg.MODEL.PINSAGE.PROJECTION.CONCAT
        self.all_features = cfg.MODEL.PINSAGE.PROJECTION.FEATURES

        self.album_features = [x for x in self.all_features if x in ['album_img_emb', 'album_text_emb']]
        data = full_graph.nodes[cfg.DATASET.ITEM].data

        if len(self.album_features) > 0:
            album_feature_size = 0
            for key in self.album_features:
                _, dim = data[key].shape
                album_feature_size += dim
            self.fc_album = torch.nn.Linear(album_feature_size, self.hidden_size)
        else:
            self.fc_album = None

        concat_size = 0
        for key in self.concat_feature_types:

            if key in self.embeddings:
                embs = self.embeddings[key]
                concat_size += embs.embedding_dim
            else:
                _, dim = data[key].shape
                concat_size += dim

        if self.fc_album is not None:
            concat_size += self.hidden_size

        self.concat_size = concat_size
        if concat_size > 0:
            self.fc = torch.nn.Linear(concat_size, self.hidden_size)
        else:
            self.fc = None
        self.add_feature_types = cfg.MODEL.PINSAGE.PROJECTION.ADD
        if cfg.MODEL.PINSAGE.PROJECTION.NORMALIZE:
            self.norm = torch.nn.LayerNorm(self.hidden_size)
        else:
            self.norm = None

    def forward(self, ndata):

        features = {}
        for key in self.all_features:

            if key in self.embeddings:
                module = self.embeddings[key]
                value = module(ndata[key])
            else:
                value = ndata[key]
            features[key] = value

        projection = 0
        for key in self.add_feature_types:
            projection = projection + features[key]

        if len(self.album_features) > 0:
            album_feature = torch.cat([features[x] for x in self.album_features], dim=1)
            album_feature = self.fc_album(album_feature)
        else:
            album_feature = None

        concat_features = []
        for key in self.concat_feature_types:
            concat_features.append(features[key])
        if album_feature is not None:
            concat_features.append(album_feature)
        if len(concat_features) > 0:
            concat_features = torch.cat(concat_features, dim=1)
            projection = projection + self.fc(concat_features)
        if self.norm:
            projection = self.norm(projection)

        return projection


class WeightedSAGEConv(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, act=F.relu):
        super().__init__()

        self.act = act
        self.Q = nn.Linear(input_dims, hidden_dims)
        self.W = nn.Linear(input_dims + hidden_dims, output_dims)
        self.reset_parameters()
        self.dropout = nn.Dropout(0.5)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.Q.weight, gain=gain)
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.constant_(self.Q.bias, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, g, h, weights):
        """
        g : graph
        h : node features
        weights : scalar edge weights
        """
        h_src, h_dst = h
        with g.local_scope():
            g.srcdata['n'] = self.act(self.Q(self.dropout(h_src)))
            g.edata['w'] = weights.float()
            g.update_all(fn.u_mul_e('n', 'w', 'm'), fn.sum('m', 'n'))
            g.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'ws'))
            n = g.dstdata['n']
            ws = g.dstdata['ws'].unsqueeze(1).clamp(min=1)
            z = self.act(self.W(self.dropout(torch.cat([n / ws, h_dst], 1))))
            z_norm = z.norm(2, 1, keepdim=True)
            z_norm = torch.where(z_norm == 0, torch.tensor(1.).to(z_norm), z_norm)
            z = z / z_norm
            return z


class SAGENet(nn.Module):
    def __init__(self, hidden_dims, n_layers):
        """
        g : DGLHeteroGraph
            The user-item interaction graph.
            This is only for finding the range of categorical variables.
        item_textsets : torchtext.data.Dataset
            The textual features of each item node.
        """
        super().__init__()

        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(WeightedSAGEConv(hidden_dims, hidden_dims, hidden_dims))

    def forward(self, blocks, h):
        for layer, block in zip(self.convs, blocks):
            h_dst = h[:block.number_of_nodes('DST/' + block.ntypes[0])]
            h = layer(block, (h, h_dst), block.edata['weights'])
        return h


class ItemToItemScorer(nn.Module):
    def __init__(self, full_graph, cfg):
        super().__init__()

        if cfg.MODEL.PINSAGE.SCORER_BIAS:
            n_nodes = full_graph.number_of_nodes(cfg.DATASET.USER)
            self.bias = nn.Parameter(torch.zeros(n_nodes))
        else:
            self.bias = None

    def _add_bias(self, edges):
        bias_src = self.bias[edges.src[dgl.NID]]
        bias_dst = self.bias[edges.dst[dgl.NID]]
        return {'s': edges.data['s'] + bias_src + bias_dst}

    def forward(self, item_item_graph, h):
        """
        item_item_graph : graph consists of edges connecting the pairs
        h : hidden state of every node
        """
        with item_item_graph.local_scope():
            item_item_graph.ndata['h'] = h
            item_item_graph.apply_edges(fn.u_dot_v('h', 'h', 's'))
            if self.bias:
                item_item_graph.apply_edges(self._add_bias)
            pair_score = item_item_graph.edata['s'][:, 0]
        return pair_score
