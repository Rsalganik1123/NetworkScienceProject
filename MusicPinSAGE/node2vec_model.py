import torch.nn as nn
import pickle 
from torch.utils.data import IterableDataset, DataLoader
import torch
import dgl
from dgl.sampling import node2vec_random_walk
class Node2vec(nn.Module):

    def __init__(self, g, embedding_dim, walk_length, p, q, num_walks=10, window_size=5, num_negatives=5,
                 use_sparse=True, weight_name=None):
        super(Node2vec, self).__init__()

        assert walk_length >= window_size

        self.g = g
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.window_size = window_size
        self.num_negatives = num_negatives
        self.N = self.g.num_nodes()
        if weight_name is not None:
            self.prob = weight_name
        else:
            self.prob = None

        self.embedding = nn.Embedding(self.N, embedding_dim, sparse=use_sparse)

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def sample(self, batch):
       
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        batch = batch.repeat(self.num_walks)
        # positive
        pos_traces = node2vec_random_walk(self.g, batch, self.p, self.q, self.walk_length, self.prob) 
        pos_traces = pos_traces.unfold(1, self.window_size, 1)  # rolling window
        pos_traces = pos_traces.contiguous().view(-1, self.window_size)

        # negative
        neg_batch = batch.repeat(self.num_negatives)
        neg_traces = torch.randint(self.N, (neg_batch.size(0), self.walk_length))
        neg_traces = torch.cat([neg_batch.view(-1, 1), neg_traces], dim=-1)
        neg_traces = neg_traces.unfold(1, self.window_size, 1)  # rolling window
        neg_traces = neg_traces.contiguous().view(-1, self.window_size)

        return pos_traces, neg_traces

    def loader(self, batch_size):
    
        return DataLoader(torch.arange(self.N), batch_size=batch_size, shuffle=True, collate_fn=self.sample)

    def forward(self, nodes=None):
        
        emb = self.embedding.weight
        if nodes is None:
            return emb
        else:
            return emb[nodes]

    def loss(self, pos_trace, neg_trace):
        
        e = 1e-15

        # Positive
        pos_start, pos_rest = pos_trace[:, 0], pos_trace[:, 1:].contiguous()  # start node and following trace
        w_start = self.embedding(pos_start).unsqueeze(dim=1)
        w_rest = self.embedding(pos_rest)
        pos_out = (w_start * w_rest).sum(dim=-1).view(-1)

        # Negative
        neg_start, neg_rest = neg_trace[:, 0], neg_trace[:, 1:].contiguous()

        w_start = self.embedding(neg_start).unsqueeze(dim=1)
        w_rest = self.embedding(neg_rest)
        neg_out = (w_start * w_rest).sum(dim=-1).view(-1)

        # compute loss
        pos_loss = -torch.log(torch.sigmoid(pos_out) + e).mean()
        neg_loss = -torch.log(1 - torch.sigmoid(neg_out) + e).mean()

        return pos_loss + neg_loss


class Node2vecModel(object):
    
    def __init__(self, g, embedding_dim, walk_length, p=1.0, q=1.0, num_walks=1, window_size=5,
                 num_negatives=5, use_sparse=True, weight_name=None, eval_set=None, eval_steps=-1, device='cpu'):

        self.model = Node2vec(g, embedding_dim, walk_length, p, q, num_walks,
                              window_size, num_negatives, use_sparse, weight_name)
        self.g = g
        self.use_sparse = use_sparse
        self.eval_steps = eval_steps
        self.eval_set = eval_set

        if device == 'cpu':
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _train_step(self, model, loader, optimizer, device):
        model.train()
        total_loss = 0
        for pos_traces, neg_traces in loader:
            pos_traces, neg_traces = pos_traces.to(device), neg_traces.to(device)
            optimizer.zero_grad()
            loss = model.loss(pos_traces, neg_traces)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def train(self, epochs, batch_size, learning_rate=0.01):
        
        self.model = self.model.to(self.device)
        loader = self.model.loader(batch_size)
        if self.use_sparse:
            optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=learning_rate)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        for i in range(epochs):
            loss = self._train_step(self.model, loader, optimizer, self.device)
            if self.eval_steps > 0:
                if epochs % self.eval_steps == 0:
                    acc = self._evaluate_step()
                    print("Epoch: {}, Train Loss: {:.4f}, Val Acc: {:.4f}".format(i, loss, acc))

    def embedding(self, nodes=None):
    
        return self.model(nodes)


def train_node2vec(graph, eval_set, args):
    """
    Train node2vec model
    """
    trainer = Node2vecModel(graph,
                            embedding_dim=128,
                            walk_length=6,
                            p=1.0,
                            q=1.0,
                            num_walks=1,
                            eval_set=eval_set,
                            eval_steps=1,
                            device="cpu")

    trainer.train(epochs=10, batch_size=32, learning_rate=0.01)

print("***LOADING DATA***") 
directory = '/home/mila/r/rebecca.salganik/'
with open(directory + 'dataset_without_im.pkl', 'rb') as f:
        dataset = pickle.load(f)
print("***LOADED DATA***") 

g = dataset['full-graph']
# val_matrix = dataset['val-matrix'].tocsr()
# test_matrix = dataset['test-matrix'].tocsr()
# item_texts = dataset['item-texts']
# user_ntype = dataset['user-type']
# item_ntype = dataset['item-type']
# user_to_item_etype = dataset['user-to-item-type']

g.nodes['playlist'].data['id'] = torch.arange(g.number_of_nodes('playlist'))
g.nodes['track'].data['id'] = torch.arange(g.number_of_nodes('track'))
h_graph = dgl.to_homogeneous(g, store_type=False) 
print(h_graph)
train_node2vec(h_graph, None, None) 

