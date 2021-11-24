import dgl
import torch
from torch.utils.data import IterableDataset, DataLoader
from .build import NODE_SAMPLER_REGISTRY


@NODE_SAMPLER_REGISTRY.register('DEFAULT')
class ItemToItemBatchSampler(IterableDataset):
    def __init__(self, g, cfg):
        self.g = g
        self.user_type = cfg.DATASET.USER
        self.item_type = cfg.DATASET.ITEM
        self.user_to_item_etype = list(g.metagraph()[self.user_type][self.item_type])[0]
        self.item_to_user_etype = list(g.metagraph()[self.item_type][self.user_type])[0]
        self.batch_size = cfg.DATASET.SAMPLER.NODES_SAMPLER.BATCH_SIZE

    def __iter__(self):
        while True:
            heads = torch.randint(0, self.g.number_of_nodes(self.item_type), (self.batch_size,))
            tails = dgl.sampling.random_walk(
                self.g,
                heads,
                metapath=[self.item_to_user_etype, self.user_to_item_etype])[0][:, 2]
            neg_tails = torch.randint(0, self.g.number_of_nodes(self.item_type), (self.batch_size,))

            mask = (tails != -1)
            yield heads[mask], tails[mask], neg_tails[mask]
