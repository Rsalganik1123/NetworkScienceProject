from torch import nn
from .layers import LinearProjector, SAGENet, ItemToItemScorer
from .build import MODEL_REGISTRY
from .loss_fn import LOSS_REGISTRY, compute_auc


class PinSAGEModel(nn.Module):
    def __init__(self, full_graph, cfg):
        super().__init__()
        self.hidden_size = cfg.MODEL.PINSAGE.HIDDEN_SIZE
        self.proj = LinearProjector(full_graph, cfg)
        self.sage = SAGENet(self.hidden_size, cfg.MODEL.PINSAGE.LAYERS)
        self.scorer = ItemToItemScorer(full_graph, cfg)
        self.loss_fn = LOSS_REGISTRY[cfg.TRAIN.LOSS]
        if cfg.MODEL.PINSAGE.REPRESENTATION_NORMALIZE:
            self.norm = nn.LayerNorm(cfg.MODEL.PINSAGE.HIDDEN_SIZE)
        else:
            self.norm = None

    def forward(self, pos_graph, neg_graph, blocks):
        h_item = self.get_repr(blocks)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        loss = self.loss_fn(pos_score, neg_score)
        auc = compute_auc(pos_score, neg_score)
        return pos_score, neg_score, loss, auc

    def get_repr(self, blocks):
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)
        h = h_item_dst + self.sage(blocks, h_item)
        if self.norm:
            h = self.norm(h)

        return h


@MODEL_REGISTRY.register('PINSAGE')
def build_pinsage_model(full_graph, cfg):
    model = PinSAGEModel(full_graph, cfg)
    return model
