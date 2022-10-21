from .imports import * 
from .device import * 

from basic_util import * 

__all__ = [
    'HeteroGraphDataset',
]


class HeteroGraphDataset:
    def __init__(self,
                 hg: dgl.DGLHeteroGraph,
                 use_gpu: bool = True,
                 mute: bool = False):
        self.device = auto_select_gpu(use_gpu=use_gpu)
        self.hg = hg.to(self.device)
        
        self.ntypes = set(self.hg.ntypes)
        self.etypes = set(self.hg.canonical_etypes)
        self.feat_dict = dict(self.hg.ndata['feat'])
        self.feat_dim_dict = { ntype: self.feat_dict[ntype].shape[-1] for ntype in self.feat_dict }
        self.infer_ntype = next(iter(self.hg.ndata['label']))
        self.label = self.hg.nodes[self.infer_ntype].data['label']
        self.train_mask = self.hg.nodes[self.infer_ntype].data.get('train_mask')
        self.val_mask = self.hg.nodes[self.infer_ntype].data.get('val_mask')
        self.test_mask = self.hg.nodes[self.infer_ntype].data.get('test_mask')
        
        if self.label.ndim == 1:
            self.num_classes = len(self.label.unique())
        else:
            raise AssertionError 

        if not mute:
            init_log()            
            
            log_info("[HeteroGraph Summary]")
            log_info(str(self.hg))
            log_info(f"[ntypes] {self.ntypes}")
            log_info(f"[infer_ntype] {self.infer_ntype}")
            log_info(f"[num_classes] {self.num_classes}")
