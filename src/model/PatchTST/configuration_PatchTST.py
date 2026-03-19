from transformers import PreTrainedConfig
from typing import Optional, List, Dict
import torch

class PatchTSTConfig(PreTrainedConfig):
    model_type = "patchtst"

    def __init__(
        self,
        enc_in: int,
        seq_len: int,
        target_window: int = None,
        n_layers: int = 3,
        n_heads: int = 8,
        d_model: int = 128,
        d_ff: int = 256,
        dropout: float = 0.0,
        fc_dropout: float = 0.0,
        head_dropout: float = 0.0,
        individual: bool = False,
        patch_len: int = 16,
        stride: int = 8,
        padding_patch: str = 'end',     
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        decomposition: bool = False,
        kernel_size: int = 25,
        max_seq_len: int = 1024,
        d_k: int = 64,
        d_v: int = 64,
        norm: str = 'LayerNorm',
        attn_dropout: float = 0.0,
        act: str = 'gelu',
        key_padding_mask: Optional[bool] = None,
        padding_var: Optional[int] = None,
        attn_mask: Optional[torch.Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = 'zeros',
        learn_pe: bool = True,
        pretrain_head: bool = False,
        head_type: str = 'prediction',
        verbose: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = target_window if target_window is not None else seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.fc_dropout = fc_dropout
        self.head_dropout = head_dropout
        self.individual = individual
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.revin = revin
        self.affine = affine
        self.subtract_last = subtract_last
        self.decomposition = decomposition
        self.kernel_size = kernel_size
        self.max_seq_len = max_seq_len
        self.d_k = d_k
        self.d_v = d_v
        self.norm = norm
        self.attn_dropout = attn_dropout
        self.act = act
        self.key_padding_mask = key_padding_mask
        self.padding_var = padding_var
        self.attn_mask = attn_mask
        self.res_attention = res_attention
        self.pre_norm = pre_norm
        self.store_attn = store_attn
        self.pe = pe
        self.learn_pe = learn_pe
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.verbose = verbose

