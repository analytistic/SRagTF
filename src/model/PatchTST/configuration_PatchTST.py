from transformers import PreTrainedConfig
from typing import Optional, List, Dict
import torch

class PatchTSTConfig(PreTrainedConfig):
    model_type = "patchtst"

    def __init__(
        self,
        enc_in = 7,
        seq_len = 12,
        pred_len = 12,
        e_layers = 3,
        n_heads = 8,
        d_model = 128,
        d_ff = 256,
        dropout = 0.2,
        fc_dropout = 0.2,
        head_dropout = 0.0,
        individual = False,
        attn_dropout = 0.0, 
        patch_len = 3,
        stride = 1,
        padding_patch = 'end',
        revin = True,
        affine = False,
        subtract_last = False,
        decomposition = False,
        kernel_size = 25,
        activation = 'gelu',
        max_seq_len:Optional[int]=1024, 
        d_k:Optional[int]=None, 
        d_v:Optional[int]=None, 
        norm:str='BatchNorm', 
        act:str="gelu", 
        key_padding_mask:str='auto',
        padding_var:Optional[int]=None, 
        attn_mask:Optional[torch.Tensor]=None, 
        res_attention:bool=True, 
        pre_norm:bool=False, 
        store_attn:bool=False, 
        pe:str='zeros', 
        learn_pe:bool=True, 
        pretrain_head:bool=False, 
        head_type = 'flatten', 
        verbose:bool=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.enc_in = enc_in
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.e_layers = e_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.e_layers = e_layers

        self.d_ff = d_ff
        self.dropout = dropout
        self.attn_dropout = attn_dropout
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
        self.activation = activation
        self.max_seq_len = max_seq_len
        self.d_k = d_k
        self.d_v = d_v
        self.norm = norm
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




