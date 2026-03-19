from transformers import PreTrainedConfig

class DP_LETConfig(PreTrainedConfig):
    model_type = "dp_let"

    def __init__(
        self,
        seq_len=32,
        patch_size=16,
        drop_rate=0.1,
        num_hidden_ff=64,
        num_nodes=64,
        pred_len=32,
        num_hidden=64,
        num_hidden_key_per_head=16,
        num_hidden_value_per_head=16,
        num_layers=2,
        num_heads=4,
        attn_drop_rate=0.1,
        flatten_drop_rate=0.1,
        res_attn_scores=False,
        patch_lens=4,
        stride=4,
        if_revin=True,
        if_decompose=False,
        if_denoise=False,
        svd_cut=31,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.drop_rate = drop_rate
        self.num_hidden_ff = num_hidden_ff
        self.num_nodes = num_nodes
        self.pred_len = pred_len
        self.num_hidden = num_hidden
        self.num_hidden_key_per_head = num_hidden_key_per_head
        self.num_hidden_value_per_head = num_hidden_value_per_head
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.attn_drop_rate = attn_drop_rate
        self.flatten_drop_rate = flatten_drop_rate
        self.res_attn_scores = res_attn_scores
        self.patch_lens = patch_lens
        self.stride = stride
        self.if_revin = if_revin
        self.if_decompose = if_decompose
        self.if_denoise = if_denoise
        self.svd_cut = svd_cut