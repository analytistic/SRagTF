from transformers import PreTrainedConfig

class ST_LinearConfig(PreTrainedConfig):
    model_type = "st_linear"

    def __init__(
        self,
        seq_len=32,
        patch_size=16,
        projection_dim=32,
        spatio_topk=4,
        channel_dim=10,
        pred_len=32,
        temperature=0.07,
        rec_dim=32,
        temporal_dim=32,
        temporal_ffn=64,
        temporal_proj=False,  
        act_fn="gelu",
        layers=2,
        num_heads=1,
        attn_dropout=0.0,
        attn_output_dropout=0.0,
        encode_dropout=0.0,
        ffn_1_dropout=0.2,
        ffn_2_dropout=0.2,
        rec_query_dropout=0.2,
        attn_query_dropout=0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.spatio_topk = spatio_topk
        self.channel_dim = channel_dim
        self.pred_len = pred_len
        self.temperature = temperature
        self.rec_dim = rec_dim
        self.temporal_dim = temporal_dim
        self.temporal_ffn = temporal_ffn
        self.temporal_proj = temporal_proj
        self.act_fn = act_fn
        self.layers = layers
        self.num_heads = num_heads
        # Dropout rates
        self.encode_dropout = encode_dropout
        self.attn_dropout = attn_dropout
        self.ffn_1_dropout = ffn_1_dropout
        self.ffn_2_dropout = ffn_2_dropout
        self.rec_query_dropout = rec_query_dropout
        self.attn_query_dropout = attn_query_dropout
        self.attn_output_dropout = attn_output_dropout
