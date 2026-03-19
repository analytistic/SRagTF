from transformers import PreTrainedConfig

class iTransformerConfig(PreTrainedConfig):
    model_type = "itransformer"

    def __init__(
        self,
        seq_len=32,
        use_norm=False,
        d_model=512,
        embed='fixed',
        freq='h',
        dropout=0.1,
        activation='gelu',
        d_ff=2048,
        layers=2,
        n_heads=8,
        e_layers=2,
        output_attention=False,
        factor=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.use_norm = use_norm
        self.d_model = d_model
        self.embed = embed
        self.freq = freq
        self.dropout = dropout
        self.activation = activation
        self.d_ff = d_ff
        self.layers = layers
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.output_attention = output_attention
        self.factor = factor
