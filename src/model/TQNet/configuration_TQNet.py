from transformers import PreTrainedConfig

class TQNetConfig(PreTrainedConfig):
    model_type = "time_mixer"

    def __init__(
        self,
        seq_len=12,
        pred_len=1,
        enc_in = 7,
        cycle_len=24,
        model_type="TQNet",
        d_model=128,
        dropout=0.1,
        use_revin=True,
        **kwargs
    ):
        super().__init__(**kwargs)
   

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.cycle_len = cycle_len
        self.model_type = model_type
        self.d_model = d_model
        self.dropout = dropout
        self.use_revin = use_revin
        
        