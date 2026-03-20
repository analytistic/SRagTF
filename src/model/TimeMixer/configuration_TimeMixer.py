from transformers import PreTrainedConfig

class TimeMixerConfig(PreTrainedConfig):
    model_type = "time_mixer"

    def __init__(
        self,
        seq_len=12,
        label_len=0,
        pred_len=1,
        down_sampling_window=3,
        e_layers=1,
        moving_avg=3,
        use_future_temporal_feature=False,
        d_model=128,
        embed="fixed",
        freq="h",
        dropout=0.1,
        enc_in=100,
        c_out=100,
        d_ff=256,
        down_sampling_method='avg',
        decomp_method='moving_avg',
        down_sampling_layers=2,
        channel_independence=1,
        use_norm=1,
        task_name='long_term_forecast',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.down_sampling_window = down_sampling_window
        self.e_layers = e_layers
        self.moving_avg = moving_avg
        self.use_future_temporal_feature = use_future_temporal_feature
        self.d_model = d_model
        self.embed = embed
        self.freq = freq
        self.dropout = dropout
        self.enc_in = enc_in
        self.c_out = c_out
        self.channel_independence = channel_independence
        self.use_norm = use_norm
        self.task_name = task_name
        self.down_sampling_method = down_sampling_method
        self.decomp_method = decomp_method
        self.down_sampling_layers = down_sampling_layers
        self.d_ff = d_ff
        
