import torch
from torch import nn
from torch.nn import functional as F
from .configuration_ST_Linear import ST_LinearConfig
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.utils.generic import ModelOutput
from dataclasses import dataclass

@dataclass
class ST_LinearForTrafficPredictionOutput(ModelOutput):
    loss: torch.Tensor | None = None
    pred: torch.Tensor | None = None
    rec_logits: tuple[torch.Tensor, ...] | None = None
    attn: tuple[torch.Tensor, ...] | None = None

@dataclass
class ST_LinearModelOutput(ModelOutput):
    pred: torch.Tensor
    rec_logits: tuple[torch.Tensor, ...] | None = None
    attn: tuple[torch.Tensor, ...] | None = None

@dataclass
class ST_Rec_ModuleOutput(ModelOutput):
    query_x: torch.Tensor | None = None
    rec_x: torch.Tensor | None = None
    rec_logits: torch.Tensor | None = None

@dataclass
class ST_Channel_ModuleOutput(ModelOutput):
    x: torch.Tensor
    attn: torch.Tensor | None = None

@dataclass
class ST_TruncateFormerOutput(ModelOutput):
    x: torch.Tensor
    rec_logits: torch.Tensor | None = None
    attn: torch.Tensor | None = None

class ST_Rec_Module(nn.Module):
    def __init__(self, config: ST_LinearConfig):
        super().__init__()
        self.config = config
        self.topk = config.spatio_topk
        self.temperature = config.temperature
        self.spatio_proj_q = nn.Linear(config.projection_dim, config.rec_dim, bias=False)
        self.spatio_proj_k = nn.Linear(config.projection_dim, config.rec_dim, bias=False)
        # self.logits_bias = nn.Parameter(torch.zeros(config.channel_dim, config.channel_dim), requires_grad=True)
        # self.query_emb = nn.Parameter(torch.randn(config.channel_dim, config.rec_dim))
        self.rec_query_dropout = nn.Dropout(config.rec_query_dropout)


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, return_logits=False) -> ST_Rec_ModuleOutput:
        B, C, _ = q.shape
        _, _, D = k.shape
        # x_q = self.spatio_proj_q(self.rec_query_dropout(self.query_emb.unsqueeze(0).expand(B, C, -1) + q))
        x_q = self.spatio_proj_q(self.rec_query_dropout(q))
        x_k = self.spatio_proj_k(k)
        logits = torch.einsum("bqd, bkd -> bqk", x_q, x_k) / self.temperature
        if self.training:
            topk_idx = logits.topk(self.topk, dim=-1).indices
            z_hard = F.one_hot(topk_idx, num_classes=C).float() # B, C, k, C
            z_soft = (logits[:, :, None, :].expand(B, C, self.topk, C) / self.temperature).clone()
            mask = torch.cat(
                [
                    torch.zeros(B, C, 1, C, device=logits.device),
                    z_hard[:, :, :-1, :].cumsum(dim=2)
                ], dim=2
            ).bool()
            # mask_dig = torch.eye(C, device=logits.device).bool()
            # mask = mask | mask_dig.unsqueeze(0).unsqueeze(2)
            z_soft = z_soft.masked_fill(mask==True, float("-inf"))
            z_soft = F.softmax(z_soft, dim=-1)
            z = z_hard + z_soft - z_soft.detach() 
            rec_x = torch.einsum("bqkc, bcd -> bqkd", z, v) 
        else:
            topk_idx = logits.topk(self.topk, dim=-1).indices 
            x_candidates = v.unsqueeze(1).expand(B, C, C, D) 
            rec_x = torch.gather(x_candidates, dim=2, index=topk_idx.unsqueeze(-1).expand(B, C, self.topk, D))

        return ST_Rec_ModuleOutput(query_x=q, rec_x=rec_x, rec_logits=logits if return_logits else None) # B, C, k, D
    
class ST_Channel_Module(nn.Module):
    def __init__(self, config: ST_LinearConfig):
        super().__init__()
        self.config = config
        self.mha = nn.MultiheadAttention(config.projection_dim, num_heads=config.num_heads, batch_first=True, dropout=config.attn_dropout)
        self.norm = nn.LayerNorm(config.projection_dim)
        # self.query_emb = nn.Parameter(torch.randn(config.channel_dim, config.projection_dim))
        self.attn_query_dropout = nn.Dropout(config.attn_query_dropout)
        self.attn_output_dropout = nn.Dropout(config.attn_output_dropout)

    def forward(self, q, k, v, return_attn=False) -> ST_Channel_ModuleOutput:
        B, C, D = q.shape
        # k = torch.cat([q.unsqueeze(2), k], dim=2)
        # v = torch.cat([q.unsqueeze(2), v], dim=2)
        x, attn = self.mha(
            # self.attn_query_dropout((q + self.query_emb.unsqueeze(0).expand(q.shape[0], -1, -1)).reshape(B*C, 1, D)),
            self.attn_query_dropout(q.reshape(B*C, 1, D)),
            k.reshape(B*C, -1, D),
            v.reshape(B*C, -1, D),
            need_weights=return_attn
        )
        x = self.norm(self.attn_output_dropout(x.reshape(B, C, D) + q))
        return ST_Channel_ModuleOutput(
            x=x,
            attn=attn if return_attn else None
        )
    
    
class ST_Temporal_Module(nn.Module):
    def __init__(self, config: ST_LinearConfig):
        super().__init__()
        self.config = config
        self.temporal_dim = config.temporal_dim
        self.activation_fn = ACT2FN[config.act_fn]
        if config.projection_dim != config.temporal_dim or config.temporal_proj:
            self.proj_input = nn.Linear(config.projection_dim, config.temporal_dim)
            self.proj_output = nn.Linear(config.temporal_dim, config.projection_dim)
        else:
            self.proj_input = nn.Identity()
            self.proj_output = nn.Identity()
        self.ffn_1_dropout = nn.Dropout(config.ffn_1_dropout)
        self.ffn_2_dropout = nn.Dropout(config.ffn_2_dropout)
        self.mlp = nn.Sequential(
            nn.Linear(config.temporal_dim, config.temporal_ffn),
            self.activation_fn,
            self.ffn_1_dropout,
            nn.Linear(config.temporal_ffn, config.temporal_dim),
            self.ffn_2_dropout,
        )
        self.norm = nn.LayerNorm(config.temporal_dim)

    def forward(self, x: torch.Tensor):
        B, C, D = x.shape
        x = self.proj_input(x)
        x = self.norm(self.mlp(x) + x)
        x = self.proj_output(x)
        return x
    
class ST_TruncateFormer(nn.Module):
    def __init__(self, config: ST_LinearConfig):
        super().__init__()
        self.config = config
        self.spatio_rec = ST_Rec_Module(config)
        self.channel_model = ST_Channel_Module(config)
        self.Inter_model = ST_Temporal_Module(config)
        

    def forward(self, x: torch.Tensor, return_attn=False, return_logits=False):
        B, C, D = x.shape
        rec_outputs = self.spatio_rec.forward(q=x, k=x, v=x, return_logits=return_logits)
        cross_outputs = self.channel_model.forward(x, rec_outputs.rec_x, rec_outputs.rec_x, return_attn=return_attn)
        x = self.Inter_model.forward(cross_outputs.x)
        return ST_TruncateFormerOutput(x=x, 
            rec_logits=rec_outputs.rec_logits if return_logits else None,
            attn=cross_outputs.attn if return_attn else None
            )



class Flat_Prediction_Module(nn.Module):
    def __init__(self, config: ST_LinearConfig):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.temporal_dim, config.pred_len)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        return x


class RevIN(nn.Module):
    """copyed from
    https://github.com/ts-kim/RevIN
    """
    def __init__(self, num_features: int, eps=1e-5, affine=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x



class TimeEncoder(nn.Module):
    def __init__(self, config: ST_LinearConfig):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.seq_len, config.projection_dim)
        self.dropout = nn.Dropout(config.encode_dropout)

    def forward(self, x: torch.Tensor):
        B, C, D = x.shape
        x = self.linear(x)
        return self.dropout(x)

        

class ST_LinearModel(nn.Module):
    def __init__(self, config: ST_LinearConfig):
        super().__init__()
        self.config = config
        self.revin = RevIN(config.channel_dim)
        self.encoder = TimeEncoder(config)
        self.layers = nn.ModuleList([ST_TruncateFormer(config) for _ in range(config.layers)])
        self.predictor = Flat_Prediction_Module(config)


    def forward(self, input_ids: torch.Tensor, return_logits=False, return_attn=False):
        input_ids = self.revin(input_ids, mode="norm")
        rec_logits = ()
        attn = ()
        input_ids = self.encoder(input_ids.permute(0, 2, 1))
        for layer in self.layers:
            outputs = layer(input_ids, return_attn=return_attn, return_logits=return_logits)
            input_ids = outputs.x
            rec_logits = rec_logits + (outputs.rec_logits,) if return_logits else rec_logits
            attn = attn + (outputs.attn,) if return_attn else attn
        x = self.predictor(input_ids).permute(0, 2, 1)
        x = self.revin(x, mode="denorm")

        return ST_LinearModelOutput(pred=x, rec_logits=rec_logits, attn=attn)
    
class ST_Loss(nn.Module):
    def __init__(self, config: ST_LinearConfig):
        super().__init__()
        self.config = config
        self.criterion = nn.MSELoss()

    def forward(self, pred: torch.Tensor, labels: torch.Tensor):
        loss = self.criterion(pred, labels)
        return loss




class ST_LinearForTrafficPrediction(PreTrainedModel):
    config_class = ST_LinearConfig
    config: ST_LinearConfig
    def __init__(self, config: ST_LinearConfig):
        super().__init__(config=config)
        self.config = config
        self.model = ST_LinearModel(config)
        self.criterion = ST_Loss(config)
        self.post_init()

    # @torch.no_grad()
    # def _init_weights(self, module: nn.Module):
    #     super()._init_weights(module)
        
    #     if isinstance(module, ST_Rec_Module):
    #         nn.init.eye_(module.spatio_proj_q.weight)
    #         nn.init.eye_(module.spatio_proj_k.weight)
    #         if module.spatio_proj_q.bias is not None:
    #             nn.init.zeros_(module.spatio_proj_q.bias)
    #         if module.spatio_proj_k.bias is not None:
    #             nn.init.zeros_(module.spatio_proj_k.bias)

    def post_init(self):
        pass
  
        

    

        

    def forward(self, timeseries: torch.Tensor, labels: torch.Tensor | None = None, return_logits: bool = True) -> ST_LinearForTrafficPredictionOutput:

        x = self.model.forward(timeseries, return_logits=return_logits)
        loss = None
        if labels is not None:
            loss = self.criterion.forward(x.pred, labels)
        return ST_LinearForTrafficPredictionOutput(loss=loss, pred=x.pred, rec_logits=x.rec_logits, attn=x.attn)
    


 



