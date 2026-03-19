import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from .configuration_DP_LET import DP_LETConfig

from torch.nn.utils import weight_norm

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, num_hidden, num_heads, attn_drop_rate=0., res_attn_score=False):
        super(ScaledDotProductAttention, self).__init__()
        self.attn_dropout = nn.Dropout(p=attn_drop_rate)
        self.res_attn_score = res_attn_score
        num_hidden_per_head = num_hidden // num_heads
        self.scale = nn.Parameter(torch.tensor(num_hidden_per_head ** -0.5), requires_grad=False)

    def forward(self, query, key, value, pre_softmax_attn_score=None, key_mask=None, attn_mask=None):
        """
        query: [batch, num_heads, query_len, num_hidden]
        key: [batch, num_heads, num_hidden, key_len]
        value: [batch, num_heads, key_len, num_hidden]
        pre_softmax_attn_score: [batch, num_head, query_len, key_len]
        key_mask: [batch, key_len]
        attn_mask: [1, query_len, query_len]
        """
        attn_score = torch.matmul(query, key) * self.scale
        if pre_softmax_attn_score is not None:
            attn_score = attn_score + pre_softmax_attn_score
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_score.masked_fill_(attn_mask, -np.inf)
            else:
                attn_score += attn_mask

        if key_mask is not None:
            attn_score.masked_fill_(key_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        attn_weights = F.softmax(attn_score, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, value)

        if self.res_attn_score:
            return output, attn_weights, attn_score
        else:
            return output, attn_weights


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """
    def __init__(self, num_hidden, num_heads, num_hidden_key_per_head=None, num_hidden_value_per_head=None,
                 res_attn_score=False, attn_drop_rate=0., proj_drop_rate=0.):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hidden_key_per_head = num_hidden // num_heads if num_hidden_key_per_head == '' else \
            num_hidden_key_per_head
        self.num_hidden_value_per_head = num_hidden // num_heads if num_hidden_value_per_head == '' \
            else num_hidden_value_per_head

        self.linear_q = nn.Linear(num_hidden, num_heads * self.num_hidden_key_per_head)
        self.linear_k = nn.Linear(num_hidden, num_heads * self.num_hidden_key_per_head)
        self.linear_v = nn.Linear(num_hidden, num_heads * self.num_hidden_value_per_head)

        self.res_attn_score = res_attn_score
        self.linear = nn.Linear(self.num_hidden_value_per_head * num_heads, num_hidden)

        self.attention = ScaledDotProductAttention(num_hidden=num_hidden, num_heads=num_heads,
                                                   attn_drop_rate=attn_drop_rate, res_attn_score=res_attn_score)
        self.dropout = nn.Dropout(proj_drop_rate)

    def forward(self, query, key, value, pre_softmax_attn_score=None, key_mask=None, attn_mask=None):
        batch_mul_nodes = query.shape[0]
        if key is None:
            key = query
        if value is None:
            value = query
        # q_s    : [bs x n_heads x max_q_len x d_k]
        # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        # v_s    : [bs x n_heads x q_len x d_v]
        query = self.linear_q(query).view(batch_mul_nodes, -1, self.num_heads, self.num_hidden_key_per_head).transpose(1, 2)
        key = self.linear_k(key).view(batch_mul_nodes, -1, self.num_heads, self.num_hidden_key_per_head).permute(0, 2, 3, 1)
        value = self.linear_v(value).view(batch_mul_nodes, -1, self.num_heads, self.num_hidden_value_per_head).transpose(1, 2)

        if self.res_attn_score:
            output, attn_weights, attn_scores = self.attention(query, key, value, pre_softmax_attn_score, key_mask,
                                                               attn_mask)
        else:
            output, attn_weights = self.attention(query, key, value, key_mask=key_mask, attn_mask=attn_mask)

        output = output.transpose(1, 2).contiguous().view(batch_mul_nodes, -1, self.num_heads * self.num_hidden_value_per_head)
        output = self.dropout(self.linear(output))

        if self.res_attn_score:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights

def positional_encoding(seq_len, num_hidden):
    """
    :param seq_len: = num_patches
    :param num_hidden:  = patch_lens->num_hidden
    :return:
    """
    W_pos = torch.empty(size=(seq_len, num_hidden))
    nn.init.uniform_(W_pos, -0.02, 0.02)
    return nn.Parameter(W_pos, requires_grad=True)

class TSTEncoderLayer(nn.Module):
    def __init__(self, num_hidden, num_heads, num_hidden_key_per_head=None, num_hidden_value_per_head=None,
                 num_hidden_ff=256, attn_drop_rate=0., drop_rate=0., res_attn_scores=False):
        super(TSTEncoderLayer, self).__init__()
        num_hidden_key_per_head = num_hidden // num_heads if num_hidden_key_per_head is None else \
            num_hidden_key_per_head
        num_hidden_value_per_head = num_hidden // num_heads if num_hidden_value_per_head is None else \
            num_hidden_value_per_head
        self.res_attn_scores = res_attn_scores
        self.attn = MultiHeadAttention(num_hidden, num_heads, num_hidden_key_per_head, num_hidden_value_per_head,
                                       res_attn_scores, attn_drop_rate=attn_drop_rate, proj_drop_rate=drop_rate)

        self.dropout_attn = nn.Dropout(drop_rate)
        self.norm_attn = nn.BatchNorm1d(num_hidden)

        self.linear1 = nn.Linear(num_hidden, num_hidden_ff)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(num_hidden_ff, num_hidden)
        self.dropout = nn.Dropout(drop_rate)

        self.dropout_ffn = nn.Dropout(drop_rate)
        self.norm_ffn = nn.BatchNorm1d(num_hidden)

    def forward(self, x, pre_softmax_attn_scores=None, key_mask=None, attn_mask=None):
        # x.shape [batch*num_nodes, in_lens, num_hidden]
        if self.res_attn_scores:
            x_out, attn_weights, attn_scores = self.attn(x, x, x, pre_softmax_attn_scores, key_mask, attn_mask)
        else:
            x_out, attn_weights = self.attn(x, x, x, key_mask, attn_mask)

        x = x + self.dropout_attn(x_out)

        x = x.transpose(1, 2)
        x = self.norm_attn(x)
        x = x.transpose(1, 2)

        x_out = self.activation(self.linear1(x))
        x_out = self.dropout(x_out)
        x_out = self.linear2(x_out)
        x = x + self.dropout_ffn(x_out)

        x = x.transpose(1, 2)
        x = self.norm_ffn(x)
        x = x.transpose(1, 2)

        if self.res_attn_scores:
            return x, attn_scores
        else:
            return x


class TSTEncoder(nn.Module):
    def __init__(self, num_hidden, num_heads, num_hidden_key_per_head=None, num_hidden_value_per_head=None,
                 num_hidden_ff=None, attn_drop_rate=0., drop_rate=0., res_attn_scores=False, num_layers=1):
        super(TSTEncoder, self).__init__()

        self.encoder_layers = nn.ModuleList([TSTEncoderLayer(num_hidden, num_heads, num_hidden_key_per_head,
                                                             num_hidden_value_per_head, num_hidden_ff,
                                                             attn_drop_rate, drop_rate,
                                                             res_attn_scores=res_attn_scores) for _ in range(num_layers)])
        self.res_attn_scores = res_attn_scores

    def forward(self, x, key_mask=None, attn_mask=None):
        output = x
        scores = None
        if self.res_attn_scores:
            for encoder_layer in self.encoder_layers:
                output, scores = encoder_layer(output, pre_softmax_attn_scores=scores, key_mask=key_mask,
                                               attn_mask=attn_mask)
            return output
        else:
            for encoder_layer in self.encoder_layers:
                output = encoder_layer(output, key_mask=key_mask, attn_mask=attn_mask)
            return output


class TSTiEncoder(nn.Module):
    def __init__(self, num_patches, patch_lens, num_layers=3, num_hidden=128, num_heads=16, num_hidden_key_per_head=None,
                 num_hidden_value_per_head=None, num_hidden_ff=256, attn_drop_rate=0.0, drop_rate=0.0,
                 res_attn_scores=False):
        """
        num_patches = N
        patch_length = P
        """
        super(TSTiEncoder, self).__init__()

        self.num_patches = num_patches
        self.patch_lens = patch_lens
        
        self.parallel_embedding = LELayer(patch_len=patch_lens, model_dim=num_hidden, 
                                                         tcn_channels=[128, 256, num_hidden])

        # Positional Encoding
        self.pos_embedding = positional_encoding(num_patches, num_hidden)
        self.dropout = nn.Dropout(drop_rate)

        self.encoder = TSTEncoder(num_hidden, num_heads, num_hidden_key_per_head, num_hidden_value_per_head,
                                  num_hidden_ff, attn_drop_rate, drop_rate, res_attn_scores, num_layers)

    def forward(self, x):
        # x.shape [batch, nodes, num_patches, patch_lens]
        num_nodes = x.shape[1]

        x = self.parallel_embedding(x)  # 输出维度 [batch, nodes, num_patches, num_hidden]
        
        # Reshape for positional embedding
        y = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
        y = self.dropout(y + self.pos_embedding)

        output = self.encoder(y)
        output = torch.reshape(output, (-1, num_nodes, output.shape[-2], output.shape[-1]))
        # output.shape [batch, num_nodes, num_patches, num_hidden]
        return output


class Flatten_Head(nn.Module):
    def __init__(self, num_hidden_flatten, out_lens, drop_rate=0.):
        super(Flatten_Head, self).__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(num_hidden_flatten, out_lens)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        # x.shape [batch, nodes, num_patches, patch_lens]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x



# Rev-in
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
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
    



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        chomp1 = Chomp1d(padding)
        relu1 = nn.ReLU()
        dropout1 = nn.Dropout(dropout)

        conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        chomp2 = Chomp1d(padding)
        relu2 = nn.ReLU()
        dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(conv1, chomp1, relu1, dropout1,
                                 conv2, chomp2, relu2, dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.net[0].weight.data.normal_(0, 0.01)
        self.net[4].weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class LELayer(nn.Module):
    def __init__(self, patch_len, model_dim, tcn_channels):
        super(LELayer, self).__init__()
        
        self.DenseLayer1 = nn.Linear(patch_len, model_dim)
        
        self.DenseLayer2 = nn.Linear(model_dim, model_dim)

        self.tcn = TemporalConvNet(num_inputs=model_dim, num_channels=tcn_channels, kernel_size=2)

    def forward(self, x):

        main_branch = self.DenseLayer1(x)  # (batch_size, num_nodes, num_patches, model_dim)
        

        parallel_branch = self.DenseLayer1(x)  # (batch_size, num_nodes, num_patches, model_dim)
        
        parallel_branch = parallel_branch.permute(0, 3, 1, 2)  # (batch_size, model_dim, num_nodes, num_patches)
        parallel_branch = parallel_branch.reshape(parallel_branch.size(0), parallel_branch.size(1), -1)  # (batch_size, model_dim, seq_len)

        parallel_branch = self.tcn(parallel_branch)             
        parallel_branch = parallel_branch.permute(0, 2, 1)     
        parallel_branch = self.DenseLayer2(parallel_branch)
        parallel_branch = parallel_branch.permute(0, 2, 1)     
        parallel_branch = self.tcn(parallel_branch)          
        parallel_branch = parallel_branch.permute(0, 2, 1)      


        parallel_branch = parallel_branch.reshape(x.size(0), x.size(1), x.size(2), -1) 
        

        output = main_branch + parallel_branch  # (batch_size, num_nodes, num_patches, model_dim)

        return output
    

def TSVDR(x, cut):

    x_ = x.clone().detach()
    U, S, V = torch.svd(x)
    S[:, cut:] = 0 
    return U @ torch.diag(S[0, :]) @ V



class DP_LET_Loss(nn.Module):
    def __init__(self, config: DP_LETConfig):
        super().__init__()
        self.config = config
        self.criterion = nn.MSELoss()

    def forward(self, pred: torch.Tensor, labels: torch.Tensor):
        loss = self.criterion(pred, labels)
        return loss

from transformers import PreTrainedModel
from transformers.utils.generic import ModelOutput
from dataclasses import dataclass

@dataclass
class DP_LET_PredictorOutput(ModelOutput):
    loss: torch.Tensor | None = None
    pred: torch.Tensor | None = None


class DP_LET_Predictor(PreTrainedModel):
    def __init__(self, configs: DP_LETConfig):
        super(DP_LET_Predictor, self).__init__(configs)

        num_nodes = configs.num_nodes
        in_lens = configs.seq_len
        out_lens = configs.pred_len

        num_layers = configs.num_layers
        num_heads = configs.num_heads
        num_hidden = configs.num_hidden
        num_hidden_key_per_head = configs.num_hidden_key_per_head
        num_hidden_value_per_head = configs.num_hidden_value_per_head
        num_hidden_ff = configs.num_hidden_ff
        drop_rate = configs.drop_rate
        attn_drop_rate = configs.attn_drop_rate
        flatten_drop_rate = configs.flatten_drop_rate
        res_attn_scores = configs.res_attn_scores
        self.patch_lens = configs.patch_lens
        self.stride = configs.stride

        num_patches = int((in_lens - self.patch_lens) / self.stride + 1)
        num_patches += 1

        self.revin_layer = RevIN(num_nodes)

        self.if_revin = configs.if_revin
        self.if_decompose = configs.if_decompose
        self.if_denoise = configs.if_denoise
        self.svd_cut = configs.svd_cut  # Default number of singular values to retain


        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.model = TSTiEncoder(num_patches, self.patch_lens, num_layers, num_hidden, num_heads,
                                    num_hidden_key_per_head, num_hidden_value_per_head, num_hidden_ff,
                                    attn_drop_rate, drop_rate, res_attn_scores)
        num_hidden_flatten = num_patches * num_hidden
        self.linear_head = Flatten_Head(num_hidden_flatten, out_lens, flatten_drop_rate)
        self.criterion = DP_LET_Loss(configs)

    def forward(self, timeseries, labels): 
        if self.if_denoise:
            timeseries = TSVDR(timeseries, self.svd_cut) 

        if self.if_revin:
            timeseries = self.revin_layer(timeseries, 'norm')
            
        x = timeseries.permute(0, 2, 1)
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_lens, step=self.stride)

        # Encoder
        x = self.model(x)

        # Prediction Head
        x = self.linear_head(x)

        x = x.permute(0, 2, 1)
        if self.if_revin:
            x = self.revin_layer(x, 'denorm')

        loss = None

        if labels is not None:
            loss = self.criterion(x, labels)
        
        return DP_LET_PredictorOutput(
            loss=loss,
            pred=x
        )
    