import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.local_global import Seasonal_Prediction, series_decomp_multi
from layers.Attention import Attention
from layers.Attention_channel import Attention_channel
from layers.Transformer_Encoder import EncoderLayer
import math
from layers.RevIN import RevIN


def get_frequency_modes(seq_len, modes=None, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len//2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index                        # 这里返回的应该是一个

class CEM(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, fc_dropout=0.0):
        super().__init__()

        self.attention_CEM = EncoderLayer(d_model, n_heads, dropout=dropout)

    def forward(self,x, modes=None, mode_select_method='random'):

        bsz,nvar,seq_len,d_model = x.shape      # x:[bsz, nvar, seq_len, d_model]
        x = x.permute(0, 2, 1, 3)               # x:[bsz, seq_len, nvar, d_model]
        x = torch.reshape(x,(bsz*seq_len, nvar, d_model))                   # x:[bsz*seq_len, nvar, d_model]
        x_ft = torch.fft.rfft(x, dim=1)

        index_channel = get_frequency_modes(nvar+1, modes=modes, mode_select_method=mode_select_method)

        x_ft_ = x_ft[:, index_channel, :]

        x_ft_real = x_ft_.real.clone()
        x_ft_real = self.attention_CEM(x_ft_real)
        x_ft_.real = x_ft_real

        x_ft_imag = x_ft_.imag.clone()
        x_ft_imag = self.attention_CEM(x_ft_imag)
        x_ft_.imag = x_ft_imag

        y_ft = torch.zeros_like(x_ft, device=x_ft.device, dtype=x_ft.dtype)
        y_ft[:, index_channel, :] = x_ft_
        y = torch.fft.irfft(y_ft, n=nvar, dim=1)
        y = torch.reshape(y,(bsz, seq_len, nvar, d_model))                   # x:[bsz*seq_len, nvar, d_model]
        y = y.permute(0, 2, 1, 3)

        return y 

class TEM(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, fc_dropout=0.0):
        super().__init__()

        self.attention_TEM = EncoderLayer(d_model, n_heads, dropout=dropout)

    def forward(self,x, modes=None, mode_select_method='random'):

        bsz,nvar,seq_len,d_model = x.shape                                  # x:[bsz, nvar, seq_len, d_model]
        x = torch.reshape(x,(bsz*nvar, seq_len, d_model))                   # x:[bsz*nvar, seq_len, d_model]
        x_ft = torch.fft.rfft(x, dim=1)                                     # x:[bsz*nvar, seq_len//2, d_model]

        index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        x_ft_ = x_ft[:, index, :]

        x_ft_real = x_ft_.real.clone()
        x_ft_real = self.attention_TEM(x_ft_real)
        x_ft_.real = x_ft_real

        x_ft_imag = x_ft_.imag.clone()
        x_ft_imag = self.attention_TEM(x_ft_imag)
        x_ft_.imag = x_ft_imag

        y_ft = torch.zeros_like(x_ft, device=x_ft.device, dtype=x_ft.dtype)
        y_ft[:, index, :] = x_ft_
        y = torch.fft.irfft(y_ft, n=seq_len, dim=1)
        y = torch.reshape(y,(bsz, nvar, seq_len, d_model))                   # x:[bsz*nvar, seq_len, d_model]
        return y

class Block(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, fc_dropout=0.0):
        super().__init__()
        self.cem = CEM(d_model, n_heads, dropout, fc_dropout)
        self.tem = TEM(d_model, n_heads, dropout, fc_dropout)

    def forward(self,x,modes=None, mode_select_method='random'):             # x:[bsz, nvar, seq_len, d_model]
        x = self.cem(x, modes=modes, mode_select_method=mode_select_method)             # x:[bsz, nvar, seq_len, d_model]
        x = self.tem(x, modes=modes, mode_select_method=mode_select_method)             # x:[bsz, nvar, seq_len, d_model] 
        return x

class tokenEmb(nn.Module):           
    def __init__(self, d_model):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, d_model))

    def forward(self,x):            # x:[bsz, seq_len, nvar]
        x = x.permute(0, 2, 1)      # x:[bsz, nvar, seq_len]
        x = x.unsqueeze(3)          # x:[bsz, nvar, seq_len, 1]
        # N*T*1 x 1*D = N*T*D       # Batch 一般可以不考虑，只是一个批量操作。
        x = x * self.tokens
        return x
    
class MLPRegression(nn.Module):
    def __init__(self, seq_len, hidden_size, pred_len):
        super(MLPRegression, self).__init__()
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_trend.weight = nn.Parameter(
                (1 / seq_len) * torch.ones([pred_len, seq_len]))

        # self.MLP = nn.Sequential(
        #     nn.Linear(seq_len, hidden_size, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, pred_len)
        # )

    def forward(self, x):
        return self.linear_trend(x)
    
class FlattenHead(nn.Module):
    def __init__(self, seq_len, pred_len, d_model, head_dropout = 0.0):
        super().__init__()
        self.layer1 = nn.Linear(seq_len * d_model, pred_len)
        self.dropout = nn.Dropout(head_dropout)
    def forward(self, x):                                                   # x: [bsz, nvar, seq_len, d_model]
        x = torch.reshape(x,(x.shape[0],x.shape[1],x.shape[2]*x.shape[3]))  # x: [bsz, nvar, seq_len*d_model]
        x = self.dropout(self.layer1(x))                                    # x: [bsz, nvar, pred_len]
        x = x.permute(0,2,1)                                                # x: [bsz, pred_len, nvar]      
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.nvar = configs.enc_in
        self.d_model = configs.d_model
        self.e_layers = configs.e_layers
        self.hidden_size = self.pred_len
        self.n_heads = configs.n_heads
        self.modes = configs.modes
        
        # 分解模块
        self.decompsition = series_decomp_multi([17, 49])
        # norm
        self.revin_layer = RevIN(self.nvar, affine=True, subtract_last=False)
        # 嵌入
        self.token_embedding = tokenEmb(self.d_model)
        # 模型主体模块
        self.blocks = nn.ModuleList([Block(d_model=self.d_model, n_heads=self.n_heads)
                                     for i in range(self.e_layers)])
        self.seasonal_mlp = FlattenHead(self.seq_len, self.pred_len, self.d_model, head_dropout=configs.head_dropout)
        self.trend_mlp = MLPRegression(self.seq_len, self.hidden_size, self.pred_len)

        # fusion
        self.W_fuse = torch.nn.Parameter(torch.ones(2))

    def forward(self, x):                                           # x: [bsz, seq_len, nvar]

        x = self.revin_layer(x, 'norm')

        seasonal, trend = self.decompsition(x)                      # seasonal: [bsz, seq_len, nvar]

        # 季节部分
        seasonal = self.token_embedding(seasonal)                   # seasonal: [bsz, nvar, seq_len, d_model]
        for block in self.blocks:
            seasonal = block(seasonal,self.modes)                   # seasonal: [bsz, nvar, seq_len, d_model]
        # 映射到输出长度结果
        seasonal = self.seasonal_mlp(seasonal)                      # seasonal: [bsz, seq_len, nvar]               

        # 趋势部分
        trend = self.trend_mlp(trend.permute(0,2,1)).permute(0,2,1) # trend: [bsz, seq_len, nvar]

        # # fusion部分
        # x = trend + seasonal

        # @lrq 合并趋势项与周期项
        
        fuse_list =[]
        fuse_list.append(seasonal.unsqueeze(0))
        fuse_list.append(trend.unsqueeze(0))
        fuse_matrix = torch.cat(fuse_list, dim=0)
        x = torch.einsum('k,kbtn->btn', self.W_fuse, fuse_matrix)

        # return y_output

        # 反变换
        x = self.revin_layer(x, 'denorm')    

        return x
    
        # B, T, N = x.shape
        # # @lrq trend-cyclical prediction block: regre or mean
        # if self.mode == 'regre':
        #     x, trend = self.decomp_multi(x)
        #     trend = self.regression_mlp(trend.permute(0,2,1)).permute(0, 2, 1)
        # elif self.mode == 'mean':
        #     mean = torch.mean(x, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        #     x, trend = self.decomp_multi(x)
        #     trend = torch.cat([trend[:, -self.seq_len:, :], mean], dim=1)

        # # x: [Batch, Input length, Channel] @ lrq 这里的embedding可以直接拿来用
        # B, T, N = x.shape
        # x = x.permute(0,2,1)
        # # embedding x: [B, N, T, D]
        # x = self.patch_embedding(x) # 8, 8, 96, 128

        # # print('x_embedding:', x.shape)

        # # print('----111')
        # if self.channel_independence == 1:
        #     # print("channel_independence:",self.channel_independence)
        #     for i in range(self.num_blocks):
        #         x = self.CEM(x)
        #         # print('the num of CEM blocks{}'.format(i))
        # if self.temporal_independence == 1:
        #     for i in range(self.num_blocks):
        #         x = self.TEM(x)
        #         # print('the num of TEM blocks{}'.format(i))
        
        # # x = self.linear(x)
        
        # # flatten
        # x = self.head(x)                                                # x: [bsz, pred_len, nvar]

        
        # # # 特征降维 @lrq 3.28
        # # x = self.dim_reduction(x)
        # # x = x.squeeze(-1)
        
        # # # @lrq 输出维度映射
        # # x = self.to_output(x)
        # # # [B N T] --> [B T N]


        # # x = x.permute(0, 2, 1)


        # # @lrq 合并趋势项与周期项
        
        # fuse_list =[]
        # fuse_list.append(x.unsqueeze(0))
        # fuse_list.append(trend.unsqueeze(0))
        # fuse_matrix = torch.cat(fuse_list, dim=0)
        # y_output = torch.einsum('k,kbtn->btn', self.W_fuse, fuse_matrix)

        # return y_output
