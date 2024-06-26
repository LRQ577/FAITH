import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.local_global import Seasonal_Prediction, series_decomp_multi
from layers.Attention import Attention
from layers.Attention_channel import Attention_channel
import math


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
    return index


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.embed_size = configs.d_model #embed_size
        self.hidden_size = 256 #hidden_size
        self.pre_length = configs.pred_len
        self.feature_size = configs.enc_in #channels
        self.seq_length = configs.seq_len
        self.channel_independence = configs.channel_independence
        self.temporal_independence = configs.temporal_independence
        self.sparsity_threshold = 0.01
        self.scale = 0.02


        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.dim_reduction = nn.Linear(self.embed_size,1)
        # num_layers num_layers
        self.num_layers = configs.num_layers
        self.num_blocks = configs.num_blocks


        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.c_out = configs.c_out #int 7 
        self.decomp_kernel = configs.decomp_kernel
        self.mode = configs.mode
        # @lrq定义多尺度分解用到的函数
        self.decomp_multi = series_decomp_multi(configs.decomp_kernel)
        self.regression = nn.Linear(configs.seq_len, configs.pred_len)
        self.regression.weight = nn.Parameter((1/configs.pred_len) * torch.ones([configs.pred_len, configs.seq_len]), requires_grad=True)

        # @lrq FEM
        self.modes = configs.modes
        # print('-------------modes={}'.format(self.modes))
        self.mode_select_method = configs.mode_select
        self.index = get_frequency_modes(self.seq_len, modes=self.modes, mode_select_method=self.mode_select_method)
        print('modes={}, index={}'.format(self.modes, self.index))

        self.index_channel = get_frequency_modes(self.c_out+1, modes=self.modes, mode_select_method=self.mode_select_method)
        print('modes={}, index_channel={}'.format(self.modes, self.index_channel))
        
        
        # 实例化 CEM 
        self.n_heads = configs.n_heads
        self.CEMs = nn.ModuleList(CEM(self.embed_size, self.num_layers, self.n_heads,self.index_channel, self.seq_len) for i in range(self.num_blocks))
        self.TEMs = nn.ModuleList(TEM(self.embed_size,self.num_layers,self.n_heads,self.index, self.c_out) for i in range(self.num_blocks))
        # # 实例化attention
        # self.n_heads = configs.n_heads
        # self.attention_TEM = Attention(self.embed_size, self.n_heads,self.index, self.c_out)
        # self.attention_CEM = Attention_channel(self.embed_size, self.n_heads,self.index_channel, self.seq_len)

        # @lrq 趋势项
        self.regression_mlp = MLPRegression(configs.seq_len, self.hidden_size, configs.pred_len) 

        # @lrq 映射输出维度
        self.to_output = nn.Linear(self.seq_len, self.pred_len)       

        self.fc = nn.Sequential(
            nn.Linear(self.seq_length * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )
        # @lrq合并用的函数
        self.W_fuse = torch.nn.Parameter(torch.ones(2))

        self.linear = nn.Linear(self.embed_size, self.embed_size)

        # window embedding
        if configs.individual:
            self.individual = True
        else:
            self.individual = False
        # window embedding 
        self.patch_embedding = PatchEmbed(nvar=configs.enc_in,d_model=configs.d_model,w_sizes=configs.w_size, individual=self.individual)
        # embed
        # self.patch_embedding = Embed(nvar=configs.enc_in,d_model=configs.d_model,w_sizes=configs.w_size, individual=self.individual)
        
        self.task_name = configs.task_name

        # flatten head 
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(seq_len=self.seq_len, pred_len=configs.pred_len,
                                    d_model=configs.d_model, task_name=self.task_name, nvar=self.feature_size)
            # self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
            # self.linear_trend.weight = nn.Parameter(
            #     (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        


    # dimension extension 这里其实比较怀疑，挤出一个维度，然后做embedding，映射，真的有效果吗？这里的embedding也不是做卷积，肯定算升维，算embedding吗
    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D   # Batch 一般可以不考虑，只是一个批量操作。
        y = self.embeddings
        return x * y

    



    def forward(self, x):

        B, T, N = x.shape
        # @lrq trend-cyclical prediction block: regre or mean
        if self.mode == 'regre':
            x, trend = self.decomp_multi(x)
            trend = self.regression_mlp(trend.permute(0,2,1)).permute(0, 2, 1)
        elif self.mode == 'mean':
            mean = torch.mean(x, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
            x, trend = self.decomp_multi(x)
            trend = torch.cat([trend[:, -self.seq_len:, :], mean], dim=1)

        # x: [Batch, Input length, Channel] @ lrq 这里的embedding可以直接拿来用
        B, T, N = x.shape
        x = x.permute(0,2,1)
        # embedding x: [B, N, T, D]
        x = self.patch_embedding(x) # 8, 8, 96, 128

        # print('x_embedding:', x.shape)

        # print('----111')
        if self.channel_independence == 1:
            # print("channel_independence:",self.channel_independence)
            for CEM in self.CEMs:
                x = CEM(x)
                # print('the num of CEM blocks{}'.format(i))
        if self.temporal_independence == 1:
            for TEM in self.TEMs:
                x = TEM(x)
                # print('the num of TEM blocks{}'.format(i))
        
        # x = self.linear(x)
        
        # flatten
        x = self.head(x)                                                # x: [bsz, pred_len, nvar]

        
        # # 特征降维 @lrq 3.28
        # x = self.dim_reduction(x)
        # x = x.squeeze(-1)
        
        # # @lrq 输出维度映射
        # x = self.to_output(x)
        # # [B N T] --> [B T N]


        # x = x.permute(0, 2, 1)


        # @lrq 合并趋势项与周期项
        
        fuse_list =[]
        fuse_list.append(x.unsqueeze(0))
        fuse_list.append(trend.unsqueeze(0))
        fuse_matrix = torch.cat(fuse_list, dim=0)
        y_output = torch.einsum('k,kbtn->btn', self.W_fuse, fuse_matrix)

        return y_output

        # @lrq 
class MLPRegression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPRegression, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.layer3(x)
        return x
    

# window embedding
class PatchEmbed(nn.Module):
    def __init__(self, nvar=1, d_model=8, w_sizes=8, seq_len=336, individual=False):
        super().__init__()
        
        self.window = w_sizes
        self.context_len = 2 * self.window + 1
        self.individual = individual
        self.d_model = d_model

        if individual:
            # 每个变量一个线性层，手动初始化权重和偏置
            self.W_Ps = nn.Parameter(torch.Tensor(nvar, self.context_len, d_model))
            self.biases = nn.Parameter(torch.Tensor(nvar, d_model))
            k = 1.0 / math.sqrt(self.context_len)
            nn.init.uniform_(self.W_Ps, -k, k)
            nn.init.uniform_(self.biases, -k, k)
        else:
            # 所有变量共享一个线性层
            self.W_P = nn.Linear(self.context_len, d_model, bias=True)

    def forward(self, x):                                           # x: [bsz, nvar, seq_len]
        bsz, nvar, seq_len = x.shape

        # 提取首尾边缘值并连接到原始张量的两端
        left_edge = x[:, :, :1].expand(-1, -1, self.window)
        right_edge = x[:, :, -1:].expand(-1, -1, self.window)
        x = torch.cat([left_edge, x, right_edge], dim=-1)           # x: [bsz, nvar, seq_len + 2 * window]

        # 按窗口大小展开
        x = x.unfold(dimension=-1, size=self.context_len, step=1)   # x: [bsz, nvar, seq_len, context_len]

        if self.individual:
            # 对每个变量应用独立的线性层，使用 einsum 模拟
            x = torch.einsum('bnij,njk->bnik', x, self.W_Ps)        # x: [bsz, nvar, seq_len, d_model]
            x = x + self.biases.view(1, nvar, 1, self.d_model)      # 加上偏置
        else:
            # 所有变量共享同一个线性层
            x = self.W_P(x)                                         # x: [bsz, nvar, seq_len, d_model]

        return x  # x: [bsz, nvar, seq_len, d_model]
    
# 普通嵌入，直接把每个时间点嵌入到高维
class Embed(nn.Module):
    def __init__(self, nvar=1, d_model=8, w_sizes=8, seq_len=336, individual=True):
        super().__init__()
        
        self.window = w_sizes
        self.context_len = 2 * self.window + 1
        self.individual = individual
        self.d_model = d_model

        if individual:
            # 每个变量一个线性层，手动初始化权重和偏置
            self.W_Ps = nn.Parameter(torch.Tensor(nvar, 1, d_model))
            self.biases = nn.Parameter(torch.Tensor(nvar, d_model))
            k = 1.0 
            nn.init.uniform_(self.W_Ps, -k, k)
            nn.init.uniform_(self.biases, -k, k)
        else:
            # 所有变量共享一个线性层
            self.W_P = nn.Linear(1, d_model, bias=False)

    def forward(self, x):                                           # x: [bsz, nvar, seq_len]
        bsz, nvar, seq_len = x.shape
        
        x = x.unsqueeze(-1)                                         # x: [bsz, nvar, seq_len, 1]
        if self.individual:
            # 对每个变量应用独立的线性层，使用 einsum 模拟
            x = torch.einsum('bnij,njk->bnik', x, self.W_Ps)        # x: [bsz, nvar, seq_len, d_model]
            # x = x + self.biases.view(1, nvar, 1, self.d_model)      # 加上偏置
        else:
            # 所有变量共享同一个线性层
            x = self.W_P(x)                                         # x: [bsz, nvar, seq_len, d_model]

        return x  # x: [bsz, nvar, seq_len, d_model]  
    
   # 不同的任务有不同的头 目前只考虑预测任务
class FlattenHead(nn.Module):
    def __init__(self, seq_len, pred_len, d_model, task_name, head_dropout = 0.1,individual=False,num_class=None, nvar=None):
        super().__init__()
        self.task_name = task_name
        self.individual = individual
        if task_name == 'long_term_forecast' or task_name == 'short_term_forecast' or\
        task_name == 'imputation' or task_name == 'anomaly_detection':
            
            if self.individual:
                # 每个变量一个线性层，手动初始化权重和偏置
                self.weight = nn.Parameter(torch.Tensor(nvar, seq_len * d_model, pred_len))
                self.bias = nn.Parameter(torch.Tensor(nvar, pred_len))
                k = 1.0 / math.sqrt(seq_len * d_model)
                nn.init.uniform_(self.weight, -k, k)
                nn.init.uniform_(self.bias, -k, k)
                self.dropout = nn.Dropout(head_dropout)
            else:
                # 所有变量共享一个线性层
                self.layer1 = nn.Linear(seq_len * d_model, pred_len)
                self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):                                                           # x: [bsz, nvar, seq_len, d_model]
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast' or \
        self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            
            if self.individual:
                x = torch.reshape(x,(x.shape[0],x.shape[1],x.shape[2]*x.shape[3]))  # x: [bsz, nvar, seq_len*d_model]
                x = torch.einsum('bni,nij->bnj', x, self.weight) + self.bias        # x: [bsz, nvar, pred_len]
                x = self.dropout(x)
                x = x.permute(0,2,1)                                                # x: [bsz, pred_len, nvar]
            
            else:
                x = torch.reshape(x,(x.shape[0],x.shape[1],x.shape[2]*x.shape[3]))  # x: [bsz, nvar, seq_len*d_model]
                x = self.dropout(self.layer1(x))                                    # x: [bsz, nvar, pred_len]
                x = x.permute(0,2,1)                                                # x: [bsz, pred_len, nvar]                                                                            
        return x 
    

class CEM(nn.Module):
    def __init__(self, d_model, num_layers, n_heads, index_channel,seq_len):
        super().__init__()
        self.num_layers = num_layers
        # self.attention_type = attention_type
        self.embed_size = d_model
        self.n_heads = n_heads
        self.index_channel = index_channel
        self.seq_len = seq_len
        self.attention_CEMs = nn.ModuleList(Attention_channel(self.embed_size, self.n_heads,self.index_channel, self.seq_len) for a in range(num_layers))


    def forward(self,x):
        # [B N T D]
        B,N,T,D = x.shape
        x = x.permute(0, 2, 1, 3)
        x_ft = torch.fft.rfft(x, dim=2)
        # print('x_ft', x_ft.shape)
        x_ft_ = torch.zeros([B, T, len(self.index_channel), D], device=x.device, dtype=x_ft.dtype)
        for i, j in enumerate(self.index_channel):
            x_ft_[ :, :, i,: ] = x_ft[ :, :, j,: ]
        # print('x_ft_', x_ft_.shape)
            
        '''
        x_ft 是经过fft变换的
        x_ft_ 是经过fft变换与频率筛选的
        B T N* D
        '''
        for attention_CEM in self.attention_CEMs:
            x_ft_ = attention_CEM(x_ft_,x_ft_,x_ft_)
        y_ft = torch.zeros([B, T, N//2 + 1, D], device=x.device, dtype=x_ft.dtype)
        for i, j in enumerate(self.index_channel):
            y_ft[ :, :, j,: ] = x_ft_[ :, :, i,: ]
        y = torch.fft.irfft(y_ft, n=N, dim=2)
        y = y.permute(0, 2, 1, 3)
        return y 
    

class TEM(nn.Module):
    def __init__(self, d_model, num_layers, n_heads, index,c_out):
        super().__init__()
        self.num_layers = num_layers        
        self.embed_size = d_model
        self.index = index
        self.n_heads = n_heads        
        self.c_out = c_out
        print('c_out', c_out)
        self.TEMs = nn.ModuleList(Attention(self.embed_size, self.n_heads,self.index, self.c_out) for a in range(num_layers))

    def forward(self,x):
    # [B N T D]
        B,N,T,D = x.shape
        x_ft = torch.fft.rfft(x, dim=2)
        # print('x_ft', x_ft.dtype)
        x_ft_ = torch.zeros([B, N, len(self.index), D], device=x.device, dtype=x_ft.dtype)
        for i, j in enumerate(self.index):
            x_ft_[ :, :, i,: ] = x_ft[ :, :, j,: ]
        # print('x_ft_', x_ft_.dtype)
        # print('-------- TEM is used')
        '''
        x_ft 是经过fft变换的
        x_ft_ 是经过fft变换与频率筛选的
        B N T* D
        '''
        for TEM in self.TEMs:
            x_ft_ = TEM(x_ft_,x_ft_,x_ft_)
        y_ft = torch.zeros([B, N, T//2 + 1, D], device=x.device, dtype=x_ft.dtype)
        for i, j in enumerate(self.index):
            y_ft[ :, :, j,: ] = x_ft_[ :, :, i,: ]
        y = torch.fft.irfft(y_ft, n=T, dim=2)
        return y
        