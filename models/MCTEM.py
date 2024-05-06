import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.local_global import Seasonal_Prediction, series_decomp_multi
from layers.Attention import Attention
from layers.Attention_channel import Attention_channel


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
        self.embed_size = 128 #embed_size
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
        self.c_out = configs.c_out
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
        

        # 实例化attention
        self.n_heads = configs.n_heads
        self.attention_TEM = Attention(self.embed_size, self.n_heads,self.index, self.c_out)
        self.attention_CEM = Attention_channel(self.embed_size, self.n_heads,self.index_channel, self.seq_len)

        # @lrq 趋势项
        self.regression_mlp = MLPRegression(configs.seq_len, self.hidden_size, configs.pred_len) 

        # @lrq 映射输出维度
        self.to_output = nn.Linear(self.seq_len, self.pred_len)       

        # self.fc = nn.Sequential(
        #     nn.Linear(self.seq_length * self.embed_size, self.hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.hidden_size, self.pre_length)
        # )
        # @lrq合并用的函数
        self.W_fuse = torch.nn.Parameter(torch.ones(2))

        self.linear = nn.Linear(self.embed_size, self.embed_size)


    # dimension extension 这里其实比较怀疑，挤出一个维度，然后做embedding，映射，真的有效果吗？这里的embedding也不是做卷积，肯定算升维，算embedding吗
    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D   # Batch 一般可以不考虑，只是一个批量操作。
        y = self.embeddings
        return x * y

    def CEM(self,x):
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
        for i in range(self.num_layers):
            x_ft_ = self.attention_CEM(x_ft_,x_ft_,x_ft_)
        y_ft = torch.zeros([B, T, N//2 + 1, D], device=x.device, dtype=x_ft.dtype)
        for i, j in enumerate(self.index_channel):
            y_ft[ :, :, j,: ] = x_ft_[ :, :, i,: ]
        y = torch.fft.irfft(y_ft, n=N, dim=2)
        y = y.permute(0, 2, 1, 3)
        return y 
    
    def TEM(self,x):
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
        for i in range(self.num_layers):
            x_ft_ = self.attention_TEM(x_ft_,x_ft_,x_ft_)
        y_ft = torch.zeros([B, N, T//2 + 1, D], device=x.device, dtype=x_ft.dtype)
        for i, j in enumerate(self.index):
            y_ft[ :, :, j,: ] = x_ft_[ :, :, i,: ]
        y = torch.fft.irfft(y_ft, n=T, dim=2)
        return y
         


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
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x) # 8, 8, 96, 128

        # print('----111')
        if self.channel_independence == 1:
            # print("channel_independence:",self.channel_independence)
            for i in range(self.num_blocks):
                x = self.CEM(x)
                # print('the num of CEM blocks{}'.format(i))
        if self.temporal_independence == 1:
            for i in range(self.num_blocks):
                x = self.TEM(x)
                # print('the num of TEM blocks{}'.format(i))
        
        # x = self.linear(x)

        # print("channel_independence:",self.channel_independence)

        # 特征降维 @lrq 3.28
        x = self.dim_reduction(x)
        x = x.squeeze(-1)
        
        # @lrq 输出维度映射
        x = self.to_output(x)
        # [B N T] --> [B T N]
        x = x.permute(0, 2, 1)


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
