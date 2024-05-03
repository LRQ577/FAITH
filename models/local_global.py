import torch.nn as nn
import torch
import torch.fft
# from .dctnet import dct,idct
import torch.nn.functional as F
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x shape: batch,seq_len,channels
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

# @lrq 3.27 多尺度分解结果加权结合
class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]

        # 假设有k个趋势成分
        self.k = len(self.moving_avg)  # 或者是您趋势成分的具体数量
        self.W_decomp = torch.nn.Parameter(torch.ones(self.k) / self.k)  # 初始化为均等权重，总和为1


    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(0))  # 增加一个新维度以便堆叠
        # 将所有趋势成分堆叠成一个矩阵 [k, B, T, N] 或者其他您趋势成分的维度
        moving_mean = torch.cat(moving_mean, dim=0)

        # 计算加权平均，结果维度应与单个趋势成分相同
        # W_decomp的形状 [k], moving_mean的形状 [k, B, T, N]
        # 使用einsum进行加权和，得到[B, T, N]
        moving_mean = torch.einsum('k,kbtn->btn', self.W_decomp, moving_mean)
        sea = x - moving_mean
        return sea, moving_mean


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate=0.1):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)

        self.initialize_weight(self.layer1)
        self.initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)

class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()
 
    def even(self, x):
        return x[:, :, ::2]
 
    def odd(self, x):
        return x[:, :, 1::2]
 
    def forward(self, x):
        '''Returns the odd and even part'''
        return (self.even(x), self.odd(x))
# class Align(nn.Module):
#     def __init__(self):
#         super(Align, self).__init__()
#     def zip_up_the_pants(even, odd):
#         even = even.permute(2, 1, 0)
#         odd = odd.permute(2, 1, 0) #L, B, D
#         even_len = even.shape[0]
#         odd_len = odd.shape[0]
#         mlen = min((odd_len, even_len))
#         _ = []
#         for i in range(mlen):
#             _.append(even[i].unsqueeze(0))
#             _.append(odd[i].unsqueeze(0))
#         if odd_len < even_len: 
#             _.append(even[-1].unsqueeze(0))
#         return torch.cat(_,0).permute(2,1,0) #B, L, D

#     def forward(self, even,odd):
#         '''Returns the odd and even part'''
#         return (self.zip_up_the_pants(even,odd))
# def zip_up_the_pants(even, odd):
def align(even, odd):
    even = even.permute(2, 1, 0)
    odd = odd.permute(2, 1, 0) #L, B, D
    even_len = even.shape[0]
    odd_len = odd.shape[0]
    mlen = min((odd_len, even_len))
    _ = []
    for i in range(mlen):
        _.append(even[i].unsqueeze(0))
        _.append(odd[i].unsqueeze(0))
    if odd_len < even_len: 
        _.append(even[-1].unsqueeze(0))
    return torch.cat(_,0).permute(2,1,0) #B, L, D

# ————————————————
# 版权声明：本文为CSDN博主「思考实践」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/weixin_43332715/article/details/127105230
class MIC(nn.Module):
    """
    MIC layer to extract local and global features
    """
    def __init__(self, feature_size=512, n_heads=8, dropout=0.05, decomp_kernel=[32], conv_kernel=[24], isometric_kernel=[18, 6], device='cuda'):
        super(MIC, self).__init__()
        self.src_mask = None
        self.conv_kernel = conv_kernel
        self.isometric_kernel = isometric_kernel
        self.device = device
        self.split = Splitting()
        # self.align = Align()
        # isometric convolution
        self.isometric_conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                                   kernel_size=i,padding=0,stride=1)
                                        for i in isometric_kernel])

        # downsampling convolution: padding=i//2, stride=i
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                             kernel_size=i,padding=i//2,stride=i)
                                  for i in conv_kernel])

        # upsampling convolution
        self.conv_trans = nn.ModuleList([nn.ConvTranspose1d(in_channels=feature_size, out_channels=feature_size,
                                                            kernel_size=i,padding=0,stride=i)
                                        for i in conv_kernel])

        self.decomp = nn.ModuleList([series_decomp(k) for k in decomp_kernel])
        self.merge = torch.nn.Conv2d(in_channels=feature_size, out_channels=feature_size, kernel_size=(len(self.conv_kernel), 1))

        self.fnn = FeedForwardNetwork(feature_size, feature_size*4, dropout)
        self.fnn_norm = torch.nn.LayerNorm(feature_size)

        self.norm = torch.nn.LayerNorm(feature_size)
        self.act = torch.nn.Tanh() #激活函数
        self.drop = torch.nn.Dropout(0.05)

    def conv_trans_conv(self, input, conv1d, conv1d_trans, isometric,k_s):
        batch, seq_len, channel = input.shape
        x = input.permute(0, 2, 1)
        # print("x_shape:",x.shape)
        # if k_s % 2 == 0:
        #     pad_l = 1 * (self.k_s - 2) // 2 + 1 #by default: stride==1 
        #     pad_r = 1 * (self.k_s) // 2 + 1 #by default: stride==1 

        # else:
        #     pad_l = 1 * (self.k_s - 1) // 2 + 1 # we fix the kernel size of the second layer as 3.
        #     pad_r = 1 * (self.k_s - 1) // 2 + 1

        # downsampling convolution
        #在这儿加入splitting

        #正常
        # x = self.drop(self.act(conv1d(x)))
        # x1 = x 
        x_even,x_odd = self.split(x)
 
        # x_even = nn.ReplicationPad1d((pad_l, pad_r))(x_even)
        # x_odd = nn.ReplicationPad1d((pad_l, pad_r))(x_odd)

        # x1 = self.drop(self.act(conv1d(x)))
        # print("x_even_shape：",x_even.shape)
        '''
        # # kernel_size[2, 4] 
        # x_shape: torch.Size([32, 512, 192])        192
        # x1_shape： torch.Size([32, 512, 17])
        # x1_shape： torch.Size([32, 512, 13])
        '''
        # x1 = self.drop(self.act(conv1d(x_even)))
        x_e = self.drop(self.act(conv1d(x_even)))
        # print("x_e.shape:",x_e.shape)
        x_o = self.drop(self.act(conv1d(x_odd)))

        x_e_o = x_o.mul(torch.exp(x_e))
        x_o_e = x_e.mul(torch.exp(x_o))
        # x1=self.align(x_e_o,x_o_e)
        x1=align(x_e_o,x_o_e)
        x1 = x1[:,:,:-1]
        x = x1
        
        ''' Local -Interaction
        x_even,x_odd = self.split(x)
 
        # x_even = nn.ReplicationPad1d((pad_l, pad_r))(x_even)
        # x_odd = nn.ReplicationPad1d((pad_l, pad_r))(x_odd)

        # x1 = self.drop(self.act(conv1d(x)))
        # print("x_even_shape：",x_even.shape)
        '''
        # # kernel_size[2, 4] 
        # x_shape: torch.Size([32, 512, 192])        192
        # x1_shape： torch.Size([32, 512, 17])
        # x1_shape： torch.Size([32, 512, 13])
        '''
        # x1 = self.drop(self.act(conv1d(x_even)))
        x_e = self.drop(self.act(conv1d(x_even)))
        # print("x_e.shape:",x_e.shape)
        x_o = self.drop(self.act(conv1d(x_odd)))

        x_e_o = x_o.mul(torch.exp(x_e))
        x_o_e = x_e.mul(torch.exp(x_o))
        # x1=self.align(x_e_o,x_o_e)
        x1=align(x_e_o,x_o_e)
        x1 = x1[:,:,:-1]
        x = x1
        '''
        # x=self.align(x_even,x_odd)
        #在这儿加入pants_zip合并xodd与xeven
        # isometric convolution 
        zeros = torch.zeros((x.shape[0], x.shape[1], x.shape[2]-1), device=self.device)
        x = torch.cat((zeros, x), dim=-1)
        # print("0",x.type())
  
        # print("1",x_f.type()


       #正常模块
        # x = dct(x) 
        # x = self.drop(self.act(isometric(x))) #self.act是激活函数Tanh
        # x = idct(x)
        x_f = torch.fft.fft(x) 
        x_f = torch.tensor(x_f,dtype = torch.float)
        # print("1",x_f.type())
        # x_f = abs(x_f)

        # x_if = torch.fft.ifft(x_f)#.to(x.device)
        # x = torch.tensor(x_if,dtype = torch.float)

        # print("2",x.type())
        # x = torch.abs(x)
        # x = self.drop(self.act(isometric(x))) #self.act是激活函数Tanh
        x = self.drop(self.act(isometric(x_f))) #self.act是激活函数Tanh

        x_if = torch.fft.ifft(x)#.to(x.device)
        x = torch.tensor(x_if,dtype = torch.float)
        
        ''' 因果频率增强模块
        
        x_f = torch.fft.fft(x) 
        x_f = torch.tensor(x_f,dtype = torch.float)
        # print("1",x_f.type())
        # x_f = abs(x_f)

        # x_if = torch.fft.ifft(x_f)#.to(x.device)
        # x = torch.tensor(x_if,dtype = torch.float)

        # print("2",x.type())
        # x = torch.abs(x)
        # x = self.drop(self.act(isometric(x))) #self.act是激活函数Tanh
        x = self.drop(self.act(isometric(x_f))) #self.act是激活函数Tanh

        x_if = torch.fft.ifft(x)#.to(x.device)
        x = torch.tensor(x_if,dtype = torch.float)
        '''
        # print("x1.shape:",x1.shape)
        # print("x.shape:",x.shape)
          #bug ,x1.shape: torch.Size([32, 512, 27]),x.shape: torch.Size([32, 512, 26])
        if x.shape[2] !=x1.shape[2]:
           
           x1 = x1[:,:,:x.shape[2]]


        # x = torch.fft.irfft(x)#
        # print("x1.shape_a:",x1.shape)
        # print("x.shape_a:",x.shape)
        x = self.norm((x+x1).permute(0, 2, 1)).permute(0, 2, 1)

        # upsampling convolution
        x = self.drop(self.act(conv1d_trans(x)))
        x = F.interpolate(x, size=seq_len)#插值
        # x = x[:, :, :seq_len]   # truncate
        # print("x.shape:",x.shape)
        # print("input.shape:",input.shape)
        x = self.norm(x.permute(0, 2, 1) + input)
        return x


    def forward(self, src):
        # multi-scale
        multi = []  
        for i in range(len(self.conv_kernel)):
            src_out, trend1 = self.decomp[i](src)
            src_out = self.conv_trans_conv(src_out, self.conv[i], self.conv_trans[i], self.isometric_conv[i],self.conv_kernel[i])
            multi.append(src_out)  

        # merge
        mg = torch.tensor([], device = self.device)
        for i in range(len(self.conv_kernel)):
            mg = torch.cat((mg, multi[i].unsqueeze(1)), dim=1)
        mg = self.merge(mg.permute(0,3,1,2)).squeeze(-2).permute(0,2,1)
        
        return self.fnn_norm(mg + self.fnn(mg))


class Seasonal_Prediction(nn.Module):
    def __init__(self, embedding_size=512, n_heads=8, dropout=0.05, d_layers=1, decomp_kernel=[32], c_out=1,
                conv_kernel=[2, 4], isometric_kernel=[18, 6], device='cuda'):
        super(Seasonal_Prediction, self).__init__()

        self.mic = nn.ModuleList([MIC(feature_size=embedding_size, n_heads=n_heads,
                                                   decomp_kernel=decomp_kernel,conv_kernel=conv_kernel, isometric_kernel=isometric_kernel, device=device)
                                      for i in range(d_layers)])

        self.projection = nn.Linear(embedding_size, c_out)

    def forward(self, dec):
        for mic_layer in self.mic:
            dec = mic_layer(dec)
        return self.projection(dec)
    



