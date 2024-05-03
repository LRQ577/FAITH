import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attention_channel(nn.Module):
    def __init__(self, d_model, n_heads, index, seq_len, d_keys=None,
                 d_values=None) :
        super(Attention_channel,self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        # self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.d_model = d_model
        self.seq_len = seq_len
        self.inner_correlation = inner_correlation(d_model,index,seq_len)


        self.sparsity_threshold = 0.01


    def forward(self, queries, keys, values, attn_mask=None):
        # x = [B N L* D ]
        B, N ,L, _ = queries.shape
        _, N, S, _ = keys.shape
        H = self.n_heads
        # print('B:{}---N:{}---L:{}---S:{}---H:{}', B, N, L, S, H)

        #  考虑实部
        queries_real = queries.real
        keys_real = keys.real
        values_real = values.real

        # print('---------------------',queries.dtype)
        queries_real = self.query_projection(queries_real).view(B, N, L, H, -1)
        keys_real = self.key_projection(keys_real).view(B, N, S, H, -1)
        values_real = self.value_projection(values_real).view(B, N, S, H, -1)


        # print('---------------------',queries.shape)

        out_real = self.inner_correlation(
            queries_real,
            keys_real,
            values_real,
            self.d_model,
            
        )
        out_real = out_real.reshape(B, N, L, -1)
        out_real = self.out_projection(out_real)


        # # 考虑虚部
        # queries_imag = queries.imag
        # keys_imag = keys.imag
        # values_imag = values.imag
        # queries_imag = self.query_projection(queries_imag).view(B, N, L, H, -1)
        # keys_imag = self.key_projection(keys_imag).view(B, N, S, H, -1)
        # values_imag = self.value_projection(values_imag).view(B, N, S, H, -1)
        # out_imag = self.inner_correlation(
        #     queries_imag,
        #     keys_imag,
        #     values_imag,
        #     self.d_model,
        # )
        # out_imag = out_imag.reshape(B, N, L, -1)
        # out_imag = self.out_projection(out_imag)



        # out = torch.stack((out_real, out_imag), dim=-1)
        # out = F.softshrink(out, lambd=self.sparsity_threshold)
        # out = torch.view_as_complex(out)
        out = out_real

        # out = out.reshape(B, N, L, -1)
        return out 
    
        
class inner_correlation(nn.Module):
    def __init__(self, d_model, index_len, seq_len,n_heads=8):
        super(inner_correlation, self).__init__()

        self.in_channels = d_model
        self.out_channels = d_model
        self.seq_len =seq_len

        self.scale = (1 / (self.in_channels * self.out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(self.seq_len ,8, self.in_channels // 8, self.out_channels // 8, len(index_len), dtype=torch.float))

             
    def forward(self, q, k, v, d_model, attn_mask=None):
        B, N, L, H, E = q.shape
        xq = q.permute(0, 1, 3, 4, 2)
        xk = k.permute(0, 1, 3, 4, 2)
        xv = v.permute(0, 1, 3, 4, 2)
        # print('xq:shape',xq.shape)
        # print('xk:shape',xk.shape)
        # print('xv:shape',xv.shape)

        # q * k
        xqk = torch.einsum("bnhex,bnhey->bnhxy", xq, xk)
        # print("xqk.shape",xqk.shape)


        # softmax(q * k)

        xqk = torch.softmax(abs(xqk), dim=-1)


        # softmax(q * k) *v
        xqkv = torch.einsum("bnlxy,bnhey->bnhex", xqk, xv)
        # xqkv = torch.complex(xqkv, torch.zeros_like(xqkv))  # Convert xqkv to complex type
        out = torch.einsum("bnhex,nheox->bnhox", xqkv, self.weights1)

        return out
    


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
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
