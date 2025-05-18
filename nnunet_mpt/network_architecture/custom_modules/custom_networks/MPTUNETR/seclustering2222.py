import operator
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

def window_partition(x, window_size):
    x_shape = x.size()
    if len(x_shape) == 5:
        b, c, d, h, w = x_shape
        x = x.view(
            b,
            c,
            d // window_size[0],
            window_size[0],
            h // window_size[1],
            window_size[1],
            w // window_size[2],
            window_size[2],
        )
        windows = (
            x.permute(0, 1, 2, 4, 6, 3, 5, 7).contiguous().view(b, -1, window_size[0] * window_size[1] * window_size[2])
        )
    return windows


class ClusterAttn(nn.Module):
    def __init__(self, dim, patch_size, kernel_size=3, num_clusters=64, normalize_input=True, qkv_bias=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.

        """
        super(ClusterAttn, self).__init__()
        self.num_clusters = num_clusters
        self.alpha = 0
        self.normalize_input = normalize_input
        self.fead = reduce(operator.mul, patch_size, 1)
        self.patch_size = patch_size
        
        self.centroids = nn.Parameter(torch.rand(num_clusters, self.fead))
        
        self.dwc = nn.Conv3d(dim, 1, kernel_size, padding=(kernel_size - 1) // 2)    
        self.upc = nn.Conv3d(1, dim, kernel_size, padding=(kernel_size - 1) // 2) 
                
        self.q = nn.Linear(self.fead, self.fead, bias=qkv_bias)
        self.kv = nn.Linear(self.fead, 2*self.fead, bias=qkv_bias)
        

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        b, _, d, h, w = x.shape
        shortcut = x
        
        dnx = self.dwc(x)
        fea = window_partition(dnx, self.patch_size)
        
        N, C = fea.shape[:2]

        if self.normalize_input:
            fea = F.normalize(fea, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = F.softmax(torch.einsum('bnd,cd->bnc', fea, self.centroids), dim=1)
        new_centroids = torch.einsum('bnc,bnd->bcd', soft_assign, fea)

        q = self.q(fea)
        kv = self.kv(new_centroids).reshape(N, self.num_clusters, 2, self.fead).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]
        
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.fead ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        qkv_output = torch.matmul(attention_weights, v)
        reqkv = qkv_output.view(b, 1, d // self.patch_size[1], h // self.patch_size[1], w // self.patch_size[2], self.patch_size[0], self.patch_size[1], self.patch_size[2])
        new_req_o = reqkv.permute(0, 1, 2, 5, 3, 6, 4, 7).contiguous().view(b, 1, d, h, w)
        
        re = (self.upc(new_req_o)+x).permute(0, 2, 3, 4, 1)
        return re


if __name__ == "__main__":
    input_tensor = torch.randn(2, 70, 70, 70, 48) 
    model = ClusterAttn(dim=48, patch_size=[7,7,7])
    output = model(input_tensor)

    print("Output shape:", output.shape)
