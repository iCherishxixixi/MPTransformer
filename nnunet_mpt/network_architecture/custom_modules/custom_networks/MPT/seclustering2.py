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
    def __init__(self, 
                 dim, 
                 patch_size, 
                 groups=8,
                 expansion=2,
                 kernel_size=3, 
                 num_clusters=128,
                 dropout = 0, 
                 normalize_input=True, 
                 qkv_bias=True):
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
        self.normalize_input = normalize_input
        patch_size = [patch_size, patch_size, patch_size]
        
        
        self.fead = reduce(operator.mul, patch_size, 1)
        self.patch_size = patch_size
        #self.patch_num = patch_num
        
        # cluster
        self.groups = groups
        self.expansion = expansion
        self.group_feature_size = self.fead * expansion // groups
        self.fc_expansion = nn.Linear(self.fead, expansion * self.fead)
        self.fc_group_attention = nn.Sequential(
                                                nn.Linear(expansion * self.fead, groups), 
                                                nn.Sigmoid())
        self.activation_bn = nn.BatchNorm1d(groups * num_clusters)
        self.proj = nn.Sequential( 
                                  nn.Dropout(dropout), 
                                  nn.Linear(self.group_feature_size, self.fead),
                                  nn.BatchNorm1d(num_clusters))
                                                
        self.cluster_weights = nn.Parameter(torch.randn(expansion * self.fead, groups * num_clusters))
        self.c = nn.Parameter(torch.randn(1, self.group_feature_size, num_clusters))
        

        # compress
        self.dwc = nn.Conv3d(dim, 1, kernel_size, padding=(kernel_size - 1) // 2)    
        self.upc = nn.Conv3d(1, dim, kernel_size, padding=(kernel_size - 1) // 2) 
        
        # Attention need        
        self.q = nn.Linear(self.fead, self.fead, bias=qkv_bias)
        self.kv = nn.Linear(self.fead, 2*self.fead, bias=qkv_bias)
        
    def init_weights(self):
        nn.init.kaiming_normal_(self.cluster_weights)
        nn.init.kaiming_normal_(self.c)        

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        b, _, d, h, w = x.shape
        shortcut = x
        
        dnx = self.dwc(x)
        fea = window_partition(dnx, self.patch_size)
             
        B, num_segs = fea.shape[:2]
        if num_segs > self.num_clusters:
            fea2 = self.fc_expansion(fea)
            # [B, num_segs, expansion*feature_size]
            group_attention = self.fc_group_attention(fea2)
            group_attention = group_attention.reshape(-1, num_segs * self.groups)
            # [B, num_segs*groups]
            reshaped_input = fea2.reshape(-1, self.expansion * self.fead)
            # [B*num_segs, expansion*feature_size]
            activation = self.activation_bn(reshaped_input @ self.cluster_weights)
            # [B*num_segs, groups*num_clusters]
            activation = activation.reshape(-1, num_segs * self.groups,
                                        self.num_clusters)
            # [B, num_segs*groups, num_clusters]
            activation = F.softmax(activation, dim=-1)
        
            activation = activation * group_attention.unsqueeze(-1)
            # [B, num_segs*groups, num_clusters]
            a_sum = activation.sum(dim=1, keepdim=True)
            # [B, 1, num_clusters]
            a = a_sum * self.c
            # [B, group_feature_size, num_clusters]
            activation = activation.transpose(1, 2)
            # [B, num_clusters, num_segs*groups]
            fea2 = fea2.reshape(-1, num_segs * self.groups,
                              self.group_feature_size)
            # [B, num_segs*groups, group_feature_size]
            new_centroids = torch.matmul(activation, fea2)
            # [B, num_clusters, group_feature_size]
            new_centroids = self.proj(new_centroids)
            # [B, num_clusters, feature_size]        
    
            '''
            N, C = fea.shape[:2]

            if self.normalize_input:
            fea = F.normalize(fea, p=2, dim=1)  # across descriptor dim

            # soft-assignment
            soft_assign = F.softmax(self.mlp(fea.permute(0,2,1)).permute(0,2,1), dim=1)
            #F.softmax(torch.einsum('bnd,cd->bnc', fea, self.centroids), dim=1)
            
             residual = fea.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign.unsqueeze(2)
            print(soft_assign.shape)
            sys.exit()
        

            new_centroids = torch.einsum('bnc,bnd->bcd', soft_assign, fea)
            '''
        else:
            new_centroids = fea
            self.num_clusters = num_segs
        q = self.q(fea)
        kv = self.kv(new_centroids).reshape(B, self.num_clusters, 2, self.fead).permute(2, 0, 1, 3)
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
