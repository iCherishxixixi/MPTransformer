import operator
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

class ClusterAttn(nn.Module):
    def __init__(self, 
                 feature_dim,
                 latent_clus_dim,
                 groups=8,
                 expansion=1,
                 kernel_size=3, 
                 num_clusters=64,
                 dropout = 0, 
                 normalize_input=True, 
                 qkv_bias=True):

        super(ClusterAttn, self).__init__()
        self.num_clusters = num_clusters
        self.normalize_input = normalize_input
        self.fead = feature_dim
        self.projd = latent_clus_dim
        
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
        
        
        # Attention need        
        self.q = nn.Linear(self.fead, self.projd, bias=qkv_bias)
        self.kv = nn.Linear(self.fead, 2*self.projd, bias=qkv_bias)
        
        self.proj2 = nn.Linear(self.projd, self.fead, bias=qkv_bias)
        
    def init_weights(self):
        nn.init.kaiming_normal_(self.cluster_weights)
        nn.init.kaiming_normal_(self.c)        

    def forward(self, x):   
        B, num_segs = x.shape[:2]
        
        fea = self.fc_expansion(x)
        # [B, num_segs, expansion*feature_size]
        group_attention = self.fc_group_attention(fea)
        group_attention = group_attention.reshape(-1, num_segs * self.groups)
        # [B, num_segs*groups]
        reshaped_input = fea.reshape(-1, self.expansion * self.fead)
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
        fea = fea.reshape(-1, num_segs * self.groups,
                            self.group_feature_size)
        # [B, num_segs*groups, group_feature_size]
        new_centroids = torch.matmul(activation, fea)
        # [B, num_clusters, group_feature_size]
        new_centroids = self.proj(new_centroids)
        # [B, num_clusters, feature_dim]        
    
        q = self.q(x)
        kv = self.kv(new_centroids).reshape(B, self.num_clusters, 2, self.projd).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]
        
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.projd ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        qkv_attn = torch.matmul(attention_weights, v)
        result = self.proj2(qkv_attn)
        return result
        

