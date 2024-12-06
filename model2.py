import numpy as np
import torch as th
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from kan import KANLinear
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from test import GraphormerLayer
from torch_geometric.nn import MessagePassing, JumpingKnowledge
from torch_geometric.nn.dense.linear import Linear
from link_transformer import LinkTransformer

# MLP
import time
import math
from torch.nn.parameter import Parameter
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import dgl

class Specformer(nn.Module):

    def __init__(self, adj, transposed_list, e, u, G, hid_dim, n_class, S, K, batchnorm, num_diseases, num_mirnas,
                 d_sim_dim, m_sim_dim, out_dim, dropout, slope, node_dropout=0.5, input_droprate=0.0,
                 hidden_droprate=0.0, nclass=64, nfeat=64, nlayer=1, hidden_dim=128, nheads=1,
                 tran_dropout=0.0, feat_dropout=0.0, prop_dropout=0.0, norm='none', attention_dropout=0.5):

        super(Specformer, self).__init__()


        self.adj = adj
        self.transposed_list = transposed_list
        self.e = e
        self.u = u
        self.G = G
        self.hid_dim = hid_dim
        self.S = S
        self.K = K
        self.n_class = n_class
        self.num_diseases = num_diseases
        self.num_mirnas = num_mirnas
        self.disease_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1)
        self.mirna_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0)

        self.m_fc = nn.Linear(G.ndata['m_sim'].shape[1], hid_dim, bias=False)
        self.d_fc = nn.Linear(G.ndata['d_sim'].shape[1], hid_dim, bias=False)
        self.m_fc1 = nn.Linear(n_class, out_dim)
        self.d_fc1 = nn.Linear( n_class, out_dim)
        self.B = nn.Linear(128, 64)
        self.dropout1 = nn.Dropout(dropout)
        self.mlp = MLP(hid_dim, out_dim, n_class, input_droprate, hidden_droprate, batchnorm)
        self.kans = KANLinear(hid_dim, out_dim)


        self.predict = nn.Linear(out_dim * 2, 1)

        self.norm = norm
        self.nfeat = nfeat
        self.nlayer = nlayer
        self.nheads = nheads
        self.hidden_dim = hidden_dim

        self.feat_encoder = nn.Sequential(
            KANLinear(nfeat, hidden_dim),
            nn.ReLU(),
            KANLinear(hidden_dim, nclass),
        )

        # for arxiv & penn
        self.linear_encoder = nn.Linear(nfeat, hidden_dim)
        self.classify = nn.Linear(hidden_dim, nclass)


        self.mha = nn.MultiheadAttention(hidden_dim, nheads, tran_dropout)

        self.feat_dp1 = nn.Dropout(feat_dropout)
        self.feat_dp2 = nn.Dropout(feat_dropout)

        self.n_rna = 495
        hid_features = [64, 128, 256]
        self.proj = nn.Linear(64, 3, bias=True)
        self.n_hid_layers = 3
        tmp = [64,64,64,64]
        # self.JK = JumpingKnowledge('cat', tmp[-1], 4 + 1)
        self.dp1 = nn.Dropout(dropout)
        self.conv1 = nn.ModuleList()
        for i in range(len(hid_features)):
            self.conv1.append(
                GraphAttentionLayer(tmp[i], tmp[i + 1], nheads, residual=True),
            )

    def forward(self, graph,  diseases, mirnas,  training=True):



        self.G.apply_nodes(lambda nodes: {'z': self.dropout1(self.d_fc(nodes.data['d_sim']))}, self.disease_nodes)
        self.G.apply_nodes(lambda nodes: {'z': self.dropout1(self.m_fc(nodes.data['m_sim']))}, self.mirna_nodes)
        feats = self.G.ndata.pop('z')

        X = feats
        X = self.kans(X)
        x = X
        u = self.u
        e = self.e
        if training:  # Training Mode

            src_nodes, dst_nodes = graph.edges()
            edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
            edge_index = edge_index.to(torch.int64)

            if self.norm == 'none':
                h = self.feat_dp1(x)
                h = self.feat_encoder(h)
                h = self.feat_dp2(h)
            else:
                h = self.feat_dp1(x)
                h = self.kans(h)

            embd_tmp = h
            embd_list = [self.proj(embd_tmp) if self.proj is not None else embd_tmp]
            for i in range(self.n_hid_layers):
                embd_tmp = self.conv1[i](embd_tmp, edge_index)
                embd_list.append(embd_tmp)
            h = self.dp1(embd_tmp)



            if self.norm == 'none':
                 h = h
            else:
                h = self.feat_dp2(h)
                h = self.kans(h)
            X = h

            feat0 = X
            h_d = feat0[:self.num_diseases]
            h_m = feat0[self.num_diseases:]

            h_m = self.dropout1(F.elu(self.m_fc1(h_m)))     # (495,64)
            h_d = self.dropout1(F.elu(self.d_fc1(h_d)))     # （383,64）
            # (878,64)
            h = th.cat((h_d, h_m), dim=0)

            # 这里的disease和mirnas就是顶点，其对应位置就顶点之间存在边的label：0或者1
            # 疾病顶点特征
            h_diseases = h[diseases]  # disease中有重复的疾病名称;(17376,64)
            # mirnas顶点的特征
            h_mirnas = h[mirnas]

            h_concat = th.cat((h_diseases, h_mirnas), 1)  # (17376,128)
            predict_score = th.sigmoid(self.predict(h_concat))
            return predict_score





def drop_node(feats, drop_rate, training):
    n = feats.shape[0]
    drop_rates = th.FloatTensor(np.ones(n) * drop_rate)

    if training:

        masks = th.bernoulli(1. - drop_rates).unsqueeze(1)
        feats = masks.to(feats.device) * feats

    else:
        feats = feats * (1. - drop_rate)

    return feats




class GraphAttentionLayer(MessagePassing):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 residual: bool, dropout: float = 0.6, slope: float = 0.2, activation: nn.Module = nn.ELU()):
        super(GraphAttentionLayer, self).__init__(aggr='add', node_dim=0)
        self.in_features = in_features
        self.out_features = out_features
        self.heads = n_heads
        self.residual = residual

        self.attn_dropout = nn.Dropout(dropout)
        self.feat_dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(negative_slope=slope)
        self.activation = activation

        self.feat_lin = Linear(in_features, out_features * n_heads, bias=True, weight_initializer='glorot')
        self.attn_vec = nn.Parameter(torch.Tensor(1, n_heads, out_features))

        # use 'residual' parameters to instantiate residual structure
        if residual:
            self.proj_r = Linear(in_features, out_features, bias=False, weight_initializer='glorot')
        else:
            self.register_parameter('proj_r', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.attn_vec)

        self.feat_lin.reset_parameters()
        if self.proj_r is not None:
            self.proj_r.reset_parameters()

    def forward(self, x, edge_idx, size=None):
        # normalize input feature matrix
        x = self.feat_dropout(x)

        x_r = x_l = self.feat_lin(x).view(-1, self.heads, self.out_features)

        # calculate normal transformer components Q, K, V
        output = self.propagate(edge_index=edge_idx, x=(x_l, x_r), size=size)

        if self.proj_r is not None:
            output = (output.transpose(0, 1) + self.proj_r(x)).transpose(1, 0)

        output = self.activation(output)
        output = output.mean(dim=1)
        # output = normalize(output, p=2., dim=-1)

        return output

    def message(self, x_i, x_j, index, ptr, size_i):
        x = x_i + x_j
        x = self.leakyrelu(x)
        alpha = (x * self.attn_vec).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_dropout(alpha)

        return x_j * alpha.unsqueeze(-1)



def plot_encoded_eigenvalues(ax, encoded, epsilon,fig):
    cax = ax.imshow(encoded.T, cmap='coolwarm', aspect='auto')
    ax.set_title(f"ε = {epsilon}")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Eigenvalues")
    fig.colorbar(cax, ax=ax)