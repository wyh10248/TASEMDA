import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data

class AdaptiveGate(nn.Module):
    """自适应门控模块，动态融合原始矩阵与重构矩阵"""
    def __init__(self, input_dim):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出α值
        )
    
    def forward(self, adj_orig, adj_recon, node_features):
        # 计算每个节点的门控权重α
        alpha = self.gate_network(node_features)  # shape: [num_nodes, 1]
        # 扩展α到矩阵维度，进行加权融合
        adaptive_adj = alpha * adj_orig + (1 - alpha) * adj_recon
        return adaptive_adj

class GAT_GCN_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GAT_GCN_Model, self).__init__()
        
        # GAT编码器部分
        self.gat1 = GATConv(input_dim, hidden_dim, heads=8)
        self.gat2 = GATConv(hidden_dim * 8, latent_dim, heads=1)
        
        # 多层GCN解码器部分
        self.gcn1 = GCNConv(latent_dim,  hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, input_dim)  # 最后一层解码器，用于输出
        
        # 自适应门控
        self.gate = AdaptiveGate(input_dim)

    def encode(self, x, edge_index):
        # GAT编码阶段
        h = F.elu(self.gat1(x, edge_index))  # 第一层GAT
        h = F.elu(self.gat2(h, edge_index))  # 第二层GAT
        return h

    def decode(self, z, edge_index):
        # 多层GCN解码阶段
        h = F.relu(self.gcn1(z, edge_index))  # 第一层GCN
        adj_recon = self.gcn2(h, edge_index) # 最后一层GCN，输出邻接矩阵
        return adj_recon

    def forward(self, x, edge_index, adj_orig):
        z = self.encode(x, edge_index)  # 获取潜在表示
        adj_recon = self.decode(z, edge_index)  # 解码邻接矩阵
        adaptive_adj = self.gate(adj_orig, adj_recon, x)
        return adaptive_adj
    
    
#-------------------------------------------------
if __name__ == '__main__':

    
        # 创建一个简单的图数据
        # 假设我们有3个节点，每个节点的特征是3维的
        # 边：节点之间的连接
        node_features = torch.tensor([[1.0, 0.0, 0.0],  # Node 0
                                      [0.0, 1.0, 0.0],  # Node 1
                                      [0.0, 0.0, 1.0]], # Node 2
                                     dtype=torch.float)
        
        # 假设我们有一个简单的边列表，连接节点0-1, 节点1-2, 节点2-0
        edge_index = torch.tensor([[0, 1, 2],   # From nodes
                                   [1, 2, 0]],  # To nodes
                                  dtype=torch.long)
        
        # 创建一个图数据对象
        data = Data(x=node_features, edge_index=edge_index)
        
        # 创建模型实例
        input_dim = 3  # 输入维度是节点特征的维度
        hidden_dim = 8
        latent_dim = 2  # 潜在空间的维度
        
        model = GAT_GCN_Model(input_dim, hidden_dim, latent_dim)
        
        # 执行前向传播
        model.eval()  # 切换到评估模式
        x_recon, mu, logvar = model(data.x, data.edge_index)
        
        print( x_recon.shape)