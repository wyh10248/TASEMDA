import torch
import torch.nn as nn
import torch.nn.functional as F
from GAGCVAE import *
from DAFormer import DomainAdaptedTransformerEncoder
import numpy as np
from utils import *
device = torch.device("cpu")


class MLP(nn.Module):
    def __init__(self, embedding_size, drop_rate):
        super(MLP, self).__init__()
        self.embedding_size = embedding_size
        self.drop_rate = drop_rate

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif type(m) == nn.Conv2d:
                nn.init.uniform_(m.weight)

        self.mlp_prediction = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 2, self.embedding_size // 4),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 4, self.embedding_size // 6),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 6, 1, bias=False),
            nn.Sigmoid()
        )
        self.mlp_prediction.apply(init_weights)

    def forward(self, rd_features_embedding):
        predict_result = self.mlp_prediction(rd_features_embedding)
        return predict_result




class TASEMDA(nn.Module):
    def __init__(self, args):
        super().__init__()
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        input_dim = args.in_dim
        hidden_dim = args.hidden_dim
        latent_dim = args.latent_dim
        num_layers = args.num_layers
        embed_dim = args.embed_dim
        num_heads = args.num_heads
        output_dim = args.output_dim
        drop_rate = args.drop_rate
        max_len = args.max_len

        self.FNN = nn.Linear(embed_dim, output_dim) 

        self.GAGCVAE = GAT_GCN_Model(input_dim, hidden_dim, latent_dim)
        self.DAFormer = DomainAdaptedTransformerEncoder(num_layers, output_dim, num_heads, max_len=max_len)
        self.mlp_prediction = MLP(output_dim, drop_rate) 
        #self.mlp_prediction = MLP(331, 0.2)
    def vae_loss(self, x, x_recon, mu, logvar):
        # 重构损失（Binary Cross-Entropy）
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='mean')
        
        # KL散度（正则项）
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kl_loss
    
    def classification_loss(self, pred, label):
        gamma = 2.0
        alpha = 0.25
        bce = F.binary_cross_entropy(pred.float(), label.float(), reduction='none')
        pt = torch.exp(-bce)
        focal_loss = alpha * (1-pt)**gamma * bce
        return focal_loss.mean()
        
    def forward(self, x, edge_index, rel_matrix, train_model):
        output1 = self.GAGCVAE(x, edge_index, x)
        hidden_X = self.FNN(output1)
        output2 = self.DAFormer(hidden_X.unsqueeze(0))
        outputs = F.leaky_relu(output2)
        if train_model:
            train_features_inputs, train_lable = train_features_choose(rel_matrix, outputs)
            train_mlp_result = self.mlp_prediction(train_features_inputs)
            return train_mlp_result, train_lable
        else:
            test_features_inputs, test_lable = test_features_choose(rel_matrix, outputs)
            test_mlp_result = self.mlp_prediction(test_features_inputs)
            return test_mlp_result, test_lable

       


