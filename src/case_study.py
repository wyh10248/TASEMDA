import numpy as np
import torch
import pandas as pd
from TASEMDA import *
from sklearn.model_selection import KFold
import random
from utils import *
import matplotlib.pyplot as plt
import argparse
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.001, type=float,help='learning rate')
parser.add_argument('--seed', type=int, default=105, help='Random seed.')
parser.add_argument('--k_fold', type=int, default=5, help='crossval_number.')
parser.add_argument('--epoch', type=int, default=170, help='train_number.')
parser.add_argument('--in_dim', type=int, default=1439, help='in_feature.')
parser.add_argument('--hidden_dim', type=int, default=256, help='hidden_dim.')
parser.add_argument('--latent_dim', type=int, default=128, help='latent_dim.')
parser.add_argument('--num_layers', type=int, default=3, help='num_layers.')
parser.add_argument('--embed_dim', type=int, default=1439, help='embed_dim.')
parser.add_argument('--num_heads', type=int, default=8, help='head_number.')
parser.add_argument('--output_dim', type=int, default=256, help='output_dim.')
parser.add_argument('--drop_rate', type=int, default=0.2, help='drop_rate.')
parser.add_argument('--max_len', type=int, default=1500, help='max_len.')
args = parser.parse_args()

device = torch.device("cpu")# -*- coding: utf-8 -*-

a = 0.5
b = 0.4
set_seed(args.seed)
results = []
# #------data1---------
data_path = '../dataset/'
data_set = 'peryton/'#292*39

A = np.loadtxt(data_path + data_set + 'A.csv',delimiter=',')
disSi = np.loadtxt(data_path + data_set + 'disfunsim.csv',delimiter=',') 
disG= np.loadtxt(data_path + data_set + 'GSD.csv',delimiter=',') 
micfu = np.loadtxt(data_path + data_set + 'microfunsim.csv',delimiter=',')
micG = np.loadtxt(data_path + data_set + 'GSM.csv',delimiter=',')
MIS = a*micfu + (1-a)*micG
DIS = b*disSi + (1-b)*disG

# positive_index_tuple = np.where(A == 1)
# rna_numbers = A.shape[0]
# dis_number = A.shape[1]
# positive_index_list = list(zip(positive_index_tuple[0], positive_index_tuple[1]))
# random.seed(args.seed)
# random.shuffle(positive_index_list)

# kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)

# print(f"seed={args.seed}, evaluating microbe-disease....")

# for fold, (train_idx, test_idx) in enumerate(kf.split(positive_index_list)):
#     print(f"------ Fold {fold+1} ------")
    
#     # 训练集和测试集的正样本索引
#     train_pos = [positive_index_list[i] for i in train_idx]
#     test_pos = [positive_index_list[i] for i in test_idx]
    
#     # 创建新矩阵，隐藏测试集正样本
#     new_A = A.copy()
#     for (i, j) in test_pos:
#         new_A[i, j] = 0
        
#     new_A_tensor = torch.from_numpy(new_A).to(device)
#     x = constructHNet(new_A, MIS, DIS).float()
#     edge_index = adjacency_matrix_to_edge_index(new_A)

#     # 模型训练
#     model = TASEMDA(args).to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    
#     #训练循环
#     model.train()
#     for epoch in range(args.epoch):
#         optimizer.zero_grad()
#         pred, label = model(x, edge_index, new_A_tensor, train_model=True)
#         loss = F.binary_cross_entropy(pred.float(), label)
#         loss.backward()
#         optimizer.step()
#         print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")
    
# torch.save(model.state_dict(), 'model.pth')
# print("success")
#Colorectal Neoplasms,Crohn Disease案例研究

    

model = TASEMDA(args).to(device)

加载状态字典
model.load_state_dict(torch.load('model.pth'))

使模型处于评估模式
model.eval()

df = pd.read_csv('D:/Desktop文件/TASEMDA/dataset/peryton/association_matrix.csv')
columns = df.columns[1:] 
print(columns.shape)
col_idx = columns.get_loc('Colorectal Neoplasms')
col_idx = columns.get_loc('Crohn Disease')
df['Colorectal Neoplasms'] = df['Colorectal Neoplasms'].replace(1, 0)
df['Crohn Disease'] = df['Crohn Disease'].replace(1, 0)
#去掉第一列（微生物名称），并去掉第一行（疾病名称）
adjacency_matrix = df.iloc[:, 1:].values
x = constructHNet(adjacency_matrix, MIS, DIS).float()
edge_index = adjacency_matrix_to_edge_index(adjacency_matrix) 
with torch.no_grad():
    pred_scores, _ = model(x, edge_index, adjacency_matrix, train_model=False)
    pred = pred_scores.reshape(1396,43)
    red = pred.detach().cpu().numpy()  # 转换为 NumPy 数组
    pred_df = pd.DataFrame(pred)  # 转换为 DataFrame
    pred_df.to_csv('pred.csv', index=False)  # 保存为 CSV 文件





















