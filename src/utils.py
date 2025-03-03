import csv
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
import random
import torch

device = torch.device("cpu")

def set_seed(seed=50):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def constructHNet(train_mirna_disease_matrix, mirna_matrix, disease_matrix):
    mat1 = np.hstack((mirna_matrix, train_mirna_disease_matrix))
    mat2 = np.hstack((train_mirna_disease_matrix.T, disease_matrix))
    mat3= np.vstack((mat1, mat2))
    node_embeddings = torch.tensor(mat3)
    return node_embeddings

def train_features_choose(rel_adj_mat, features_embedding):
    rna_nums = rel_adj_mat.shape[0]
    features_embedding_rna = features_embedding[0:rna_nums, :]
    features_embedding_dis = features_embedding[rna_nums:features_embedding.size()[0], :]
    train_features_input, train_lable = [], []
    # positive position index
    positive_index_tuple = torch.where(rel_adj_mat == 1)
    positive_index_list = list(zip(positive_index_tuple[0], positive_index_tuple[1]))

    for (r, d) in positive_index_list:
        # positive samples
        train_features_input.append((features_embedding_rna[r, :] * features_embedding_dis[d, :]).unsqueeze(0))
        train_lable.append(1)
        # negative samples
        negative_colindex_list = []
        for i in range(1):
            j = np.random.randint(rel_adj_mat.size()[1])
            while (r, j) in positive_index_list:
                j = np.random.randint(rel_adj_mat.size()[1])
            negative_colindex_list.append(j)
        for nums_1 in range(len(negative_colindex_list)):
            train_features_input.append(
                (features_embedding_rna[r, :] * features_embedding_dis[negative_colindex_list[nums_1], :]).unsqueeze(0))
        for nums_2 in range(len(negative_colindex_list)):
            train_lable.append(0)
    train_features_input = torch.cat(train_features_input, dim=0)
    train_lable = torch.FloatTensor(np.array(train_lable)).unsqueeze(1)
    return train_features_input.to(device), train_lable.to(device)

def test_features_choose(rel_adj_mat, features_embedding):
    rna_nums, dis_nums = rel_adj_mat.shape[0], rel_adj_mat.shape[1]
    features_embedding_rna = features_embedding[0:rna_nums, :]
    features_embedding_dis = features_embedding[rna_nums:features_embedding.size()[0], :]
    test_features_input, test_lable = [], []

    for i in range(rna_nums):
        for j in range(dis_nums):
            test_features_input.append((features_embedding_rna[i, :] * features_embedding_dis[j, :]).unsqueeze(0))
            test_lable.append(rel_adj_mat[i, j])

    test_features_input = torch.cat(test_features_input, dim=0)
    
    return test_features_input.to(torch.float32),test_lable

def sort_matrix(score_matrix, interact_matrix):
    '''
    实现矩阵的列元素从大到小排序
    1、np.argsort(data,axis=0)表示按列从小到大排序
    2、np.argsort(data,axis=1)表示按行从小到大排序
    '''
    sort_index = np.argsort(-score_matrix, axis=0)  # 沿着行向下(每列)的元素进行排序
    score_sorted = np.zeros(score_matrix.shape)
    y_sorted = np.zeros(interact_matrix.shape)
    for i in range(interact_matrix.shape[1]):
        score_sorted[:, i] = score_matrix[:, i][sort_index[:, i]]
        y_sorted[:, i] = interact_matrix[:, i][sort_index[:, i]]
    return y_sorted, score_sorted




#计算邻接矩阵
def get_adjacency_matrix(similarity_matrix, threshold):
    n = similarity_matrix.shape[0]
    adjacency_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            if similarity_matrix[i][j] >= threshold:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1

    return adjacency_matrix


def adjacency_matrix_to_edge_index(adjacency_matrix):
  adjacency_matrix = torch.tensor(adjacency_matrix)
  num_nodes = adjacency_matrix.shape[0]
  edge_index = torch.nonzero(adjacency_matrix, as_tuple=False).t()
  return edge_index

def calculate_performace(pred_y, labels):  # 删除test_num参数
    tp = fp = tn = fn = 0
    
    # 确保两个数组长度一致
    assert len(pred_y) == len(labels), f"预测结果和标签长度不一致: {len(pred_y)} vs {len(labels)}"
    
    for true_label, pred_label in zip(labels, pred_y):  # 使用zip遍历配对元素
        if true_label == 1:
            if pred_label == 1:
                tp += 1
            else:
                fn += 1
        else:
            if pred_label == 0:
                tn += 1
            else:
                fp += 1

    # 计算结果指标
    acc = (tp + tn) / len(labels)
    
    # 处理分母为0的情况
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    MCC = (tp * tn - fp * fn) / denominator if denominator != 0 else 0
    
    f1_score = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0

    return acc, precision, sensitivity, specificity, MCC, f1_score

def transfer_label_from_prob(proba, threshold):
    proba = (proba - proba.min()) / (
            proba.max() - proba.min())
    label = [1 if val >= threshold else 0 for val in proba]
    return label

