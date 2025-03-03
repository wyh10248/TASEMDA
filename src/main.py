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

#dataset = 'HMDAD' 
dataset = 'peryton'
if dataset == 'HMDAD':
    embed_dim_default = 331
    max_len_x = 500
    epoch = 210
    a = 0.8
    b = 0.1
elif dataset == 'peryton':
    embed_dim_default = 1439
    max_len_x = 1500
    epoch = 170
    a = 0.5
    b = 0.4
else:
    raise ValueError(f"Unknown dataset: {dataset}")
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.001, type=float,help='learning rate')
parser.add_argument('--seed', type=int, default=105, help='Random seed.')
parser.add_argument('--k_fold', type=int, default=5, help='crossval_number.')
parser.add_argument('--epoch', type=int, default=epoch, help='train_number.')
parser.add_argument('--in_dim', type=int, default=embed_dim_default, help='in_feature.')
parser.add_argument('--hidden_dim', type=int, default=256, help='hidden_dim.')
parser.add_argument('--latent_dim', type=int, default=128, help='latent_dim.')
parser.add_argument('--num_layers', type=int, default=3, help='num_layers.')
parser.add_argument('--embed_dim', type=int, default=embed_dim_default, help='embed_dim.')
parser.add_argument('--num_heads', type=int, default=8, help='head_number.')
parser.add_argument('--output_dim', type=int, default=256, help='output_dim.')
parser.add_argument('--drop_rate', type=int, default=0.2, help='drop_rate.')
parser.add_argument('--max_len', type=int, default=max_len_x, help='max_len.')
args = parser.parse_args()

device = torch.device("cpu")


def cross_validation_experiment(A, microSimi, disSimi, args):
    positive_index_tuple = np.where(A == 1)
    rna_numbers = A.shape[0]
    dis_number = A.shape[1]
    positive_index_list = list(zip(positive_index_tuple[0], positive_index_tuple[1]))
    random.seed(args.seed)
    random.shuffle(positive_index_list)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    
    all_metrics = {
        'tpr': [], 'fpr': [], 'recall': [], 
        'precision': [], 'accuracy': [], 'F1': []
    }
    
    print(f"seed={args.seed}, evaluating microbe-disease....")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(positive_index_list)):
        print(f"------ Fold {fold+1} ------")
        
        # 训练集和测试集的正样本索引
        train_pos = [positive_index_list[i] for i in train_idx]
        test_pos = [positive_index_list[i] for i in test_idx]
        
        # 创建新矩阵，隐藏测试集正样本
        new_A = A.copy()
        for (i, j) in test_pos:
            new_A[i, j] = 0
            
        roc_A = new_A + A
        new_A_tensor = torch.from_numpy(new_A).to(device)
        x = constructHNet(new_A, microSimi, disSimi).float()
        edge_index = adjacency_matrix_to_edge_index(new_A)
        
        # 模型训练
        model = TASEMDA(args).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
        
        #训练循环
        model.train()
        for epoch in range(args.epoch):
            optimizer.zero_grad()
            pred, label = model(x, edge_index, new_A_tensor, train_model=True)
            loss = F.binary_cross_entropy(pred.float(), label)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

        # 模型评估
        model.eval()
        with torch.no_grad():
            pred_scores, _ = model(x, edge_index, new_A_tensor, train_model=False)
        
        # 展平预测分数和真实标签
        y_true = A.flatten()
        y_pred = pred_scores.cpu().numpy().flatten()
        
        # 排除训练集正样本（仅评估测试集正样本和所有负样本）
        test_mask = np.zeros_like(A, dtype=bool).flatten()
        for (i, j) in test_pos:
            test_mask[i * A.shape[1] + j] = True
        neg_mask = (A.flatten() == 0)
        eval_mask = test_mask | neg_mask
        
        y_true_eval = y_true[eval_mask]
        y_pred_eval = y_pred[eval_mask]
        
        # 按预测分数排序
        sorted_indices = np.argsort(-y_pred_eval)
        y_true_sorted = y_true_eval[sorted_indices]
        
        tp = np.cumsum(y_true_sorted)
        fp = np.cumsum(1 - y_true_sorted)
        tn = np.sum(y_true_eval == 0) - fp
        fn = np.sum(y_true_eval) - tp
        
        
        # 避免除以零
        epsilon = 1e-7
        tpr = tp / (tp + fn + epsilon)
        fpr = fp / (fp + tn + epsilon)
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        
        # 存储指标
        all_metrics['tpr'].append(tpr)
        all_metrics['fpr'].append(fpr)
        all_metrics['precision'].append(precision)
        all_metrics['recall'].append(recall)
        all_metrics['F1'].append(f1)
    
    # 计算平均指标
    df_tpr = pd.DataFrame(all_metrics['tpr'])
    mean_tpr = df_tpr.mean(axis=0, skipna=True).values  # 计算均值，忽略 NaN
    
    # 类似地处理其他指标
    df_fpr = pd.DataFrame(all_metrics['fpr'])
    mean_fpr = df_fpr.mean(axis=0, skipna=True).values
    
    df_precision = pd.DataFrame(all_metrics['precision'])
    mean_precision = df_precision.mean(axis=0, skipna=True).values
    
    df_recall = pd.DataFrame(all_metrics['recall'])
    mean_recall = df_recall.mean(axis=0, skipna=True).values
    
    df_f1 = pd.DataFrame(all_metrics['F1'])
    mean_f1 = df_f1.mean(axis=0, skipna=True).values
    
    # 计算AUC和AUPR
    auc = np.trapz(mean_tpr, mean_fpr)
    aupr = np.trapz(mean_precision, mean_recall)
    
    print(f"Average AUC: {auc:.4f}, AUPR: {aupr:.4f}")
    print(f"Average F1: {np.mean(mean_f1):.4f}")
    
   
    print("recall:%.4f,precision:%.4f" % (np.mean(mean_recall), np.mean(mean_precision)))
    
    # 绘图代码...
    plt.plot(mean_fpr, mean_tpr, label='mean ROC=%0.4f' % auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=0)
    plt.show()
    plt.plot(mean_recall, mean_precision, label='mean AUPR=%0.4f' % aupr)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend(loc=0)
    plt.show()
    
        

def main(args):
# #------data1---------
    # data_path = '../dataset/'
    # data_set = 'HMDAD/'#292*39

    # A = np.loadtxt(data_path + data_set + 'A.csv',delimiter=',')
    # disSi = np.loadtxt(data_path + data_set + 'disfunsim.csv',delimiter=',') 
    # disG= np.loadtxt(data_path + data_set + 'GSD.csv',delimiter=',') 
    # micfu = np.loadtxt(data_path + data_set + 'microfunsim.csv',delimiter=',')
    # micG = np.loadtxt(data_path + data_set + 'GSM.csv',delimiter=',')
    # MIS = a*micfu + (1-a)*micG
    # DIS = b*disSi + (1-b)*disG
#------data2---------
    data_path = '../dataset/'
    data_set = 'peryton/'#1396*43

    A = np.loadtxt(data_path + data_set + 'A.csv', delimiter=',')
    disSi = np.loadtxt(data_path + data_set + 'disfunsim.csv',delimiter=',') 
    disG= np.loadtxt(data_path + data_set + 'GSD.csv',delimiter=',') 
    micfu = np.loadtxt(data_path + data_set + 'microfunsim.csv',delimiter=',')
    micG = np.loadtxt(data_path + data_set + 'GSM.csv',delimiter=',')
    MIS = a*micfu + (1-a)*micG
    DIS = b*disSi + (1-b)*disG
    
#-------------------------

    cross_validation_experiment(A, MIS, DIS, args)


if __name__== '__main__':
    main(args)

 