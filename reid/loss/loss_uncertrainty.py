# 文件名: loss_uncertrainty.py

import torch
from torch import nn
import torch.nn.functional as F

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension."""
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def hard_example_mining_set(dist_mat, labels_sample, labels, return_inds=False):
    """
    (修正版)
    为集合三元组损失进行难样本挖掘。
    这个版本移除了有风险的 .view() 操作，使其对批次结构更具鲁棒性，
    能够处理每个锚点对应不同数量正/负样本的情况。

    Args:
      dist_mat: 距离矩阵, shape [M, N] (e.g., [768, 128])
      labels_sample: `dist_mat`行对应的标签, shape [M] (e.g., [768])
      labels: `dist_mat`列对应的标签, shape [N] (e.g., [128])
    """
    M, N = dist_mat.size()

    # is_pos[i, j] = 1 if labels_sample[i] == labels[j]
    is_pos = labels_sample.unsqueeze(1).expand(M, N).eq(labels.unsqueeze(0).expand(M, N))
    is_neg = ~is_pos

    dist_ap_list = []
    dist_an_list = []
    
    # 遍历 dist_mat 的每一行 (对应 sample_feat 中的每一个样本)
    for i in range(M):
        # 找到当前样本 i 的所有正样本 (在 mean_feat 中)
        pos_mask = is_pos[i]
        
        # 找到当前样本 i 的所有负样本 (在 mean_feat 中)
        neg_mask = is_neg[i]

        # 如果没有正样本，则跳过此样本 (理论上不应发生，但作为保护)
        if not pos_mask.any():
            continue

        # 找到最难的正样本（距离最大）
        hardest_positive_dist = dist_mat[i][pos_mask].max()
        dist_ap_list.append(hardest_positive_dist)

        # 找到最难的负样本（距离最小）
        # 添加保护，以防万一没有负样本
        if neg_mask.any():
            hardest_negative_dist = dist_mat[i][neg_mask].min()
            dist_an_list.append(hardest_negative_dist)
        else:
            # 如果没有负样本, 我们可以用正样本的距离加上一个margin作为替代
            # 这样loss会是0，不会产生负面影响
            dist_an_list.append(hardest_positive_dist + 1.0) 

    # 确保我们找到了有效的样本对
    if not dist_ap_list:
        # 如果整个批次都没有有效的样本对, 返回一个0损失
        dummy_dist = torch.tensor(0.0, device=dist_mat.device, requires_grad=True)
        return dummy_dist, dummy_dist

    dist_ap = torch.stack(dist_ap_list)
    dist_an = torch.stack(dist_an_list)

    return dist_ap, dist_an


class TripletLoss_set(object):
    """
    Triplet loss using HARDER example mining.
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, merge_feat, labels, normalize_feature=False):
        """
        Args:
          merge_feat: a nested feature, shape [batch_size, 1+n_sampling, feat_dim]
          labels: ground truth labels with shape [batch_size]
        """
        # merge_feat 的形状是 [128, 7, 2048]
        BS = merge_feat.size(0)
        
        # mean_feat 是锚点 (anchors)
        mean_feat = merge_feat[:, 0, :]   # shape: [128, 2048]
        
        # sample_feat 是用于比较的正/负样本池
        sample_feat = merge_feat[:, 1:, :]    # shape: [128, 6, 2048]
        n_sample = sample_feat.size(1)
        sample_feat = sample_feat.reshape(BS * n_sample, -1)  # shape: [768, 2048]

        # 为 sample_feat 创建对应的标签
        labels_sample = labels.unsqueeze(1).expand(BS, n_sample).reshape(BS * n_sample)

        if normalize_feature:
            sample_feat = F.normalize(sample_feat, p=2, dim=1)
            mean_feat = F.normalize(mean_feat, p=2, dim=1)
            
        # dist_mat 的形状是 [768, 128]
        dist_mat = euclidean_dist(sample_feat, mean_feat)
        
        # 使用修正后的、更健壮的难样本挖掘函数
        dist_ap, dist_an = hard_example_mining_set(dist_mat, labels_sample, labels)

        dist_ap = dist_ap * (1.0 + self.hard_factor)
        dist_an = dist_an * (1.0 - self.hard_factor)

        y = dist_an.new_ones(dist_an.size())
        
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
            
        return loss, dist_ap, dist_an