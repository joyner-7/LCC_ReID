# 文件名: evaluators.py

from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import numpy as np
import torch

from .evaluation_metrics import cmc, mean_ap, mean_ap_cuhk03
# === MODIFICATION ===: 移除对 extract_cnn_feature 的依赖
# from .feature_extraction import extract_cnn_feature
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking
from torch.nn import functional as F

# === MODIFICATION START: 修改 extract_features 函数以适应新模型 ===
def extract_features(model, data_loader, training_phase=None):
    model.eval() # 确保模型处于评估模式
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, data_batch in enumerate(data_loader):
            # 解包数据批次
            if len(data_batch) == 5:
                imgs, fnames, pids, cids, domians = data_batch
            else: # 兼容旧的4元组格式
                imgs, fnames, pids, cids = data_batch
                
            data_time.update(time.time() - end)
            
            # 直接调用模型并解包新的8元素元组
            # 在评估模式下，我们只关心第一个元素 s_features_id
            outputs = model(imgs.cuda())
            feats = outputs[0].cpu() # s_features_id

            for fname, output, pid in zip(fnames, feats, pids):
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

    return features, labels
# === MODIFICATION END ===


def extract_features_print(model, data_loader, training_phase=None):
    # === MODIFICATION START: 同样修改这个函数 ===
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    features = OrderedDict()
    labels = OrderedDict()
    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, cids, domians) in enumerate(data_loader):
            data_time.update(time.time() - end)
            
            # 直接调用模型并解包
            outputs = model(imgs.cuda())
            feats = outputs[0].cpu() # s_features_id
            
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid
            batch_time.update(time.time() - end)
            end = time.time()
    return features, labels
    # === MODIFICATION END ===


def pairwise_distance(features, query=None, gallery=None, metric=False):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not False:
            x = F.normalize(x, p=2, dim=1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m

    # 这个解包逻辑取决于你的query/gallery列表的结构，暂时保持不变
    # 如果你的列表是5元组，这个解包是正确的
    try:
        x = torch.cat([features[f].unsqueeze(0) for f, _, _, _, _ in query], 0)
        y = torch.cat([features[f].unsqueeze(0) for f, _, _, _, _ in gallery], 0)
    except IndexError: # 兼容可能存在的3元组或4元组格式
        x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
        y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)

    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not False:
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    
    dist_m.addmm_(x, y.t(), beta=1, alpha=-2)
    
    return dist_m.cpu(), x.cpu().numpy(), y.cpu().numpy()


def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False, cuhk03=False):
    if query is not None and gallery is not None:
        # 这个解包逻辑取决于你的query/gallery列表的结构，暂时保持不变
        try:
            query_ids = [pid for _, pid, _, _, _ in query]
            gallery_ids = [pid for _, pid, _, _, _ in gallery]
            query_cams = [cam for _, _, cam, _, _ in query]
            gallery_cams = [cam for _, _, cam, _, _ in gallery]
        except IndexError:
            query_ids = [pid for _, pid, _ in query]
            gallery_ids = [pid for _, pid, _ in gallery]
            query_cams = [cam for _, _, cam in query]
            gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # 计算平均精度均值 (Mean AP)
    if cuhk03:
        mAP = mean_ap_cuhk03(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    else:
        mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    if not cmc_flag:
        return mAP

    if cuhk03:
        cmc_configs = {
            'cuhk03': dict(separate_camera_set=True,
                           single_gallery_shot=True,
                           first_match_break=False)
        }
        cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                                query_cams, gallery_cams, **params)
                      for name, params in cmc_configs.items()}
        print('CUHK03 CMC Scores:')
        for k in cmc_topk:
            print('  top-{:<4}{:12.1%}'
                  .format(k, cmc_scores['cuhk03'][k - 1]))
        return cmc_scores['cuhk03'][0], mAP
    else:
        cmc_configs = {
            'market1501': dict(separate_camera_set=False,
                               single_gallery_shot=False,
                               first_match_break=True),
        }
        cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                                query_cams, gallery_cams, **params)
                      for name, params in cmc_configs.items()}
        print('CMC Scores:')
        for k in cmc_topk:
            print('  top-{:<4}{:12.1%}'
                  .format(k, cmc_scores['market1501'][k - 1]))
        return cmc_scores['market1501'][0], mAP


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, metric=None, cmc_flag=False,
                 rerank=False, pre_features=None, cuhk03=False, training_phase=None):
        if pre_features is None:
            features, _ = extract_features(self.model, data_loader)
        else:
            features = pre_features

        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery, metric=metric)

        results = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery,
                               cmc_flag=cmc_flag, cuhk03=cuhk03)

        if not rerank:
            return results

        print('Applying person re-ranking ...')
        distmat_qq, _, _ = pairwise_distance(features, query, query, metric=metric)
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery, metric=metric)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())

        return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery,
                            cmc_flag=cmc_flag, cuhk03=cuhk03)