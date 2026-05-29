# 文件名: cloth-change2026/CVPR2026/reid/utils/data/sampler.py

from __future__ import absolute_import
from collections import defaultdict
import math

import numpy as np
import copy
import random
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


def No_index(a, b):
    """
    返回列表 a 中所有元素不等于 b 的索引列表。
    """
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


class RandomIdentitySampler(Sampler):
    """
    随机身份采样器，从每个身份中随机采样若干个实例。
    """
    
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)

        # 兼容4元组和5元组的数据格式
        try:
            for index, (_, pid, _, _) in enumerate(data_source):
                self.index_dic[pid].append(index)
        except ValueError:
            for index, (_, pid, _, _, _) in enumerate(data_source):
                self.index_dic[pid].append(index)

        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)
    
    def __len__(self):
        return self.num_samples * self.num_instances
    
    def __iter__(self):
        indices = torch.randperm(self.num_samples).tolist()
        ret = []

        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]

            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            
            ret.extend(t)
        
        return iter(ret)


class MultiDomainRandomIdentitySampler(Sampler):
    """
    多域随机身份采样器。
    """
    
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances

        self.domain2pids = defaultdict(list)
        self.pid2index = defaultdict(list)

        # 统一使用5元组解包
        for index, (_, pid, _, _, domain) in enumerate(data_source):
            if pid not in self.domain2pids[domain]:
                self.domain2pids[domain].append(pid)
            self.pid2index[pid].append(index)

        self.pids = list(self.pid2index.keys())
        self.domains = list(sorted(self.domain2pids.keys()))
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        ret = []
        domain2pids = copy.deepcopy(self.domain2pids)

        for _ in range(8):
            for domain in self.domains:
                if len(domain2pids[domain]) >= 8:
                    pids = np.random.choice(domain2pids[domain], size=8, replace=False)
                else: # 如果该域ID不足8个，则有放回抽样
                    pids = np.random.choice(domain2pids[domain], size=8, replace=True)

                for pid in pids:
                    idxs = copy.deepcopy(self.pid2index[pid])
                    if len(idxs) >= 2:
                        idxs = np.random.choice(idxs, size=2, replace=False)
                    else: # 如果该ID实例不足2个，则有放回抽样
                        idxs = np.random.choice(idxs, size=2, replace=True)
                    ret.extend(idxs)

        return iter(ret)


class RandomMultipleGallerySampler(Sampler):
    """
    多样本采样器，用于从一个身份的多个摄像头视角或多个实例中采样数据。
    """
    
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances

        # ==================================================================
        # 最终决定版：移除 try-except，直接使用可以处理5元组的解包逻辑
        for index, (_, pid, cam, _, _) in enumerate(data_source):
            self.index_pid[index] = pid
            self.pid_cam[pid].append(cam)
            self.pid_index[pid].append(index)
        # ==================================================================

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])
            
            # ==================================================================
            # 最终决定版：使用5个变量来解包 self.data_source[i] 这个5元组
            _, i_pid, i_cam, _, _ = self.data_source[i]
            # ==================================================================

            ret.append(i)

            pid_i = self.index_pid[i]
            cams = self.pid_cam[pid_i]
            index = self.pid_index[pid_i]
            select_cams = No_index(cams, i_cam)

            num_to_sample = self.num_instances - 1
            if num_to_sample <= 0:
                continue

            if select_cams:
                if len(select_cams) >= num_to_sample:
                    cam_indexes = np.random.choice(select_cams, size=num_to_sample, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=num_to_sample, replace=True)
                for kk in cam_indexes:
                    ret.append(index[kk])
            else:
                select_indexes = No_index(index, i)
                if not select_indexes:
                    for _ in range(num_to_sample):
                        ret.append(i)
                    continue
                
                if len(select_indexes) >= num_to_sample:
                    ind_indexes = np.random.choice(select_indexes, size=num_to_sample, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=num_to_sample, replace=True)

                ret.extend(ind_indexes)

        return iter(ret)