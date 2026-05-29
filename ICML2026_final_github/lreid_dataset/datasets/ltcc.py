# 文件名: ltcc.py

from __future__ import division, print_function, absolute_import
import os.path as osp
import re
import glob
import warnings
from collections import defaultdict
import random

from lreid_dataset.incremental_datasets import IncrementalPersonReIDSamples

class IncrementalSamples4ltcc(IncrementalPersonReIDSamples):
    """
    LTCC (Long-Term Clothes-Changing) 数据集解析器 (基于ID列表构建测试集)。
    """
    
    dataset_dir = 'LTCC_ReID'

    def __init__(self, root, **kwargs):
        dataset_path = osp.join(root, self.dataset_dir)
        self.info_dir = osp.join(dataset_path, 'info')
        self.train_img_dir = osp.join(dataset_path, 'train')
        self.test_img_dir = osp.join(dataset_path, 'test')
        
        self._check_before_run(dataset_path)

        train_cc_ids = self._read_ids(osp.join(self.info_dir, 'cloth-change_id_train.txt'))
        test_cc_ids = self._read_ids(osp.join(self.info_dir, 'cloth-change_id_test.txt'))
        train_sc_ids = self._read_ids(osp.join(self.info_dir, 'cloth-unchange_id_train.txt'))
        test_sc_ids = self._read_ids(osp.join(self.info_dir, 'cloth-unchange_id_test.txt'))
        
        train_ids = train_cc_ids | train_sc_ids
        test_ids = test_cc_ids | test_sc_ids

        all_train_imgs = self._scan_images(self.train_img_dir, train_ids)
        all_test_imgs = self._scan_images(self.test_img_dir, test_ids)
        
        train, pid2label = self._relabel(all_train_imgs)
        query, gallery_sc, gallery_cc = self._build_test_sets(all_test_imgs, test_cc_ids)
        
        self.gallery_sc = gallery_sc
        self.gallery_cc = gallery_cc
        
        super(IncrementalSamples4ltcc, self).__init__(train, query, gallery_sc, **kwargs)
        self.images_dir = ''

    def _read_ids(self, file_path):
        """从txt文件中读取ID列表。"""
        with open(file_path, 'r') as f:
            ids = {int(line.strip()) for line in f if line.strip()}
        return ids

    def _check_before_run(self, dataset_path):
        """检查所需目录和文件是否存在。"""
        if not osp.isdir(dataset_path):
            raise RuntimeError(f"'{dataset_path}' is not a valid directory.")
        if not osp.isdir(self.info_dir):
            raise RuntimeError(f"'{self.info_dir}' not found. ID list files are required.")
        required_files = ['cloth-change_id_train.txt', 'cloth-change_id_test.txt', 
                          'cloth-unchange_id_train.txt', 'cloth-unchange_id_test.txt']
        for f in required_files:
            if not osp.isfile(osp.join(self.info_dir, f)):
                raise RuntimeError(f"'{osp.join(self.info_dir, f)}' not found.")

    def _analysis_file_name(self, file_name):
        """从文件名解析信息。"""
        match = re.search(r'(\d+)_(\d+)_c(\d+)', file_name)
        if match:
            pid, clothes_id, camid = map(int, match.groups())
            return pid, camid, clothes_id
        return None

    def _scan_images(self, dir_path, allowed_ids):
        """扫描目录，只保留在 allowed_ids 中的图像。"""
        if not osp.isdir(dir_path):
            warnings.warn(f"Directory not found, skipping: {dir_path}")
            return []
        img_paths = glob.glob(osp.join(dir_path, '*.png')) + glob.glob(osp.join(dir_path, '*.jpg'))
        dataset = []
        for img_path in img_paths:
            parsed_info = self._analysis_file_name(osp.basename(img_path))
            if parsed_info and parsed_info[0] in allowed_ids:
                pid, camid, clothes_id = parsed_info
                camid -= 1
                dataset.append((img_path, pid, camid, clothes_id, 'ltcc'))
        return dataset

    def _relabel(self, dataset):
        """对给定的数据集进行PID重标签。"""
        if not dataset:
            return [], {}
        pids = sorted(list({item[1] for item in dataset}))
        pid2label = {pid: label for label, pid in enumerate(pids)}
        
        relabeled_dataset = []
        for item in dataset:
            path, pid, cid, clid, dom = item
            relabeled_dataset.append((path, pid2label[pid], cid, clid, dom))
        
        return relabeled_dataset, pid2label

    def _build_test_sets(self, test_imgs, cc_ids):
        """
        动态构建Query, Gallery_SC, 和 Gallery_CC。
        """
        query = []
        
        # 按 pid 对所有测试图像进行分组
        pid_groups = defaultdict(list)
        for item in test_imgs:
            pid_groups[item[1]].append(item)
            
        for pid, items in pid_groups.items():
            # 获取该ID的所有服装ID
            unique_clids = sorted(list({item[3] for item in items}))
            # 随机选择一套服装作为该ID的Query
            query_clid = random.choice(unique_clids)
            
            for item in items:
                if item[3] == query_clid:
                    query.append(item)
        
        # gallery_sc 包含所有测试图片 (评估时会自动忽略与query同ID同cam的图片)
        gallery_sc = test_imgs

        # gallery_cc: 从 gallery_sc 中移除 junk images
        # Junk image 的定义: 对于一个换装ID，其在gallery中的图片，如果与该ID在query中的服装相同，则为junk
        query_info = {(item[1], item[3]) for item in query} # (pid, clid)
        gallery_cc = []
        for g_item in gallery_sc:
            g_pid, g_clid = g_item[1], g_item[3]
            # Junk condition:
            if g_pid in cc_ids and (g_pid, g_clid) in query_info:
                continue # 这是Junk图像，跳过
            gallery_cc.append(g_item)

        return query, gallery_sc, gallery_cc