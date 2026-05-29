# 文件名: lreid_dataset/datasets/prcc.py
# --- MODIFIED: To support both Same-Clothes and Cross-Clothes evaluation ---

from __future__ import division, print_function, absolute_import
import os
import os.path as osp
import glob
import warnings
from prettytable import PrettyTable # 需要导入

# 你自己的项目中可能没有这个基类，或者它的定义不同。
# 为了代码独立和健壮，我们让这个类不继承任何东西。
# from lreid_dataset.incremental_datasets import IncrementalPersonReIDSamples

class IncrementalSamples4prcc: # <--- 不再继承
    """
    PRCC (Person Re-identification under Changing Clothes) 数据集解析器
    """
    dataset_dir = 'prcc'
    
    # 增加一个 _show_info 方法用于打印数据集信息
    def _show_info(self, train, query_cc, query_sc, gallery):
        def analyze(samples):
            if not samples: return 0, 0, 0
            pid_num = len(set([sample[1] for sample in samples]))
            cid_num = len(set([sample[2] for sample in samples]))
            sample_num = len(samples)
            return sample_num, pid_num, cid_num

        train_info = analyze(train)
        query_cc_info = analyze(query_cc)
        query_sc_info = analyze(query_sc)
        gallery_info = analyze(gallery)

        table = PrettyTable(['set', 'images', 'identities', 'cameras'])
        self.num_train_pids=train_info[1]
        table.add_row(['prcc', '', '', ''])
        table.add_row(['train', str(train_info[0]), str(train_info[1]), str(train_info[2])])
        table.add_row(['query (cross-clothes)', str(query_cc_info[0]), str(query_cc_info[1]), str(query_cc_info[2])])
        table.add_row(['query (same-clothes)', str(query_sc_info[0]), str(query_sc_info[1]), str(query_sc_info[2])])
        table.add_row(['gallery', str(gallery_info[0]), str(gallery_info[1]), str(gallery_info[2])])
        print(table)


    def __init__(self, root, **kwargs):
        dataset_path = osp.join(root, self.dataset_dir, 'rgb')
        self.train_path = osp.join(dataset_path, 'train')
        self.test_path = osp.join(dataset_path, 'test')

        self._check_before_run(dataset_path)

        train_raw, train_pids = self._parse_train_path(self.train_path)
        train, self.pid2label = self._relabel(train_raw, pids=train_pids, is_train=True)

        query_cc, query_sc, gallery = self._parse_test_path(self.test_path)
        
        self.train = train
        self.query = query_cc      # 默认的 query 是 cross-clothes
        self.query_cc = query_cc
        self.query_sc = query_sc
        self.gallery = gallery
        self.images_dir = ''
        
        self._show_info(self.train, self.query_cc, self.query_sc, self.gallery)
        
        if len(self.query_cc) == 0 or len(self.gallery) == 0:
            print("警告: Cross-Clothes Query 或 Gallery 为空！")
        if len(self.query_sc) == 0:
            print("警告: Same-Clothes Query 为空！请检查 test/B 目录。")


    def _check_before_run(self, dataset_path):
        if not osp.isdir(dataset_path):
            raise RuntimeError(f"'{dataset_path}' is not a valid directory.")
        if not osp.isdir(self.train_path):
            raise RuntimeError(f"Train directory not found at '{self.train_path}'")
        if not osp.isdir(self.test_path):
            raise RuntimeError(f"Test directory not found at '{self.test_path}'")
            
    def _parse_train_path(self, base_path):
        pid_folders = glob.glob(osp.join(base_path, '*'))
        dataset = []
        pid_container = set()
        for pid_folder in pid_folders:
            if not osp.isdir(pid_folder): continue
            try:
                pid = int(osp.basename(pid_folder))
                pid_container.add(pid)
            except ValueError:
                warnings.warn(f"在训练集路径 {pid_folder} 中检测到非数字PID文件夹，已跳过。")
                continue
            img_paths = glob.glob(osp.join(pid_folder, '*.jpg'))
            for img_path in img_paths:
                base_name = osp.basename(img_path)
                cam_char = base_name[0]
                clothes_id = 0 if cam_char in ['A', 'B'] else 1
                cam_id = {'A': 0, 'B': 1, 'C': 2}.get(cam_char, -1)
                if cam_id == -1:
                    warnings.warn(f"在训练集文件 {img_path} 中检测到未知相机/服装前缀 '{cam_char}'。已跳过。")
                    continue
                dataset.append((img_path, pid, cam_id, clothes_id, 'prcc'))
        return dataset, sorted(list(pid_container))

    def _parse_test_path(self, base_path):
        gallery_raw, gallery_pids = self._parse_test_subdir(osp.join(base_path, 'A'))
        query_sc_raw, query_sc_pids = self._parse_test_subdir(osp.join(base_path, 'B'))
        query_cc_raw, query_cc_pids = self._parse_test_subdir(osp.join(base_path, 'C'))

        test_pids = sorted(list(gallery_pids | query_sc_pids | query_cc_pids))

        gallery, _ = self._relabel(gallery_raw, pids=test_pids)
        query_sc, _ = self._relabel(query_sc_raw, pids=test_pids)
        query_cc, _ = self._relabel(query_cc_raw, pids=test_pids)
        
        return query_cc, query_sc, gallery

    def _parse_test_subdir(self, path):
        dataset = []
        pid_container = set()
        if not osp.isdir(path):
            return dataset, pid_container
        cam_char = osp.basename(path)
        clothes_id = 0 if cam_char in ['A', 'B'] else 1
        cam_id = {'A': 0, 'B': 1, 'C': 2}.get(cam_char, -1)
        if cam_id == -1:
            warnings.warn(f"在测试集路径 {path} 中检测到未知的相机/服装目录 '{cam_char}'。")
            return dataset, pid_container
        pid_folders = glob.glob(osp.join(path, '*'))
        for pid_folder in pid_folders:
            if not osp.isdir(pid_folder): continue
            try:
                pid = int(osp.basename(pid_folder))
                pid_container.add(pid)
            except ValueError:
                warnings.warn(f"在测试集路径 {pid_folder} 中检测到非数字PID文件夹，已跳过。")
                continue
            img_paths = glob.glob(osp.join(pid_folder, '*.jpg'))
            for img_path in img_paths:
                dataset.append((img_path, pid, cam_id, clothes_id, 'prcc'))
        return dataset, pid_container

    def _relabel(self, dataset_raw, pids, is_train=False):
        if not dataset_raw: return [], {}
        pid2label = {pid: label for label, pid in enumerate(pids)}
        relabeled_dataset = []
        for item in dataset_raw:
            path, pid, cid, clid, dom = item
            if pid not in pid2label: continue
            relabeled_pid = pid2label[pid]
            if is_train:
                relabeled_dataset.append((path, relabeled_pid, cid, clid, dom, pid)) 
            else:
                relabeled_dataset.append((path, relabeled_pid, cid, clid, dom))
        return relabeled_dataset, pid2label