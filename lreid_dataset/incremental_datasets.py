# 文件名: incremental_datasets.py

import numpy as np
from PIL import Image
import copy
import os
from prettytable import PrettyTable
from collections import defaultdict, OrderedDict
import warnings
import re

# 定义一个用于遍历目录的函数
def os_walk(folder_dir):
    for root, dirs, files in os.walk(folder_dir):
        files = sorted(files)
        dirs = sorted(dirs)
        return root, dirs, files

# 增量行人重识别样本基类
class IncrementalPersonReIDSamples(object):
    """
    一个通用的增量行人重识别数据集基类。
    """
    
    _junk_pids = []

    def __init__(self, train=None, query=None, gallery=None, **kwargs):
        """
        构造函数。子类在处理完自己的数据后，应调用 super().__init__(...)
        """
        # =====================================================================
        # 关键修改：将 images_dir 改为普通实例属性，在这里初始化。
        # 子类可以在自己的 __init__ 中覆盖这个值。
        self.images_dir = ''
        # =====================================================================

        self.train = train if train is not None else []
        self.query = query if query is not None else []
        self.gallery = gallery if gallery is not None else []

        # 获取训练集中的总人物ID数量，这对于模型初始化很重要
        self.num_train_pids = self.get_n_pids(self.train)

        # 展示数据集的统计信息
        self._show_info()

    def _relabels_incremental(self, samples, label_index):
        """
        重新排序标签，将可能不连续的原始ID映射到从0开始的连续标签。
        """
        if not samples:
            return []
            
        pids = sorted(list(set(s[label_index] for s in samples)))
        pid2label = {pid: i for i, pid in enumerate(pids)}

        new_samples = []
        for sample in samples:
            new_sample = list(sample)
            new_sample[label_index] = pid2label[sample[label_index]]
            new_samples.append(tuple(new_sample))
            
        return new_samples

    def _load_images_path(self, folder_dir, domain_name='market'):
        """
        加载指定文件夹下的所有图像路径，并解析文件名。
        """
        samples = []
        root_path, _, files_name = next(os.walk(folder_dir))
        
        for file_name in files_name:
            if '.jpg' in file_name or '.png' in file_name:
                parsed_info = self._analysis_file_name(file_name)
                if parsed_info:
                    identi_id, camera_id, *rest = parsed_info
                    sample_data = [os.path.join(root_path, file_name), identi_id, camera_id, domain_name] + rest
                    samples.append(sample_data)
        return samples

    def _analysis_file_name(self, file_name):
        """
        (可被子类重写) 默认的文件名解析器。
        """
        match = re.search(r'([-\d]+)_c(\d+)', file_name)
        if match:
            pid, camid = match.groups()
            try:
                return int(pid), int(camid)
            except (ValueError, TypeError):
                warnings.warn(f"Could not parse PID/CID from filename: {file_name}")
                return None
        warnings.warn(f"Default parser failed for filename: {file_name}")
        return None

    def get_n_pids(self, data):
        """
        计算给定数据列表中的独立人物ID（pid）数量。
        """
        if not data: return 0
        return len(set(item[1] for item in data))

    def get_n_cids(self, data):
        """
        计算给定数据列表中的独立摄像头ID（cid）数量。
        """
        if not data: return 0
        return len(set(item[2] for item in data))

    def _show_info(self, if_show=True):
        """
        使用 PrettyTable 库展示数据集的统计信息。
        """
        if not if_show:
            return
            
        try:
            train_imgs, train_pids, train_cams = len(self.train), self.get_n_pids(self.train), self.get_n_cids(self.train)
            query_imgs, query_pids, query_cams = len(self.query), self.get_n_pids(self.query), self.get_n_cids(self.query)
            gallery_imgs, gallery_pids, gallery_cams = len(self.gallery), self.get_n_pids(self.gallery), self.get_n_cids(self.gallery)

            table = PrettyTable()
            table.field_names = ['Split', '# Images', '# PIDs', '# Cams']
            table.title = f'Statistics for: {self.__class__.__name__}'
            table.add_row(['Train', f'{train_imgs:,}', train_pids, train_cams])
            table.add_row(['Query', f'{query_imgs:,}', query_pids, query_cams])
            table.add_row(['Gallery', f'{gallery_imgs:,}', gallery_pids, gallery_cams])
            print(table)
        except Exception as e:
            print(f"Could not display dataset info. Error: {e}")
            self.num_train_pids = 0

def Incremental_combine_test_samples(samples_list):
    '''combine more than one samples (e.g. market.train and duke.train) as a samples'''

    all_gallery, all_query = [], []

    def _generate_relabel_dict(s_list):
        pids_in_list, pid2relabel_dict = [], {}
        for new_label, samples in enumerate(s_list):
            if str(samples[1]) + str(samples[3]) not in pids_in_list:
                pids_in_list.append(str(samples[1]) + str(samples[3]))
        for i, pid in enumerate(sorted(pids_in_list)):
            pid2relabel_dict[pid] = i
        return pid2relabel_dict
    def _replace_pid2relabel(s_list, pid2relabel_dict, pid_dimension=1):
        new_list = copy.deepcopy(s_list)
        for i, sample in enumerate(s_list):
            new_list[i] = list(new_list[i])
            new_list[i][pid_dimension] = pid2relabel_dict[str(sample[pid_dimension])+str(sample[pid_dimension + 2])]
        return new_list

    for samples_class in samples_list:
        all_gallery.extend(samples_class.gallery)
        all_query.extend(samples_class.query)
    pid2relabel_dict = _generate_relabel_dict(all_gallery)

    gallery = _replace_pid2relabel(all_gallery, pid2relabel_dict, pid_dimension=1)
    query = _replace_pid2relabel(all_query, pid2relabel_dict, pid_dimension=1)

    return query, gallery

def Incremental_combine_train_samples(samples_list):
    '''combine more than one samples (e.g. market.train and duke.train) as a samples'''
    all_samples, new_samples = [], []
    all_pid_per_step, all_cid_per_step, output_all_per_step = OrderedDict(), OrderedDict(), defaultdict(dict)
    max_pid, max_cid = 0, 0
    for step, samples in enumerate(samples_list):
        for a_sample in samples:
            img_path = a_sample[0]
            local_pid = a_sample[1]
            try:
                dataset_name = a_sample[3]
                global_pid = max_pid + a_sample[1]
                global_cid = max_cid + int(a_sample[2])
            except:
                print(a_sample)
                assert False
            all_samples.append([img_path, global_pid, global_cid, dataset_name, local_pid])
            if step in all_pid_per_step.keys():
                all_pid_per_step[step].add(global_pid)
            else:
                all_pid_per_step[step] = set()
                all_pid_per_step[step].add(global_pid)

            if step in all_cid_per_step.keys():
                all_cid_per_step[step].add(global_cid)
            else:
                all_cid_per_step[step] = set()
                all_cid_per_step[step].add(global_cid)
        for k, v in all_cid_per_step.items():
            all_cid_per_step[k] = sorted(v)
        max_pid = sum([len(v) for v in all_pid_per_step.values()])
        max_cid = sum([len(v) for v in all_cid_per_step.values()])

    return all_samples, all_pid_per_step, all_cid_per_step

class IncrementalReIDDataSet:
    def __init__(self, samples, total_step, transform):
        self.samples = samples
        self.transform = transform
        self.total_step = total_step

    def __getitem__(self, index):
        this_sample = copy.deepcopy(self.samples[index])
        this_sample = list(this_sample)
        this_sample.append(this_sample[0])
        this_sample[0] = self._loader(this_sample[0])
        if self.transform is not None:
            this_sample[0] = self.transform(this_sample[0])
        this_sample[1] = np.array(this_sample[1])
        return this_sample

    def __len__(self):
        return len(self.samples)

    def _loader(self, img_path):
        return Image.open(img_path).convert('RGB')