# 文件名： CVPR2024-DKP/lreid_dataset/datasets/get_data_loaders.py

import torchvision.transforms as T
import copy
import os.path as osp
import os
from torch.utils.data import DataLoader
from collections import defaultdict
import random
import numpy as np

from reid.utils.data.sampler import RandomMultipleGallerySampler
from reid.utils.data import IterLoader
from reid.utils.data.preprocessor import Preprocessor
import lreid_dataset.datasets as datasets

def build_prcc_clothes_split_loaders(cfg):
    print("="*60)
    print("正在为PRCC构建基于 (ID-服装) 组合和分割的增量数据加载器...")
    print("="*60)
    
    data_dir = cfg.data_dir
    height, width = cfg.height, cfg.width
    batch_size = cfg.batch_size
    workers = cfg.workers
    num_instances = cfg.num_instances
    
    dataset = datasets.create('prcc', root=data_dir)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3), 
        T.RandomHorizontalFlip(p=0.5), 
        T.Pad(10), 
        T.RandomCrop((height, width)), 
        T.ToTensor(), 
        normalizer, 
        T.RandomErasing(p=0.5)
    ])
    
    test_transformer = T.Compose([T.Resize((height, width), interpolation=3), T.ToTensor(), normalizer])
    
    all_test_samples = list(set(dataset.query_cc) | set(dataset.query_sc) | set(dataset.gallery))
    print(f"\n构建完整测试集加载器，包含 {len(all_test_samples)} 张独特的图像。")

    full_test_loader = DataLoader(
        Preprocessor(all_test_samples, root=None, transform=test_transformer),
        batch_size=128, num_workers=workers, shuffle=False, pin_memory=True)
    prcc_test_info = [dataset, full_test_loader]

    pid_clothes_pairs = sorted(list(set([(item[5], item[3]) for item in dataset.train])))
    random.shuffle(pid_clothes_pairs)

    all_task_sets = []
    original_train_data = dataset.train
    num_tasks = 5
    for task_idx in range(num_tasks):
        task_name = f"PRCC_Task_{task_idx + 1}"
        
        start_idx = len(pid_clothes_pairs) * task_idx // num_tasks
        end_idx = len(pid_clothes_pairs) * (task_idx + 1) // num_tasks
        pairs_for_current_task = set(pid_clothes_pairs[start_idx:end_idx])
        
        print(f"\n--- 正在准备任务 {task_idx + 1}/{num_tasks}: '{task_name}' ---")
        
        current_task_data_6_elements = [item for item in original_train_data if (item[5], item[3]) in pairs_for_current_task]
        
        if not current_task_data_6_elements:
            print(f"警告：任务 {task_idx + 1} 没有筛选到任何训练数据，将跳过此任务。")
            continue

        current_task_data = [(item[0], item[1], item[2], item[3], item[4]) for item in current_task_data_6_elements]
        pic_num_in_task = len(current_task_data)
        num_pids_in_task = len(set([item[1] for item in current_task_data]))
        print(f"  本任务总图像数: {pic_num_in_task}, 涉及的ID数: {num_pids_in_task}")
        
        # ==================================================================
        # === 关键修复：强制禁用 Sampler 以恢复 iters=~28 的行为 ===
        rmgs_flag = False  # 强制禁用 Sampler 标志
        sampler = None     # 确保 sampler 为 None
        # ==================================================================
        
        task_batch_size = batch_size # 恢复使用原始 batch_size

        temp_data_loader = DataLoader(
            Preprocessor(current_task_data, root=None, transform=train_transformer),
            batch_size=task_batch_size, num_workers=workers, sampler=sampler,
            shuffle=not rmgs_flag, pin_memory=True, drop_last=True
        )
        
        iters = len(temp_data_loader)
        if iters == 0:
            print("  警告: 计算出的迭代次数为0，此任务将被跳过或不进行训练。")
        else:
            print(f"  每个Epoch的迭代次数 (iters) 自动设定为: {iters}")

        train_loader = IterLoader(temp_data_loader, length=iters)
                       
        init_loader = DataLoader(
            Preprocessor(current_task_data, root=None, transform=test_transformer),
            batch_size=128, num_workers=workers, shuffle=False, pin_memory=True, drop_last=False)

        set_info = [dataset, num_pids_in_task, train_loader, full_test_loader, init_loader, task_name, pic_num_in_task]
        all_task_sets.append(set_info)

    print("\n" + "="*60 + "\n所有PRCC增量任务已准备就绪。\n" + "="*60 + "\n")
    return all_task_sets, prcc_test_info


def build_ltcc_pid_split_loaders(cfg):
    """
    为LTCC构建增量数据加载器的新版本。
    该版本基于 (ID-服装) 组合进行随机划分，以确保任务间的ID重叠。
    """
    print("="*60)
    print("正在为LTCC构建基于 (ID-服装) 组合分割的增量数据加载器 (新策略)...")
    print("="*60)
    
    data_dir = cfg.data_dir
    height, width = cfg.height, cfg.width
    batch_size = cfg.batch_size
    workers = cfg.workers
    
    dataset = datasets.create('ltcc', data_dir)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3), 
        T.RandomHorizontalFlip(p=0.5), 
        T.Pad(10), 
        T.RandomCrop((height, width)), 
        T.ToTensor(), 
        normalizer, 
        T.RandomErasing(p=0.5)
    ])
    
    test_transformer = T.Compose([T.Resize((height, width), interpolation=3), T.ToTensor(), normalizer])
    
    all_test_samples = list(set(dataset.query) | set(dataset.gallery_sc) | set(dataset.gallery_cc))
    print(f"\n构建完整测试集加载器，包含 {len(all_test_samples)} 张独特的图像。")
    
    full_test_loader = DataLoader(
        Preprocessor(all_test_samples, root=None, transform=test_transformer),
        batch_size=128, num_workers=workers, shuffle=False, pin_memory=True)
    ltcc_test_info = [dataset, full_test_loader]

    # --- 步骤 1: 筛选ID (保留您之前的逻辑) ---
    pid_to_clothes = defaultdict(set)
    for item in dataset.train:
        original_pid = item[1]
        clothes_id = item[3]
        pid_to_clothes[original_pid].add(clothes_id)
    
    pid_to_clothes_count = {pid: len(clothes) for pid, clothes in pid_to_clothes.items()}
    allowed_pids = {pid for pid, count in pid_to_clothes_count.items() if count <= 5}
    filtered_train_data = [item for item in dataset.train if item[1] in allowed_pids]
    print(f"已根据“服装数<=5”的规则进行过滤，剩余 {len(filtered_train_data)} 张训练图像。")

    # --- 步骤 2: 以 (PID, 服装ID) 为单位进行划分 ---
    # 提取所有唯一的 (pid, clothes_id) 组合
    pid_clothes_pairs = sorted(list(set([(item[1], item[3]) for item in filtered_train_data])))
    print(f"从过滤后的数据中提取出 {len(pid_clothes_pairs)} 个独特的 (ID-服装) 组合作为划分单位。")

    # 随机打乱组合列表：支持独立 task-order-seed，避免影响其它随机过程
    task_order_seed = getattr(cfg, 'task_order_seed', None)
    if task_order_seed is not None:
        local_rng = random.Random(task_order_seed)
        local_rng.shuffle(pid_clothes_pairs)
        print(f"已按 task_order_seed={task_order_seed} 打乱 (ID-服装) 组合列表。")
    else:
        random.shuffle(pid_clothes_pairs)
        print("已将 (ID-服装) 组合列表随机打乱。")

    all_task_sets = []
    num_tasks = 5
    for task_idx in range(num_tasks):
        task_name = f"LTCC_Clothes_Task_{task_idx + 1}"
        print(f"\n--- 正在准备任务 {task_idx + 1}/{num_tasks}: '{task_name}' ---")
        
        # --- 步骤 3: 切片，为当前任务分配 (ID-服装) 组合 ---
        start_idx = len(pid_clothes_pairs) * task_idx // num_tasks
        end_idx = len(pid_clothes_pairs) * (task_idx + 1) // num_tasks
        pairs_for_current_task = set(pid_clothes_pairs[start_idx:end_idx])

        # --- 步骤 4: 根据分配好的组合，筛选出对应的所有图像 ---
        current_task_data = [item for item in filtered_train_data if (item[1], item[3]) in pairs_for_current_task]
        
        if not current_task_data:
            print(f"警告：任务 {task_idx + 1} 没有筛选到任何训练数据，将跳过此任务。")
            continue

        pic_num_in_task = len(current_task_data)
        num_pids_in_task = len(set([item[1] for item in current_task_data]))
        print(f"  本任务总图像数: {pic_num_in_task}, 涉及的ID数: {num_pids_in_task}")
        
        # --- 步骤 5: 创建数据加载器 (与之前逻辑相同) ---
        task_batch_size = batch_size
        temp_data_loader = DataLoader(
            Preprocessor(current_task_data, root=None, transform=train_transformer),
            batch_size=task_batch_size, num_workers=workers, sampler=None,
            shuffle=True, pin_memory=True, drop_last=True
        )

        iters = len(temp_data_loader)
        if iters == 0:
            print("  警告: 计算出的迭代次数为0，此任务将被跳过或不进行训练。")
        else:
            print(f"  每个Epoch的迭代次数 (iters) 自动设定为: {iters}")
                       
        train_loader = IterLoader(temp_data_loader, length=iters)
                       
        init_loader = DataLoader(
            Preprocessor(current_task_data, root=None, transform=test_transformer),
            batch_size=128, num_workers=workers, shuffle=False, pin_memory=True, drop_last=False)

        set_info = [dataset, num_pids_in_task, train_loader, full_test_loader, init_loader, task_name, pic_num_in_task]
        all_task_sets.append(set_info)

    print("\n" + "="*60 + "\n所有LTCC增量任务的数据加载器已准备就绪 (采用新策略)。\n" + "="*60 + "\n")
    return all_task_sets, ltcc_test_info