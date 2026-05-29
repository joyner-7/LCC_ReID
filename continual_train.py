# 文件名: continual_train.py

from __future__ import print_function, absolute_import
import argparse
import os
import os.path as osp
import sys
import copy
import numpy as np

import pdb

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import TensorDataset, DataLoader
from torch.backends import cudnn
import torch
import torch.nn as nn
import random

from reid.evaluators import Evaluator, extract_features as extract_features_for_eval
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.utils.lr_scheduler import WarmupMultiStepLR
from reid.utils.feature_tools import *
from reid.models.layers import DataParallel
from reid.models.resnet_uncertainty import ResNetSimCLR
from reid.trainer import Trainer # 确保 trainer.py 也在同一个目录下
from torch.utils.tensorboard import SummaryWriter

from lreid_dataset.datasets.get_data_loaders import build_ltcc_pid_split_loaders, build_prcc_clothes_split_loaders
from tools.Logger_results import Logger_res


def fuse_and_calibrate_intra_id_prototypes(old_info, new_proto, new_count):
    old_proto = old_info['proto']
    old_count = old_info['count']

    if old_count >= new_count:
        base_proto, n_base = old_proto, old_count
        query_proto, n_query = new_proto, new_count
    else:
        base_proto, n_base = new_proto, new_count
        query_proto, n_query = old_proto, old_count
    
    bias_vector = query_proto - base_proto
    
    total_samples_for_alpha = n_query + n_base
    if total_samples_for_alpha > 0:
        alpha = n_query / total_samples_for_alpha
    else:
        alpha = 0

    calibrated_query_proto = query_proto - alpha * bias_vector
    
    total_count = old_count + new_count
    
    if old_count >= new_count:
        fused_proto = (old_count * old_proto + new_count * calibrated_query_proto) / total_count
    else:
        fused_proto = (new_count * new_proto + old_count * calibrated_query_proto) / total_count

    return fused_proto, total_count


def main():
    args = parser.parse_args()

    if args.seed is not None:
        print("setting the seed to",args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    log_name = 'log.txt'
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, log_name))
    else:
        log_dir = osp.dirname(args.test_folder)
        sys.stdout = Logger(osp.join(log_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))
    log_res_name='log_res.txt'
    logger_res=Logger_res(osp.join(args.logs_dir, log_res_name))

    if args.dataset == 'ltcc':
        print("为LTCC基于PID分组的增量学习进行设置。")
        all_train_sets, test_info = build_ltcc_pid_split_loaders(args)
    elif args.dataset == 'prcc':
        print("为PRCC基于(ID-服装)组合的增量学习进行设置。")
        all_train_sets, test_info = build_prcc_clothes_split_loaders(args)
    else:
        raise ValueError(f"错误：不支持的数据集 '{args.dataset}'。请选择 'ltcc' 或 'prcc'。")
    
    dataset_obj_for_meta = all_train_sets[0][0]
    total_num_pids = dataset_obj_for_meta.num_train_pids
    print(f"检测到 {args.dataset.upper()} 数据集中总共有 {total_num_pids} 个训练ID。")
    print(">>> 采用动态分类器策略：初始只初始化当前任务所需的类别数 <<<")
    
    all_test_only_sets = [test_info]

    visible_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if visible_gpu_count < 1:
        raise RuntimeError("未检测到可见的CUDA设备。请先设置可用GPU，例如 CUDA_VISIBLE_DEVICES=0")
    device_ids = list(range(visible_gpu_count))
    main_device = torch.device(f'cuda:{device_ids[0]}')
    print(f"检测到可见GPU数量: {visible_gpu_count}, DataParallel设备索引: {device_ids}")

    # 初始初始化时，只给一个最小的 num_classes (或者第一个任务的ID数，如果能获取到)
    # 为了安全起见，先用第一个任务的ID数量来初始化，或者干脆 0 (如果模型支持)
    # 这里我们先获取第一个任务的ID数量
    _, _, _, _, _, _, first_task_pic_num = all_train_sets[0]
    # 由于 num_train_pids 是总数，我们需要知道第一个任务的具体ID数
    # 但 dataset_obj 里并没有直接提供每个任务的ID数，只能在训练循环中处理
    # 我们可以先初始化一个空的或小的分类器，然后在 train_dataset 里动态调整
    
    # 暂时先用 total_num_pids 初始化，后续在 train_dataset 中我们手动 resize 它
    # 或者更彻底一点，直接修改 num_classes 为 0，然后在 loop 中处理
    # 为了兼容 DataParallel 的参数传递，我们还是先初始化一个
    
    model=ResNetSimCLR(num_classes=1, # 初始设为1，后续动态扩展
                       uncertainty=True,
                       n_sampling=args.n_sampling,
                       id_dim=args.id_dim,
                       bias_dim=args.bias_dim)
    model.to(main_device)
    
    model = DataParallel(model, device_ids=device_ids)
    writer = SummaryWriter(log_dir=args.logs_dir)
    
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['mAP']
        print("=> Start epoch {}  best mAP {:.1%}".format(start_epoch, best_mAP))
   
    out_channel = args.id_dim
    
    proto_type = {} 
    seen_pids = set()  # 记录所有已经见过的ID
    
    num_total_tasks = len(all_train_sets)
    print(f"总共将进行 {num_total_tasks} 个增量任务的训练。")

    for set_index in range(num_total_tasks):
        current_task_name = all_train_sets[set_index][5]
        print(f"\n{'='*20}>> 开始增量任务 {set_index + 1}/{num_total_tasks}: {current_task_name} <<{'='*20}")
        
        model_old = copy.deepcopy(model) if set_index > 0 else None
        
        model = train_dataset(args, proto_type, all_train_sets, all_test_only_sets, set_index, model, out_channel,
                                            writer, logger_res=logger_res,
                                            model_old=model_old, seen_pids=seen_pids)
        
        print(f"\n===== [评估点 1] 刚刚在 '{current_task_name}' 上训练完成后的性能 =====")
        test_model(model, all_train_sets, all_test_only_sets, set_index, logger_res=logger_res, dataset_name=args.dataset)

        _, _, _, _, init_loader_current, _, pic_num_current_task = all_train_sets[set_index]
        
        print(f"\n>> 更新任务 '{current_task_name}' 的原型到全局原型库...")
        _, _, _, _, features_mean_current, pids_current = extract_features_uncertain(
            model, init_loader_current, get_mean_feature=True, return_bias=False
        )

        updated_pids = []
        num_unique_pids_in_task = len(set(pids_current))
        n_new_per_pid = pic_num_current_task / float(num_unique_pids_in_task) if num_unique_pids_in_task > 0 else 0

        for i, pid in enumerate(pids_current):
            new_proto = features_mean_current[i].unsqueeze(0)
            
            if pid in proto_type:
                old_proto_info = proto_type[pid]
                fused_proto, total_count = fuse_and_calibrate_intra_id_prototypes(
                    old_proto_info, new_proto, n_new_per_pid
                )
                proto_type[pid] = {'proto': fused_proto, 'count': total_count}
            else:
                proto_type[pid] = {'proto': new_proto, 'count': n_new_per_pid}
            updated_pids.append(pid)
            
        print(f"原型库更新完毕。共更新/校准/添加了 {len(set(updated_pids))} 个ID的原型。")
        print(f"当前原型库中总ID数: {len(proto_type)}")
        
        # 更新已见过的ID集合（用于 F_id 知识蒸馏）
        current_task_pids = set(updated_pids)
        new_pids = current_task_pids - seen_pids
        repeated_pids = current_task_pids & seen_pids
        seen_pids.update(current_task_pids)
        print(f">> 当前任务新增ID数: {len(new_pids)}, 重复ID数: {len(repeated_pids)}")
        print(f">> 累计已见过的ID总数: {len(seen_pids)}")

        if set_index > 0:
            pic_num_current = all_train_sets[set_index][6]
            pic_num_total = sum([all_train_sets[i][6] for i in range(set_index + 1)])
            alpha = pic_num_current / pic_num_total
            print(f"执行模型融合: alpha (当前任务权重) = {alpha:.3f}")
            
            model = linear_combination(args, model, model_old,0.5)
            
            print(f"\n===== [评估点 2] 在与旧模型融合后的性能 (任务 {set_index + 1}) =====")
            test_model(model, all_train_sets, all_test_only_sets, set_index, logger_res=logger_res, dataset_name=args.dataset)    
    
    print(f"\n所有 {num_total_tasks} 个增量学习任务全部完成！")


def train_dataset(args, proto_type, all_train_sets, all_test_only_sets, set_index, model, out_channel, writer, logger_res=None, model_old=None, seen_pids=None):
    if len(proto_type) > 0:
        print(f"训练任务 {set_index + 1} 时，使用包含 {len(proto_type)} 个ID的单一原型库进行偏差对齐引导。")

    dataset, num_classes, train_loader, test_loader, init_loader, name, picnum = all_train_sets[set_index]
    Epochs = args.epochs0 if 0 == set_index else args.epochs

    # --- 动态分类器扩展逻辑 ---
    # 1. 获取当前任务中出现的所有 PID
    print("正在分析当前任务的 ID 分布以动态调整分类器...")
    current_task_pids = set()
    # 遍历 init_loader 快速获取所有 PID (不需要全部遍历，只需要 unique pids)
    # init_loader 通常是 Shuffle=False 的
    temp_pids = []
    for _, _, pids, _, _ in init_loader:
        temp_pids.extend(pids.tolist())
    current_task_pids = set(temp_pids)
    max_pid_in_task = max(current_task_pids)
    print(f"当前任务 '{name}' 包含 {len(current_task_pids)} 个唯一 ID，最大 PID 为 {max_pid_in_task}")

    # 2. 检查是否需要扩展分类器
    # 注意：model 是 DataParallel，需要操作 model.module
    current_classifier_size = model.module.classifier.weight.size(0)
    required_size = max_pid_in_task + 1 # PID 是从 0 开始的

    if required_size > current_classifier_size:
        print(f"扩充分类器: {current_classifier_size} -> {required_size}")
        old_weight = model.module.classifier.weight.data
        # 创建新层 (Bias=False)
        new_classifier = nn.Linear(model.module.id_dim, required_size, bias=False)
        # 初始化新层权重 (随机)
        nn.init.normal_(new_classifier.weight, std=0.001)
        # 复制旧权重
        new_classifier.weight.data[:current_classifier_size] = old_weight
        # 替换模型中的分类器
        model.module.classifier = new_classifier
        model.module.classifier.cuda() # 确保在 GPU 上
        
        # 重新包装 DataParallel (虽然直接修改 module 属性通常可行，但为了保险)
        # model = DataParallel(model.module, device_ids=[0,1]) 
        # 注意：在函数内部重新包装可能会断开外部引用的连接，这里直接修改 module 属性通常对 DataParallel 是生效的，
        # 只要后续 optimizer 是重新初始化的。
        
    else:
        print(f"当前分类器大小 ({current_classifier_size}) 足够覆盖当前任务 (最大PID {max_pid_in_task})，无需扩充。")

    # 3. 初始化新 ID 的权重 (类似于 Task 1 的中心初始化)
    # 我们只对新增的 ID 或者 Task 1 的 ID 进行中心初始化
    print("使用 extract_features_uncertain 计算类别中心以初始化/更新分类器权重...")
    _, _, _, _, class_centers, pids_of_centers = extract_features_uncertain(model, init_loader, get_mean_feature=True)
    
    print(f"将 {len(pids_of_centers)} 个ID的中心特征更新到分类器。")
    for i, pid in enumerate(pids_of_centers):
        # 只有当权重是新创建的（或者我们希望用中心特征覆盖随机初始化）时才更新
        # 这里我们选择：如果是 Task 1，全部初始化；如果是后续 Task，只初始化新扩充的部分
        # 或者更激进一点：总是用特征中心作为该 ID 的权重初始值（这在 ReID 中是常见的做法，即 Proxy）
        model.module.classifier.weight.data[pid] = class_centers[i].cuda()
        
    model.cuda()
    add_num = 0
    
    # 4. 重新初始化优化器 (关键！因为参数对象变了)
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=args.momentum)
        
    lr_scheduler = WarmupMultiStepLR(optimizer, args.milestones, gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)
    
    if seen_pids is None:
        seen_pids = set()
    trainer = Trainer(args, model, old_model=model_old, writer=writer, seen_pids=seen_pids)

    print(f'####### 在 {name} 上开始训练 #######')
    for epoch in range(0, Epochs):
        train_loader.new_epoch()
        trainer.train(epoch, train_loader, optimizer, training_phase=set_index + 1,
                      proto_type=proto_type, train_iters=len(train_loader), add_num=add_num)
        lr_scheduler.step()       
       
        if ((epoch + 1) % args.eval_epoch == 0 or epoch + 1 == Epochs):
            mAP = 0.
            if args.middle_test:
                print(f"\n--- Epoch {epoch + 1} 中期评估 ---")
                mAP = test_model(model, all_train_sets, all_test_only_sets, set_index, logger_res=logger_res, dataset_name=args.dataset)
          
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': mAP,
            }, True, fpath=osp.join(args.logs_dir, '{}_checkpoint.pth.tar'.format(name)))
  
    return model

def test_model(model, all_train_sets, all_test_sets, set_index, logger_res=None, dataset_name='ltcc'):
    model.eval()
    evaluator = Evaluator(model)
    dataset_obj, test_loader = all_test_sets[0]
    current_task_name = all_train_sets[set_index][5]
    print(f"\n在{dataset_name.upper()}完整测试集上评估模型（在 '{current_task_name}' 上训练后）...")
    print("正在提取所有测试图像的特征...")
    features, _ = extract_features_for_eval(model, test_loader)
    
    if dataset_name == 'prcc':
        print("\n===== 评估设置: PRCC Same-Clothes (Query B vs Gallery A) =====")
        if not hasattr(dataset_obj, 'query_sc') or not dataset_obj.query_sc:
            print("错误：PRCC测试集的 same-clothes query (query_sc) 为空，跳过SC评估。")
            mAP_sc, R1_sc = 0.0, 0.0
        else:
            R1_sc, mAP_sc = evaluator.evaluate(data_loader=None, query=dataset_obj.query_sc, gallery=dataset_obj.gallery, cmc_flag=True, pre_features=features)
        results_sc_str = f"| Same-Clothes Setting | mAP: {mAP_sc*100:.1f}% / Rank-1: {R1_sc*100:.1f}% |"
        print(results_sc_str)

        print("\n===== 评估设置: PRCC Cross-Clothes (Query C vs Gallery A) =====")
        if not hasattr(dataset_obj, 'query_cc') or not dataset_obj.query_cc or not dataset_obj.gallery:
            print("错误：PRCC测试集的 cross-clothes query 或 gallery 为空，跳过CC评估。")
            mAP_cc, R1_cc = 0.0, 0.0
        else:
            R1_cc, mAP_cc = evaluator.evaluate(data_loader=None, query=dataset_obj.query_cc, gallery=dataset_obj.gallery, cmc_flag=True, pre_features=features)
        results_cc_str = f"| Cross-Clothes Setting | mAP: {mAP_cc*100:.1f}% / Rank-1: {R1_cc*100:.1f}% |"
        print(results_cc_str)
        
        if logger_res:
            logger_res.append(f"\n--- 评估节点：任务 '{current_task_name}' (set_index: {set_index}) 之后 ---")
            logger_res.append(results_sc_str)
            logger_res.append(results_cc_str)
            logger_res.append(f"PRCC_SC:\t{mAP_sc*100:.1f}\t{R1_sc*100:.1f}")
            logger_res.append(f"PRCC_CC:\t{mAP_cc*100:.1f}\t{R1_cc*100:.1f}")
            
        return mAP_cc

    elif dataset_name == 'ltcc':
        print("\n===== 评估设置: Standard / General (SC) =====")
        if not dataset_obj.query or not hasattr(dataset_obj, 'gallery_sc') or not dataset_obj.gallery_sc:
            print("错误：LTCC测试集的 query 或 gallery_sc 为空，跳过SC评估。")
            mAP_sc, R1_sc = 0.0, 0.0
        else:
            R1_sc, mAP_sc = evaluator.evaluate(data_loader=None, query=dataset_obj.query, gallery=dataset_obj.gallery_sc, cmc_flag=True, pre_features=features)
        results_sc_str = f"| SC Setting | mAP: {mAP_sc*100:.1f}% / Rank-1: {R1_sc*100:.1f}% |"
        print(results_sc_str)

        print("\n===== 评估设置: Cloth-Changing (CC) =====")
        if not dataset_obj.query or not hasattr(dataset_obj, 'gallery_cc') or not dataset_obj.gallery_cc:
            print("错误：LTCC测试集的 query 或 gallery_cc 为空，跳过CC评估。")
            mAP_cc, R1_cc = 0.0, 0.0
        else:
            R1_cc, mAP_cc = evaluator.evaluate(data_loader=None, query=dataset_obj.query, gallery=dataset_obj.gallery_cc, cmc_flag=True, pre_features=features)
        results_cc_str = f"| CC Setting | mAP: {mAP_cc*100:.1f}% / Rank-1: {R1_cc*100:.1f}% |"
        print(results_cc_str)

        performance_gap = mAP_sc - mAP_cc
        gap_str = f"性能差距 (mAP_SC - mAP_CC): {performance_gap*100:.1f}%"
        print(f"\n{gap_str}")

        if logger_res:
            logger_res.append(f"\n--- 评估节点：任务 '{current_task_name}' (set_index: {set_index}) 之后 ---")
            logger_res.append(results_sc_str)
            logger_res.append(results_cc_str)
            logger_res.append(gap_str)
            logger_res.append(f"SC:\t{mAP_sc*100:.1f}\t{R1_sc*100:.1f}")
            logger_res.append(f"CC:\t{mAP_cc*100:.1f}\t{R1_cc*100:.1f}")
            
        return mAP_cc
    else:
        print(f"警告: 数据集 '{dataset_name}' 没有对应的评估协议。")
        return 0.0

def linear_combination(args, model, model_old, alpha, model_old_id=-1):
    model_old_state_dict = model_old.state_dict()
    model_state_dict = model.state_dict()
    model_new = copy.deepcopy(model)
    model_new_state_dict = model_new.state_dict()
    for k, v in model_state_dict.items():
        if k in model_old_state_dict and model_old_state_dict[k].shape == v.shape:
            model_new_state_dict[k] = alpha * v + (1 - alpha) * model_old_state_dict[k]
        else:
            print(f"警告：参数 {k} 尺寸不匹配或在旧模型中不存在，将直接采用新模型的权重。")
            model_new_state_dict[k] = v
    model_new.load_state_dict(model_new_state_dict)
    return model_new

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Continual training for lifelong person re-identification")
    parser.add_argument('--dataset', type=str, default='prcc', choices=['ltcc', 'prcc'], help="选择要使用的数据集: 'ltcc' 或 'prcc'")
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4, help="for RandomMultipleGallerySampler")
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'])
    parser.add_argument('--lr', type=float, default=0.008)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[30], help='milestones for LR decay')
    parser.add_argument('--resume', type=str, default=None, metavar='PATH')
    parser.add_argument('--evaluate', action='store_true', help="evaluation only")
    parser.add_argument('--epochs0', type=int, default=40, help="Epochs for the first clothing task")
    parser.add_argument('--epochs', type=int, default=30, help="Epochs for subsequent clothing tasks")
    parser.add_argument('--eval_epoch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=3)
    parser.add_argument('--data-dir', type=str, metavar='PATH', default='/data0/data_lzj/')
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=osp.join('/data1/lzj_log/ICML_2026/chaocanshu/'))
    parser.add_argument('--test_folder', type=str, default='', help="test the models in a file")
    parser.add_argument('--middle_test', action='store_true', help="test during middle step")
    
    parser.add_argument('--id-dim', type=int, default=1536, help="Dimension of the identity feature space")
    parser.add_argument('--bias-dim', type=int, default=512, help="Dimension of the bias feature space")
    parser.add_argument('--recon-weight', type=float, default=1.2, help="Weight for the feature reconstruction loss")

    parser.add_argument('--lifelong-warmup-epochs', type=int, default=5, help="Number of epochs to warm up feature learning before applying lifelong losses")
    
    parser.add_argument('--AF_weight', default=1.5, type=float, help="anti-forgetting weight")
    parser.add_argument('--fid-kd-weight', default=0.5, type=float,
                        help="weight for repeated f_id knowledge distillation")
    parser.add_argument('--triplet-margin', default=0.3, type=float,
                        help="margin for TripletLoss")
    parser.add_argument('--bias-swap-method', default='random', type=str,
                        choices=['random', 'hard', 'semi-hard'],
                        help="bias swapping sampling strategy")
    parser.add_argument('--bias-swap-semihard-low', default=0.3, type=float,
                        help="semi-hard lower percentile in similarity ranking")
    parser.add_argument('--bias-swap-semihard-high', default=0.6, type=float,
                        help="semi-hard upper percentile in similarity ranking")
    parser.add_argument('--task-order-seed', default=None, type=int,
                        help="seed used only for LTCC task order shuffling")
    parser.add_argument('--n_sampling', default=0, type=int, help="number of sampling by Gaussian distribution (deprecated, set to 0)")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    
    main()

# 示例运行命令:
# CUDA_VISIBLE_DEVICES=3,4 python continual_train.py --dataset prcc
# CUDA_VISIBLE_DEVICES=0,1 python continual_train.py --dataset ltcc --recon-mode pixel --epochs0 40 --logs-dir logs/verify_pixel