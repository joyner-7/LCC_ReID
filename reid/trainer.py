from __future__ import print_function, absolute_import
import time
import random
from torch.nn import functional as F
import torch
import torch.nn as nn

from reid.loss.loss_uncertrainty import euclidean_dist
from reid.loss.triplet_loss_transreid import TripletLoss
from .utils.meters import AverageMeter


class Trainer(object):
    def __init__(self, args, model, old_model=None, writer=None, seen_pids=None, style_memory=None):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.old_model = old_model
        self.writer = writer
        # 目前仅在外部统计重复ID时使用 seen_pids，这里保留接口以兼容调用方
        self.seen_pids = seen_pids if seen_pids is not None else set()  # 已见过的ID集合
        # 使用标准 Batch-hard Triplet Loss，支持通过参数扫描 margin
        self.criterion_triple = TripletLoss(margin=args.triplet_margin, normalize_feature=False)
            
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_recon = nn.MSELoss()
        
        self.AF_weight = args.AF_weight
        self.n_sampling = args.n_sampling
        
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if self.old_model is not None:
            self.old_model.to(self.device)
            self.old_model.eval()
            for param in self.old_model.parameters():
                param.requires_grad = False
        self.ranking_loss_proto = nn.MarginRankingLoss(margin=0.3)

    def hard_example_mining_multi_proto(self, dist_mat, labels, proto_labels):
        N = dist_mat.size(0)
        is_pos = labels.unsqueeze(1) == proto_labels.unsqueeze(0)
        is_neg = ~is_pos
        any_pos_prototype_exists = is_pos.any(dim=1)
        valid_indices = torch.where(any_pos_prototype_exists)[0]
        if len(valid_indices) == 0:
            return torch.tensor([]).to(dist_mat.device), torch.tensor([]).to(dist_mat.device)
        dist_mat_valid = dist_mat[valid_indices]
        is_pos_valid = is_pos[valid_indices]
        is_neg_valid = is_neg[valid_indices]
        dist_ap_mat = dist_mat_valid.clone()
        dist_ap_mat[~is_pos_valid] = float('inf')
        dist_ap = dist_ap_mat.min(dim=1)[0]
        dist_an_mat = dist_mat_valid.clone()
        dist_an_mat[~is_neg_valid] = float('inf')
        dist_an = dist_an_mat.min(dim=1)[0]
        return dist_ap, dist_an

    def train(self, epoch, data_loader_train, optimizer, training_phase,
            proto_type=None, train_iters=200, add_num=0):

        self.model.train()

        for m in self.model.module.base.modules():
            if isinstance(m, nn.BatchNorm2d):
                if m.weight.requires_grad == False and m.bias.requires_grad == False:
                    m.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr1 = AverageMeter()
        losses_tr2 = AverageMeter() 
        losses_recon = AverageMeter()
        losses_kd = AverageMeter()
        
        # losses_kd = AverageMeter()     # 知识蒸馏损失

        end = time.time()

        for i in range(train_iters):
            try:
                train_inputs = data_loader_train.next()
            except StopIteration:
                print(f"Data loader for epoch {epoch} exhausted after {i} iterations.")
                break
                
            data_time.update(time.time() - end)

            s_inputs, targets, cids, clothes_ids = self._parse_data(train_inputs)
            targets += add_num

            (s_features_id, f_id, f_bias, reconstructed_map, 
            merge_feat_id, cls_outputs_id, out_var_id, base_out) = self.model(s_inputs)

            # 尝试从 out_var_id 中解析衣服分类输出 (假设结构: [cloth_logits, ...])
            cls_outputs_cloth = None
            if isinstance(out_var_id, (list, tuple)) and len(out_var_id) > 0:
                cls_outputs_cloth = out_var_id[0]

            loss_ce, loss_tp1 = 0, 0
            loss_tp2 = torch.tensor(0.0, device=self.device)
            loss_recon = torch.tensor(0.0, device=self.device)
            
            # loss_kd = torch.tensor(0.0, device=self.device)     # 知识蒸馏损失
            # loss_aug removed

            batch_size = s_features_id.size(0)

            # --- 1. 身份判别损失 ---
            loss_tp1, _ = self.criterion_triple(s_features_id, targets)
            loss_tp1 = loss_tp1 * 1.5
            
            loss_ce = self.criterion_ce(cls_outputs_id[:, 0], targets)
            
            # --- 2. 终身学习损失 (原型相关) ---
            proto_features_base = None # 用于KD的基础原型(不含augment)

            if proto_type and len(proto_type) > 0:
                # 提取纯净原型（不含采样）用于后续KD和Loss计算基础
                proto_features_list_raw = [info['proto'] for pid, info in sorted(proto_type.items())]
                proto_labels_list = sorted(proto_type.keys())
                
                if len(proto_features_list_raw) > 0:
                    # 处理维度
                    clean_list = []
                    for p in proto_features_list_raw:
                        if p.dim() == 1: p = p.unsqueeze(0)
                        clean_list.append(p)
                    
                    proto_features_base = torch.cat(clean_list, dim=0).to(s_features_id.device)
                    proto_labels = torch.tensor(proto_labels_list, dtype=torch.long).to(s_features_id.device)

                    # --- 计算 L_tp2 (原型排序损失) ---
                    if proto_features_base.dim() == 2:
                        # 如果需要采样，为了L_tp2造一个扩充版
                        if self.args.n_sampling > 0:
                            sampled_proto_features = self.gaussian_sample(proto_features_base, self.args.n_sampling)
                            sampled_proto_labels = proto_labels.repeat(self.args.n_sampling)
                            final_proto_features = torch.cat([proto_features_base, sampled_proto_features], dim=0)
                            final_proto_labels = torch.cat([proto_labels, sampled_proto_labels], dim=0)
                        else:
                            final_proto_features = proto_features_base
                            final_proto_labels = proto_labels
                        
                        dist_mat_proto = euclidean_dist(s_features_id, final_proto_features)
                        dist_ap_proto, dist_an_proto = self.hard_example_mining_multi_proto(
                            dist_mat_proto, targets, final_proto_labels)
                        
                        if dist_ap_proto.numel() > 0:
                            y = dist_an_proto.new_ones(dist_an_proto.size())
                            loss_tp2 = self.ranking_loss_proto(dist_an_proto, dist_ap_proto, y)

            # --- 3. 特征解耦损失 ---
            # 3.1 重建损失
            loss_recon_self = self.criterion_recon(reconstructed_map, base_out.detach())
            shuffled_indices = self._select_bias_swap_indices(f_id.detach(), targets)
            f_bias_swapped = f_bias[shuffled_indices]
            f_total_hybrid = torch.cat([f_id, f_bias_swapped], dim=1)
            reconstructed_map_hybrid = self.model.module.decoder(f_total_hybrid)
            loss_recon_cross = self.criterion_recon(reconstructed_map_hybrid, base_out.detach())
            loss_recon = loss_recon_self + loss_recon_cross

            # --- 4. 计算总损失 ---
            repeat_mask = self._build_repeat_mask(targets)
            loss_kd = self._fid_kd_loss(s_inputs, repeat_mask, f_id)

            loss = (loss_ce + loss_tp1 + self.AF_weight * loss_tp2 +
                    self.args.recon_weight * loss_recon +
                    self.args.fid_kd_weight * loss_kd)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_ce.update(loss_ce.item())
            losses_tr1.update(loss_tp1.item())
            losses_tr2.update(loss_tp2.item())
            losses_recon.update(loss_recon.item())
            losses_kd.update(loss_kd.item())

            batch_time.update(time.time() - end)
            end = time.time()

        # --- Epoch结束后打印 ---
        print(
            f"Epoch: [{epoch}] "
            f"Time {batch_time.avg:.3f}  "
            f"L_ce {losses_ce.avg:.3f} | "
            f"L_tp1 {losses_tr1.avg:.3f} | "
            f"L_tp2 {losses_tr2.avg:.3f} | "
            f"L_rec {losses_recon.avg:.3f} | "
            f"L_kd {losses_kd.avg:.3f}"
        )

    def _parse_data(self, inputs):
        imgs, fnames, pids, cids, clothes_ids = inputs
        inputs_on_device = imgs.to(self.device)
        if isinstance(pids, list):
            targets_on_device = torch.tensor(pids).to(self.device)
        else:
            targets_on_device = pids.to(self.device)
        clothes_ids_on_device = None
        if clothes_ids is not None:
            if isinstance(clothes_ids, list):
                clothes_ids_on_device = torch.tensor(clothes_ids).to(self.device)
            else:
                clothes_ids_on_device = clothes_ids.to(self.device)
        return inputs_on_device, targets_on_device, cids, clothes_ids_on_device

    def gaussian_sample(self, proto_features, n_samples):
        C, feature_dim = proto_features.size()
        #noise = torch.randn(C * n_samples, feature_dim).to(proto_features.device) * 0.15 
        sampled_prototypes = proto_features.repeat(n_samples, 1) #+ noise
        return sampled_prototypes

    def _select_bias_swap_indices(self, f_id_detached, targets):
        method = getattr(self.args, "bias_swap_method", "random").lower()
        if method not in {"random", "hard", "semi-hard"}:
            method = "random"

        batch_size = f_id_detached.size(0)
        if batch_size <= 1:
            return torch.arange(batch_size, device=self.device, dtype=torch.long)

        target_col = targets.view(-1)
        diff_mask = target_col.unsqueeze(1) != target_col.unsqueeze(0)
        any_diff = diff_mask.any(dim=1)

        # 若 batch 内不存在可交换的不同 ID 样本，直接返回随机索引避免中断训练
        if not any_diff.any():
            return torch.randperm(batch_size, device=self.device)

        # 基线随机策略：每个样本从不同 ID 候选中随机选一个
        if method == "random":
            return self._random_diff_identity_indices(diff_mask)

        # hard / semi-hard 复用当前 batch 的 f_id，不重复前向传播
        norm_feat = F.normalize(f_id_detached, p=2, dim=1)
        sim_matrix = torch.mm(norm_feat, norm_feat.t())

        selected = torch.empty(batch_size, device=self.device, dtype=torch.long)
        random_fallback = self._random_diff_identity_indices(diff_mask)

        for i in range(batch_size):
            candidates = torch.where(diff_mask[i])[0]
            if candidates.numel() == 0:
                selected[i] = random_fallback[i]
                continue

            cand_sims = sim_matrix[i, candidates]
            sorted_idx = torch.argsort(cand_sims, descending=True)
            sorted_candidates = candidates[sorted_idx]

            if method == "hard":
                selected[i] = sorted_candidates[0]
                continue

            # semi-hard: 取相似度排序后 top30%~60% 区间
            n = sorted_candidates.numel()
            left = int(n * self.args.bias_swap_semihard_low)
            right = int(n * self.args.bias_swap_semihard_high) - 1
            left = max(0, min(left, n - 1))
            right = max(left, min(right, n - 1))

            semi_candidates = sorted_candidates[left:right + 1]
            if semi_candidates.numel() > 0:
                ridx = torch.randint(0, semi_candidates.numel(), (1,), device=self.device)
                selected[i] = semi_candidates[ridx]
            else:
                selected[i] = random_fallback[i]

        return selected

    def _random_diff_identity_indices(self, diff_mask):
        batch_size = diff_mask.size(0)
        indices = torch.empty(batch_size, device=self.device, dtype=torch.long)
        all_idx = torch.arange(batch_size, device=self.device)
        for i in range(batch_size):
            candidates = all_idx[diff_mask[i]]
            if candidates.numel() > 0:
                ridx = torch.randint(0, candidates.numel(), (1,), device=self.device)
                indices[i] = candidates[ridx]
            else:
                # 极端情况下退化为随机位置（允许与自身相同，确保训练不中断）
                indices[i] = torch.randint(0, batch_size, (1,), device=self.device)
        return indices

    def _build_repeat_mask(self, targets):
        if targets is None:
            return torch.tensor([], dtype=torch.bool, device=self.device)
        if not self.seen_pids:
            return torch.zeros_like(targets, dtype=torch.bool, device=self.device)
        target_list = targets.detach().long().cpu().tolist()
        repeat_flags = [pid in self.seen_pids for pid in target_list]
        return torch.tensor(repeat_flags, dtype=torch.bool, device=self.device)

    def _fid_kd_loss(self, inputs, repeat_mask, new_f_id):
        if (self.old_model is None or repeat_mask is None or
                repeat_mask.numel() == 0 or not repeat_mask.any()):
            return torch.tensor(0.0, device=self.device)
        with torch.no_grad():
            (_, old_f_id, _, _, _, _, _, _) = self.old_model(inputs)
        if old_f_id.shape != new_f_id.shape:
            min_len = min(old_f_id.shape[0], new_f_id.shape[0])
            old_f_id = old_f_id[:min_len]
            new_f_id = new_f_id[:min_len]
            repeat_mask = repeat_mask[:min_len]
        if not repeat_mask.any():
            return torch.tensor(0.0, device=self.device)
        return F.mse_loss(new_f_id[repeat_mask], old_f_id[repeat_mask])
