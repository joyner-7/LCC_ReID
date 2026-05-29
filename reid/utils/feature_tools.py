import torch
import torch.nn.functional as F
from tqdm import tqdm
import collections
import numpy as np

from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data import transforms as T
from torch.utils.data import DataLoader
from .data.sampler import RandomIdentitySampler, MultiDomainRandomIdentitySampler

def extract_features_adv(model, data_loader):
    features_all = []
    labels_all = []
    model.eval()
    with torch.no_grad():
        for i, (imgs,pids) in enumerate(data_loader):
            features = model(imgs.cuda())[0].cpu()
            for feature, pid in zip(features, pids):
                features_all.append(feature)
                labels_all.append(int(pid))
    model.train()
    return features_all, labels_all

def extract_features(model, data_loader):
    features_all = []
    labels_all = []
    fnames_all = []
    camids_all = []
    model.eval()
    with torch.no_grad():
        for i, (imgs, fnames, pids, cids, domains) in enumerate(data_loader):
            features = model(imgs.cuda())[0].cpu()
            for fname, feature, pid, cid in zip(fnames, features, pids, cids):
                features_all.append(feature)
                labels_all.append(int(pid))
                fnames_all.append(fname)
                camids_all.append(cid)
    model.train()
    return features_all, labels_all, fnames_all, camids_all


def extract_features_iter(model, data_loader):
    features_all = []
    labels_all = []
    fnames_all = []
    camids_all = []
    model.eval()
    with torch.no_grad():
        for i in range(len(data_loader)):
            imgs, fnames, pids, cids, domains = data_loader.next()
            features = model(imgs.cuda())[0].cpu()
            for fname, feature, pid, cid in zip(fnames, features, pids, cids):
                features_all.append(feature)
                labels_all.append(int(pid))
                fnames_all.append(fname)
                camids_all.append(cid)
    model.train()
    return features_all, labels_all, fnames_all, camids_all

def initial_classifier(model, data_loader):
    pid2features = collections.defaultdict(list)
    features_all, labels_all, fnames_all, camids_all = extract_features(model, data_loader)
    for feature, pid in zip(features_all, labels_all):
        pid2features[pid].append(feature)
    class_centers = [torch.stack(pid2features[pid]).mean(0) for pid in sorted(pid2features.keys())]
    class_centers = torch.stack(class_centers)
    return F.normalize(class_centers, dim=1).float().cuda()

def obtain_voronoi_loader(dataset,new_labels, add_num=0, batch_size = 32,num_instance=4,workers=8):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.Resize((256, 128), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((256, 128)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])
    voronoi_set=[]
    if not isinstance(new_labels, list):
        new_labels = new_labels.tolist()
    for instance, lablel in zip(dataset.train, new_labels):
        a=(instance[0], lablel, instance[2], instance[3])
        voronoi_set.append(a)
    voronoi_loader = DataLoader(Preprocessor(voronoi_set, root=dataset.images_dir, transform=train_transformer),
                                batch_size=batch_size,num_workers=workers, sampler=RandomIdentitySampler(voronoi_set, num_instance),
                                pin_memory=True, drop_last=True)
    return voronoi_loader, voronoi_set

# --- 【BUG FIX AREA START】 ---
# 重写整个 extract_features_uncertain 函数以修复维度错误
def extract_features_uncertain(model, data_loader, get_mean_feature=False, return_bias=False):
    training_status = model.training
    model.eval()

    all_features_id_list = []
    all_features_bias_list = [] if return_bias else None
    all_pids_list = []
    all_camids_list = []

    with torch.no_grad():
        for data_batch in tqdm(data_loader, desc="Extracting features"):
            # 根据加载器返回的元组长度来解包数据
            if len(data_batch) == 5:
                imgs, _, pids, camids, _ = data_batch
            elif len(data_batch) == 4:
                imgs, pids, camids, _ = data_batch
            else:
                raise ValueError(f"Data loader returned a tuple of unexpected length: {len(data_batch)}")

            imgs = imgs.cuda()
            
            # 由于模型在eval模式下现在也返回8个元素的元组，我们可以用一种统一的方式解包
            # 我们需要 final_merge_feat_id (用于计算ID特征) 和 f_bias (如果需要)
            _, _, f_bias, _, final_merge_feat_id, _, _, _ = model(imgs)

            # 关键修复：
            # final_merge_feat_id 的形状是 (B, S+1, D)，其中S是采样次数
            # 我们取其在采样维度上的均值作为这张图像的鲁棒特征表示
            # 这确保了 mean_id_feat 的形状是 (B, D)，修复了原始bug
            mean_id_feat = final_merge_feat_id.mean(dim=1)

            all_features_id_list.append(mean_id_feat.cpu())
            if return_bias:
                # f_bias 的形状是 (B, D_bias)，可以直接使用
                all_features_bias_list.append(f_bias.cpu())
            
            all_pids_list.extend(pids.cpu().numpy())
            if isinstance(camids, torch.Tensor):
                all_camids_list.extend(camids.cpu().numpy())
            else:
                all_camids_list.extend(camids)

    all_features_id = torch.cat(all_features_id_list, dim=0)
    all_features_bias = torch.cat(all_features_bias_list, dim=0) if return_bias and all_features_bias_list else None

    if get_mean_feature:
        unique_pids = sorted(list(set(all_pids_list)))
        mean_features_id_list = []
        
        all_pids_np = np.array(all_pids_list)

        for pid in unique_pids:
            indices = np.where(all_pids_np == pid)[0]
            # all_features_id 已经是 (N, D) 的形状，所以这里的切片和求均值是正确的
            pid_features_id = all_features_id[indices]
            mean_feature_id = pid_features_id.mean(0)
            mean_features_id_list.append(mean_feature_id)

        mean_features_id = torch.stack(mean_features_id_list, dim=0)
        
        model.train(training_status)
        
        # 返回每个ID的平均特征，以及其他兼容性信息
        # 注意：在get_mean_feature模式下，我们不返回all_features_bias，因为它没有按ID聚合
        return all_features_id, None, all_pids_list, all_camids_list, mean_features_id, unique_pids

    model.train(training_status)

    return all_features_id, all_features_bias, all_pids_list, all_camids_list, None, None
# --- 【BUG FIX AREA END】 ---

def extract_features_voro(model, data_loader, get_mean_feature=False):
    features_all, labels_all, fnames_all, camids_all = extract_features(model, data_loader)
    if get_mean_feature:
        features_collect = {}
        for feature, label in zip(features_all, labels_all):
            if label in features_collect:
                features_collect[label].append(feature)
            else:
                features_collect[label] = [feature]
        labels_named = list(set(labels_all))
        labels_named.sort()
        features_mean=[]
        for x in labels_named:
            if x in features_collect.keys():
                features_mean.append(torch.stack(features_collect[x]).mean(dim=0))
            else:
                features_mean.append(torch.zeros_like(features_all[0]))
        return features_all, labels_all, fnames_all, camids_all, torch.stack(features_mean),labels_named
    else:
        return features_all, labels_all, fnames_all, camids_all

def select_replay_samples(model, dataset, training_phase=0, add_num=0, old_datas=None, select_samples=2,batch_size = 32,workers=8):
    replay_data = []
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformer = T.Compose([
        T.Resize((256, 128), interpolation=3),
        T.ToTensor(),
        normalizer
    ])
    train_transformer = T.Compose([
        T.Resize((256, 128), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((256, 128)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])
    train_loader = DataLoader(Preprocessor(dataset.train, root=dataset.images_dir,transform=transformer),
                              batch_size=batch_size, num_workers=workers, shuffle=True, pin_memory=True, drop_last=False)
    
    features_all, labels_all, fnames_all, camids_all = extract_features(model, train_loader)
    pid2features = collections.defaultdict(list)
    pid2fnames = collections.defaultdict(list)
    pid2cids = collections.defaultdict(list)

    for feature, pid, fname, cid in zip(features_all, labels_all, fnames_all, camids_all):
        pid2features[pid].append(feature)
        pid2fnames[pid].append(fname)
        pid2cids[pid].append(cid)

    labels_all = list(set(labels_all))

    class_centers = [torch.stack(pid2features[pid]).mean(0) for pid in sorted(pid2features.keys())]
    class_centers = F.normalize(torch.stack(class_centers), dim=1)
    select_pids = np.random.choice(labels_all, 250, replace=False)
    for pid in select_pids:
        feautures_single_pid = F.normalize(torch.stack(pid2features[pid]), dim=1, p=2)
        center_single_pid = class_centers[pid]
        simi = torch.mm(feautures_single_pid, center_single_pid.unsqueeze(0).t())
        simi_sort_inx = torch.sort(simi, dim=0)[1][:2]
        for id in simi_sort_inx:
            replay_data.append((pid2fnames[pid][id], pid+add_num, pid2cids[pid][id], training_phase-1))

    if old_datas is None:
        data_loader_replay = DataLoader(Preprocessor(replay_data, root=dataset.images_dir, transform=train_transformer),
                             batch_size=batch_size,num_workers=workers, sampler=RandomIdentitySampler(replay_data, select_samples),
                             pin_memory=True, drop_last=True)
    else:
        replay_data.extend(old_datas)
        data_loader_replay = DataLoader(Preprocessor(replay_data, root=dataset.images_dir, transform=train_transformer),
                             batch_size=training_phase*batch_size,num_workers=workers,
                             sampler=MultiDomainRandomIdentitySampler(replay_data, select_samples),
                             pin_memory=True, drop_last=True)

    return data_loader_replay, replay_data

def get_pseudo_features(data_specific_batch_norm, training_phase, x, domain, unchange=False):
    fake_feat_list = []
    if unchange is False:
        for i in range(training_phase):
            if int(domain[0]) == i:
                data_specific_batch_norm[i].train()
                fake_feat_list.append(data_specific_batch_norm[i](x)[..., 0, 0])
            else:
                data_specific_batch_norm[i].eval()
                fake_feat_list.append(data_specific_batch_norm[i](x)[..., 0, 0])
        for i in range(training_phase):
            data_specific_batch_norm[i].train()
    else:
        for i in range(training_phase):
            data_specific_batch_norm[i].eval()
            fake_feat_list.append(data_specific_batch_norm[i](x)[..., 0, 0])
    return fake_feat_list