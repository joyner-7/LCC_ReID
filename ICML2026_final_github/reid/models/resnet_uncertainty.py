import copy

import torch.nn as nn
import torchvision.models as models

import torchvision
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal

class GeneralizedMeanPooling(nn.Module):
    r"""应用2D幂平均自适应池化到由多个输入平面组成的输入信号上。
    计算函数为： :math:f(X) = pow(sum(pow(X, p)), 1/p)
    - 当 p = 无穷大时，得到最大池化
    - 当 p = 1 时，得到平均池化
    输出的大小为 H x W，适用于任何输入大小。
    输出特征的数量等于输入平面的数量。
    参数:
    output_size: 目标输出大小，形状为 H x W。
    可以是元组 (H, W) 或单个 H 表示方形图像 H x H
    H 和 W 可以是 int 类型，或者 None，表示大小与输入相同。
    """
    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)  # 幂参数
        self.output_size = output_size  # 输出大小
        self.eps = eps  # 避免数值计算中的零值

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)  # 避免零值并计算幂
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)  # 自适应池化并计算幂的倒数

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.p) + ', ' \
            + 'output_size=' + str(self.output_size) + ')'

class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)

class Normalize(nn.Module):
    def __init__(self, power=2, dim=1):
        super(Normalize, self).__init__()
        self.power = power  # 正则化的幂
        self.dim = dim  # 规范化的维度
    
    def forward(self, x):
        norm = x.pow(self.power).sum(self.dim, keepdim=True).pow(1. / self.power)  # 计算范数
        out = x.div(norm + 1e-4)  # 进行规范化
        return out
        
# === MODIFICATION START: 新增解码器模块 ===
# 这个解码器用于实现自监督偏差分解与重组任务。
# 它的目标是接收一个扁平化的特征向量，并将其重构为 self.base 输出的特征图的形状。
class FeatureDecoder(nn.Module):
    def __init__(self, input_dim=2048, target_channels=1024, target_h=16, target_w=8):
        super(FeatureDecoder, self).__init__()
        self.target_channels = target_channels
        self.target_h = target_h
        self.target_w = target_w

        # 一个线性层将输入特征映射到足以重塑为初始特征图的维度
        self.fc = nn.Linear(input_dim, target_channels * target_h * target_w)
        
        # 使用一系列转置卷积层来优化和重构特征图
        self.decoder_net = nn.Sequential(
            nn.BatchNorm2d(target_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(target_channels, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, target_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        # 通过线性层并重塑为特征图
        x = self.fc(x)
        x = x.view(x.size(0), self.target_channels, self.target_h, self.target_w)
        # 通过解码网络进行优化
        x = self.decoder_net(x)
        return x
# === MODIFICATION END ===

class ResNetSimCLR(nn.Module):
    # === MODIFICATION START: 修改__init__以支持特征解耦 ===
    def __init__(self, base_model='resnet50', id_dim=1536, bias_dim=512, n_sampling=2, pool_len=8, normal_feature=True,
                num_classes=-1, uncertainty=True):
        super(ResNetSimCLR, self).__init__()

        # === MODIFICATION ===: 定义特征维度
        self.id_dim = id_dim
        self.bias_dim = bias_dim
        out_dim = id_dim + bias_dim  # 总特征维度

        # 定义不同的 ResNet 模型
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=True)}
        self.resnet = self._get_basemodel(base_model)  # 获取基础模型
        self.base = nn.Sequential(*list(self.resnet.children())[:-3])  # 去掉最后几层
        
        dim_mlp = 1024 # ResNet50 layer3 output channels
        
        # === MODIFICATION ===: 修改线性层以进行特征解耦
        self.linear_id = nn.Linear(dim_mlp, self.id_dim)    # 线性层用于生成身份特征 F_id
        self.linear_bias = nn.Linear(dim_mlp, self.bias_dim) # 线性层用于生成偏差特征 F_bias
        
        # === MODIFICATION ===: 方差路径现在只为身份特征 F_id 服务
        self.linear_var_id = nn.Linear(dim_mlp, self.id_dim) # 用于计算 F_id 方差的线性层

        self.pool_len = 8  # 池化层的长度
        self.conv_var = nn.Conv2d(dim_mlp, dim_mlp, kernel_size=(pool_len, pool_len), bias=False)  # 用于计算方差的卷积层

        self.n_sampling = n_sampling  # 采样次数
        self.n_samples = torch.Size(np.array([n_sampling, ]))  # 采样尺寸
        self.pooling_layer = GeneralizedMeanPoolingP(3)  # 定义池化层

        # === MODIFICATION ===: 归一化层现在明确作用于身份特征
        self.l2norm_id, self.l2norm_var_id, self.l2norm_sample_id = Normalize(2, 1), Normalize(2, 1), Normalize(2, 2)

        print('using resnet50 as a backbone')  # 打印使用的模型
        '''xkl add'''
        print("##########normalize matchiing feature:", normal_feature)  # 打印是否匹配特征
        self.normal_feature = normal_feature  # 是否规范化特征
        self.uncertainty = uncertainty  # 是否使用不确定性
        
        # === MODIFICATION ===: BNNeck 和 分类器现在都只处理 id_dim 维的身份特征
        self.bottleneck = nn.BatchNorm1d(self.id_dim)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1)
        nn.init.constant_(self.bottleneck.bias, 0)

        self.classifier = nn.Linear(self.id_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
        self.relu = nn.ReLU()

        # === MODIFICATION ===: 实例化新增的解码器
        self.decoder = FeatureDecoder(input_dim=out_dim, target_channels=1024, target_h=16, target_w=8)
    # === MODIFICATION END ===

    def _get_basemodel(self, model_name):
        model = self.resnet_dict[model_name]  # 根据名称获取模型
        return model

    # === MODIFICATION START: 修改forward以实现解耦和重构 ===
    def forward(self, x, fkd=False):
        BS = x.size(0)
        # base_out 是需要被解码器重构的目标特征图
        base_out = self.base(x)
        
        pooled_feat = self.pooling_layer(base_out)
        pooled_feat = pooled_feat.view(pooled_feat.size(0), -1)

        # 1. 特征解耦: 生成身份特征和偏差特征 (都在归一化之前)
        f_id = self.linear_id(pooled_feat)
        f_bias = self.linear_bias(pooled_feat)
        
        # 2. 计算 F_id 的方差路径
        out_var_feat = self.conv_var(base_out)
        out_var_feat = self.pooling_layer(out_var_feat)
        out_var_feat += 1e-4
        out_var_feat = out_var_feat.view(out_var_feat.size(0), -1)
        out_var_feat = self.linear_var_id(out_var_feat)

        # 对身份特征 F_id 进行L2归一化，得到最终用于ReID匹配的 s_features_id
        s_features_id = self.l2norm_id(f_id)

        # 处理方差特征
        var_choice = 'L2'
        if var_choice == 'L2':
            out_var_id = self.l2norm_var_id(out_var_feat)
            out_var_id = self.relu(out_var_id) + 1e-4
        else:
            out_var_id = F.softmax(out_var_feat, dim=1).clone()

        # 3. 对 F_id 进行不确定性采样和分类
        if self.uncertainty:
            BS_current, D_id = s_features_id.size()
            tdist = MultivariateNormal(loc=s_features_id, scale_tril=torch.diag_embed(out_var_id))
            samples = tdist.rsample(self.n_samples)
            samples = self.l2norm_sample_id(samples)

            current_merge_feat = torch.cat((s_features_id.unsqueeze(0), samples), dim=0)
            current_merge_feat_flat = current_merge_feat.reshape(-1, D_id)
            
            # 使用 BatchNorm1d，所以输入需要是 (N, C)
            bn_feat = self.bottleneck(current_merge_feat_flat)
            current_cls_outputs_flat = self.classifier(bn_feat)

            final_merge_feat_id = current_merge_feat_flat.reshape(self.n_sampling + 1, BS_current, D_id).permute(1,0,2)
            final_cls_outputs_id = current_cls_outputs_flat.reshape(self.n_sampling + 1, BS_current, -1).permute(1,0,2)
        else:
            bn_feat = self.bottleneck(s_features_id)
            current_cls_outputs = self.classifier(bn_feat)
            final_cls_outputs_id = current_cls_outputs.unsqueeze(1).repeat(1, self.n_sampling + 1, 1)
            final_merge_feat_id = s_features_id.unsqueeze(1).repeat(1, self.n_sampling + 1, 1)
            
        # 4. 特征重组与解码 (只在训练时需要)
        reconstructed_map = None
        if self.training:
            f_total = torch.cat([f_id, f_bias], dim=1)
            reconstructed_map = self.decoder(f_total)

        # 5. 返回所有需要的组件
        if self.training:
            # 训练时返回所有用于计算损失的组件
            return (
                s_features_id,           # 归一化身份特征 (用于Re-ID损失)
                f_id,                    # 原始身份特征 (用于偏差交换)
                f_bias,                  # 偏差特征 (用于偏差交换)
                reconstructed_map,       # 重构的特征图
                final_merge_feat_id,     # 带采样的身份特征
                final_cls_outputs_id,    # 分类输出
                out_var_id,              # 身份特征的方差
                base_out                 # 原始特征图 (重构目标)
            )
        else: # 评估或特征提取时
            # --- BUG FIX ---
            # 为了与训练时的输出格式保持一致，并为 feature_tools 提供所需的所有特征，
            # 我们返回一个包含 None 占位符的 8 元素元组。
            # 这可以防止在 feature_tools.py 中出现解包错误。
            return (
                s_features_id,           # 归一化身份特征
                f_id,                    # 原始身份特征
                f_bias,                  # 偏差特征
                None,                    # 重构的特征图 (评估时为None)
                final_merge_feat_id,     # 带采样的身份特征
                final_cls_outputs_id,    # 分类输出
                out_var_id,              # 身份特征的方差
                None                     # 原始特征图 (评估时为None)
            )
# === MODIFICATION END ===

if __name__ == '__main__':
    m = ResNetSimCLR(uncertainty=True, num_classes=100, id_dim=1536, bias_dim=512)
    input_tensor = torch.zeros(10, 3, 256, 128)

    m.train()
    outputs = m(input_tensor)
    (s_features_id, f_id, f_bias, reconstructed_map, 
    final_merge_feat_id, final_cls_outputs_id, out_var_id, base_out) = outputs
    
    print("--- Outputs in Training Mode ---")
    print(f"s_features_id (L2-norm ID):  {s_features_id.shape}")
    print(f"f_id (pre-norm ID):          {f_id.shape}")
    print(f"f_bias (Bias Features):      {f_bias.shape}")
    print(f"reconstructed_map:           {reconstructed_map.shape}")
    print(f"base_out (Recon Target):     {base_out.shape}")
    print(f"final_merge_feat_id:         {final_merge_feat_id.shape}")
    print(f"final_cls_outputs_id:        {final_cls_outputs_id.shape}")
    print(f"out_var_id (ID Variance):    {out_var_id.shape}")

    m.eval()
    eval_outputs = m(input_tensor)
    (s_features_id, f_id, f_bias, reconstructed_map, 
    final_merge_feat_id, final_cls_outputs_id, out_var_id, base_out) = eval_outputs
    print("\n--- Output in Eval Mode (consistent 8-tuple format) ---")
    print(f"s_features_id for retrieval: {s_features_id.shape}")
    print(f"f_bias for bias bank:        {f_bias.shape}")
    print(f"final_merge_feat_id for proto: {final_merge_feat_id.shape}")
    print(f"reconstructed_map is None:   {reconstructed_map is None}")