#!/bin/bash
# 新功能测试脚本
# 用于快速验证正交约束和F_id知识蒸馏功能

echo "=========================================="
echo "  新功能测试：正交约束 + F_id知识蒸馏"
echo "=========================================="
echo ""

# 配置
DATASET="ltcc"  # 可改为 prcc
GPUS="0,1"
LOG_DIR="/data1/lzj_log/test_new_features_$(date +%Y%m%d_%H%M%S)"

echo "配置信息："
echo "  数据集: $DATASET"
echo "  GPU: $GPUS"
echo "  日志目录: $LOG_DIR"
echo ""

# ========================================
# 实验1: 基线（无新功能）
# ========================================
echo "----------------------------------------"
echo "实验1: 基线（禁用新功能）"
echo "----------------------------------------"
CUDA_VISIBLE_DEVICES=$GPUS python continual_train.py \
    --dataset $DATASET \
    --orth-weight 0.0 \
    --kd-weight 0.0 \
    --epochs0 40 \
    --epochs 30 \
    --logs-dir "${LOG_DIR}/baseline" \
    --middle_test

echo "✓ 实验1完成"
echo ""

# ========================================
# 实验2: 只启用正交约束
# ========================================
echo "----------------------------------------"
echo "实验2: 只启用正交约束"
echo "----------------------------------------"
CUDA_VISIBLE_DEVICES=$GPUS python continual_train.py \
    --dataset $DATASET \
    --orth-weight 0.1 \
    --kd-weight 0.0 \
    --epochs0 40 \
    --epochs 30 \
    --logs-dir "${LOG_DIR}/orth_only" \
    --middle_test

echo "✓ 实验2完成"
echo ""

# ========================================
# 实验3: 只启用知识蒸馏
# ========================================
echo "----------------------------------------"
echo "实验3: 只启用知识蒸馏"
echo "----------------------------------------"
CUDA_VISIBLE_DEVICES=$GPUS python continual_train.py \
    --dataset $DATASET \
    --orth-weight 0.0 \
    --kd-weight 2.0 \
    --epochs0 40 \
    --epochs 30 \
    --logs-dir "${LOG_DIR}/kd_only" \
    --middle_test

echo "✓ 实验3完成"
echo ""

# ========================================
# 实验4: 完整版（两者都启用）
# ========================================
echo "----------------------------------------"
echo "实验4: 完整版（正交约束 + 知识蒸馏）"
echo "----------------------------------------"
CUDA_VISIBLE_DEVICES=$GPUS python continual_train.py \
    --dataset $DATASET \
    --orth-weight 0.1 \
    --kd-weight 2.0 \
    --epochs0 40 \
    --epochs 30 \
    --logs-dir "${LOG_DIR}/full_version" \
    --middle_test

echo "✓ 实验4完成"
echo ""

# ========================================
# 结果汇总
# ========================================
echo "=========================================="
echo "  所有实验完成！"
echo "=========================================="
echo ""
echo "结果保存在: $LOG_DIR"
echo ""
echo "查看各实验的性能："
echo "  1. 基线:         cat ${LOG_DIR}/baseline/log_res.txt"
echo "  2. 正交约束:     cat ${LOG_DIR}/orth_only/log_res.txt"
echo "  3. 知识蒸馏:     cat ${LOG_DIR}/kd_only/log_res.txt"
echo "  4. 完整版:       cat ${LOG_DIR}/full_version/log_res.txt"
echo ""
echo "快速对比命令："
echo "  grep 'CC:' ${LOG_DIR}/*/log_res.txt | tail -4"
echo ""

# 自动生成对比报告
echo "生成对比报告..."
python3 << EOF
import os
import re

log_dir = "$LOG_DIR"
experiments = ["baseline", "orth_only", "kd_only", "full_version"]
names = ["基线", "正交约束", "知识蒸馏", "完整版"]

print("\n" + "="*60)
print("                    性能对比报告")
print("="*60)
print(f"{'实验':<15} {'CC mAP':<10} {'CC R1':<10} {'SC mAP':<10} {'SC R1':<10}")
print("-"*60)

for exp, name in zip(experiments, names):
    log_file = os.path.join(log_dir, exp, "log_res.txt")
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            content = f.read()
            # 提取最后一次CC和SC的结果
            cc_matches = re.findall(r'CC:\s+([\d.]+)\s+([\d.]+)', content)
            sc_matches = re.findall(r'SC:\s+([\d.]+)\s+([\d.]+)', content)
            
            if cc_matches and sc_matches:
                cc_map, cc_r1 = cc_matches[-1]
                sc_map, sc_r1 = sc_matches[-1]
                print(f"{name:<15} {cc_map:<10} {cc_r1:<10} {sc_map:<10} {sc_r1:<10}")
            else:
                print(f"{name:<15} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
    else:
        print(f"{name:<15} {'未完成':<10} {'未完成':<10} {'未完成':<10} {'未完成':<10}")

print("="*60)
print("\n提示：数值越高越好")
print("")
EOF

echo "全部完成！🎉"

