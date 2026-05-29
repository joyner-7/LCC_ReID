# 文件名: __init__.py

from __future__ import absolute_import
import warnings

# 当前主实验仅使用换装行人重识别数据集 LTCC 与 PRCC
from .ltcc import IncrementalSamples4ltcc
from .prcc import IncrementalSamples4prcc

# 数据集工厂字典，将字符串名称映射到对应的类
__factory = {
    'ltcc': IncrementalSamples4ltcc,
    'prcc': IncrementalSamples4prcc,
}


def names():
    """返回所有已注册的数据集名称。"""
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    创建一个数据集实例。

    Args:
        name (str): 数据集的名称 ('ltcc' 或 'prcc').
        root (str): 数据集所在的根目录路径。
        *args, **kwargs: 传递给数据集构造函数的其他参数。

    Returns:
        一个数据集类的实例。
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    """(已弃用) 旧的API，为了兼容性保留。"""
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)
