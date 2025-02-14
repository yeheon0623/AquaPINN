# initialize_he.py
import torch
import math

def initialize_he(sz, num_in, dtype=torch.float32):
    """
    使用 He 初始化方法随机生成权重矩阵。
    参数：
      sz: 权重矩阵的尺寸，如 (输出维度, 输入维度)
      num_in: 输入单元数量（用于计算标准差）
      dtype: 数据类型，默认为 torch.float32
    返回：
      一个开启梯度计算的 torch.Tensor
    """
    parameter = torch.randn(sz, dtype=dtype) * math.sqrt(2.0 / num_in)
    parameter.requires_grad_()
    return parameter
