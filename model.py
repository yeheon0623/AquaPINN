# model.py
import torch

def fully_connect(x, weights, bias):
    """
    实现 MATLAB 中的 fullyconnect：线性变换 weights*x + bias
    假设 x 的尺寸为 (输入维度, batch_size)，weights 尺寸为 (输出维度, 输入维度)，bias 尺寸为 (输出维度,)
    """
    return torch.matmul(weights, x) + bias.unsqueeze(1)

def model(parameters, dlX, dlZ, dlT):
    """
    第一个神经网络。参数 parameters 为字典（例如包含 'fc1','fc2',...），
    输入 dlX, dlZ, dlT 均为形状 (1, batch_size) 的张量。
    """
    # 拼接输入（竖直拼接）
    dlXZT = torch.cat((dlX, dlZ, dlT), dim=0)
    num_layers = len(parameters)
    
    # 第一层
    weights = parameters['fc1']['Weights']
    bias = parameters['fc1']['Bias']
    dlH = fully_connect(dlXZT, weights, bias)
    
    # 后续各层：先 tanh 激活，再全连接
    for i in range(2, num_layers + 1):
        layer_name = f'fc{i}'
        dlH = torch.tanh(dlH)
        weights = parameters[layer_name]['Weights']
        bias = parameters[layer_name]['Bias']
        dlH = fully_connect(dlH, weights, bias)
    dlH = torch.tanh(dlH)
    return dlH
