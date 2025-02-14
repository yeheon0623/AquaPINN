# model_2.py
import torch

def fully_connect(x, weights, bias):
    return torch.matmul(weights, x) + bias.unsqueeze(1)

def model_2(parameters2, dlX, dlT):
    """
    第二个神经网络。参数 parameters2 为字典，
    输入变量仅包括 dlX 和 dlT（形状均为 (1, batch_size)）。
    """
    dlXT = torch.cat((dlX, dlT), dim=0)
    num_layers = len(parameters2)
    
    weights = parameters2['fc1']['Weights']
    bias = parameters2['fc1']['Bias']
    dlS = fully_connect(dlXT, weights, bias)
    
    for i in range(2, num_layers + 1):
        layer_name = f'fc{i}'
        dlS = torch.tanh(dlS)
        weights = parameters2[layer_name]['Weights']
        bias = parameters2[layer_name]['Bias']
        dlS = fully_connect(dlS, weights, bias)
    dlS = torch.tanh(dlS)
    return dlS
