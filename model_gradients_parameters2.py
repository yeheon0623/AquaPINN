# model_gradients_parameters2.py
import torch
from model import model
from model_2 import model_2

def mse_loss(pred, target):
    return torch.mean((pred - target) ** 2)

def model_gradients_parameters2(parameters, parameters2, dlX, dlZ, dlT, S0IC):
    """
    Compute the gradients for the second neural network (parameters2).
    Assume S0IC has shape (3, batch_size): the first row represents x,
    the second row represents t, and the third row is the target z.
    """
    dlS = model_2(parameters2, S0IC[0:1, :], S0IC[1:2, :])
    lossS = mse_loss(dlS, S0IC[2:3, :])
    
    dlS = model_2(parameters2, dlX, dlT)
    H = model(parameters, dlX, dlS, dlT)
    loss_surface = mse_loss(H - dlS, torch.zeros_like(H))
    
    loss2 = lossS + loss_surface
    loss2.backward(retain_graph=True)
    
    gradients2 = {}
    for key, layer in parameters2.items():
        gradients2[key] = {
            'Weights': layer['Weights'].grad.clone() if layer['Weights'].grad is not None else None,
            'Bias': layer['Bias'].grad.clone() if layer['Bias'].grad is not None else None
        }
    
    return gradients2, loss2, parameters2
