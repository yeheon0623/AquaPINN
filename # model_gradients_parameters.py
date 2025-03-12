# model_gradients_parameters.py
import torch
from model import model

def mse_loss(pred, target):
    return torch.mean((pred - target)**2)

def model_gradients_parameters(parameters,
                               parameters2,
                               dlX, dlZ, dlT,
                               dlX0, dlZ0, dlT0, dlH0,
                               dlX0BC3, dlX0BC4, dlX0BC5, dlX0BC6,
                               dlZ0BC3, dlZ0BC4, dlZ0BC5, dlZ0BC6,
                               dlT0BC3, dlT0BC4, dlT0BC5, dlT0BC6,
                               T, S):
    H = model(parameters, dlX, dlZ, dlT)
    H_sum = H.sum()
    Hx = torch.autograd.grad(H_sum, dlX, create_graph=True)[0]
    Hz = torch.autograd.grad(H_sum, dlZ, create_graph=True)[0]
    Ht = torch.autograd.grad(H_sum, dlT, create_graph=True)[0]
    
    dH_Hx = torch.autograd.grad(Hx.sum(), dlX, create_graph=True)[0]
    dH_Hz = torch.autograd.grad(Hz.sum(), dlZ, create_graph=True)[0]
    
    Tx = T * dH_Hx
    Tz = T * dH_Hz
    SH = S * Ht
    f = Tx + Tz - SH
    lossF = mse_loss(f, torch.zeros_like(f))
    
    dlH0Pred = model(parameters, dlX0, dlZ0, dlT0)
    lossU = mse_loss(dlH0Pred, dlH0)
    
    dlH0Pred_IMP1 = model(parameters, dlX0BC3, dlZ0BC3, dlT0BC3)
    gradients_Z1 = torch.autograd.grad(dlH0Pred_IMP1.sum(), dlZ0BC3, create_graph=True)[0]
    lossZ1 = mse_loss(gradients_Z1, torch.zeros_like(gradients_Z1))
    
    dlH0Pred_IMP2 = model(parameters, dlX0BC4, dlZ0BC4, dlT0BC4)
    gradients_Z2 = torch.autograd.grad(dlH0Pred_IMP2.sum(), dlZ0BC4, create_graph=True)[0]
    lossZ2 = mse_loss(gradients_Z2, torch.zeros_like(gradients_Z2))
    
    dlH0Pred_IMP3 = model(parameters, dlX0BC5, dlZ0BC5, dlT0BC5)
    gradients_X1 = torch.autograd.grad(dlH0Pred_IMP3.sum(), dlX0BC5, create_graph=True)[0]
    lossX1 = mse_loss(gradients_X1, torch.zeros_like(gradients_X1))
    
    dlH0Pred_IMP4 = model(parameters, dlX0BC6, dlZ0BC6, dlT0BC6)
    gradients_X2 = torch.autograd.grad(dlH0Pred_IMP4.sum(), dlX0BC6, create_graph=True)[0]
    lossX2 = mse_loss(gradients_X2, torch.zeros_like(gradients_X2))
    
    loss = lossF + lossU + lossZ1 + lossZ2 + lossX1 + lossX2
    
    loss.backward(retain_graph=True)
    gradients = {}
    for key, layer in parameters.items():
        gradients[key] = {
            'Weights': layer['Weights'].grad.clone() if layer['Weights'].grad is not None else None,
            'Bias': layer['Bias'].grad.clone() if layer['Bias'].grad is not None else None
        }
    return gradients, loss, parameters
