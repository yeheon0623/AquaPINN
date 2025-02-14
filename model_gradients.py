# model_gradients.py
import torch
from model import model
from model_2 import model_2

def mse_loss(pred, target):
    return torch.mean((pred - target) ** 2)

def model_gradients(parameters, parameters2,
                    dlX, dlZ, dlT,
                    dlX0, dlZ0, dlT0, dlH0,
                    S0IC,
                    dlX0BC3, dlX0BC4, dlX0BC5, dlX0BC6,
                    dlZ0BC3, dlZ0BC4, dlZ0BC5, dlZ0BC6,
                    dlT0BC3, dlT0BC4, dlT0BC5, dlT0BC6,
                    T, S):
    """
    Compute the gradients and loss for the coupled neural networks.
    T and S are constant tensors used in the groundwater flow equation.
    """
    # Compute predicted head using the first neural network
    H = model(parameters, dlX, dlZ, dlT)
    H_sum = H.sum()
    Hx = torch.autograd.grad(H_sum, dlX, create_graph=True)[0]
    Hz = torch.autograd.grad(H_sum, dlZ, create_graph=True)[0]
    Ht = torch.autograd.grad(H_sum, dlT, create_graph=True)[0]
    
    # Compute second-order derivatives for x and z
    dH_Hx = torch.autograd.grad(Hx.sum(), dlX, create_graph=True)[0]
    dH_Hz = torch.autograd.grad(Hz.sum(), dlZ, create_graph=True)[0]
    
    # Formulate the residual of the groundwater flow equation
    Tx = T * dH_Hx
    Tz = T * dH_Hz
    SH = S * Ht
    f = Tx + Tz - SH
    lossF = mse_loss(f, torch.zeros_like(f))
    
    # Compute the loss for initial and Dirichlet conditions
    dlH0Pred = model(parameters, dlX0, dlZ0, dlT0)
    lossU = mse_loss(dlH0Pred, dlH0)
    
    # Compute the loss for Neumann conditions (for z-direction)
    dlH0Pred_IMP1 = model(parameters, dlX0BC3, dlZ0BC3, dlT0BC3)
    gradients_Z1 = torch.autograd.grad(dlH0Pred_IMP1.sum(), dlZ0BC3, create_graph=True)[0]
    lossZ1 = mse_loss(gradients_Z1, torch.zeros_like(gradients_Z1))
    
    dlH0Pred_IMP2 = model(parameters, dlX0BC4, dlZ0BC4, dlT0BC4)
    gradients_Z2 = torch.autograd.grad(dlH0Pred_IMP2.sum(), dlZ0BC4, create_graph=True)[0]
    lossZ2 = mse_loss(gradients_Z2, torch.zeros_like(gradients_Z2))
    
    # Compute the loss for Neumann conditions (for x-direction)
    dlH0Pred_IMP3 = model(parameters, dlX0BC5, dlZ0BC5, dlT0BC5)
    gradients_X1 = torch.autograd.grad(dlH0Pred_IMP3.sum(), dlX0BC5, create_graph=True)[0]
    lossX1 = mse_loss(gradients_X1, torch.zeros_like(gradients_X1))
    
    dlH0Pred_IMP4 = model(parameters, dlX0BC6, dlZ0BC6, dlT0BC6)
    gradients_X2 = torch.autograd.grad(dlH0Pred_IMP4.sum(), dlX0BC6, create_graph=True)[0]
    lossX2 = mse_loss(gradients_X2, torch.zeros_like(gradients_X2))
    
    # Use the second neural network to predict the free surface coordinate
    dlS = model_2(parameters2, S0IC[0:1, :], S0IC[1:2, :])
    lossS = mse_loss(dlS, S0IC[2:3, :])
    
    # Enforce consistency of the free surface prediction with the head predicted by the first network
    dlS_surface = model_2(parameters2, dlX, dlT)
    H_surface = model(parameters, dlX, dlS_surface, dlT)
    loss_surface = mse_loss(H_surface - dlS_surface, torch.zeros_like(H_surface))
    
    # Combine all loss components
    loss = lossF + lossU + lossZ1 + lossZ2 + lossX1 + lossX2 + lossS + loss_surface
    loss.backward(retain_graph=True)
    
    # Collect gradients from both networks
    gradients = { 'parameters': {}, 'parameters2': {} }
    for key, layer in parameters.items():
        gradients['parameters'][key] = {
            'Weights': layer['Weights'].grad.clone() if layer['Weights'].grad is not None else None,
            'Bias': layer['Bias'].grad.clone() if layer['Bias'].grad is not None else None
        }
    for key, layer in parameters2.items():
        gradients['parameters2'][key] = {
            'Weights': layer['Weights'].grad.clone() if layer['Weights'].grad is not None else None,
            'Bias': layer['Bias'].grad.clone() if layer['Bias'].grad is not None else None
        }
    return gradients, loss, {**parameters, **parameters2}
