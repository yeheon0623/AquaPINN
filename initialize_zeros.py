# initialize_zeros.py
import torch

def initialize_zeros(sz, dtype=torch.float32):
    """
    Initialize the bias tensor with zeros.
    
    Parameters:
      sz (tuple): The size of the bias tensor, e.g., (output_dimension,).
      dtype (torch.dtype): The data type (default is torch.float32).
      
    Returns:
      torch.Tensor: A tensor initialized with zeros and with gradients enabled.
    """
    parameter = torch.zeros(sz, dtype=dtype)
    parameter.requires_grad_()
    return parameter
