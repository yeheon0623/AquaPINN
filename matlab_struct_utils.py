# matlab_struct_utils.py
import numpy as np
import torch
import scipy.io

def matlab_struct_to_dict(mat_struct):
    # Convert MATLAB struct/cell to a Python dict/list/ndarray
    ...

def convert_weights_bias_to_tensors(params_dict):
    # Traverse each layer (e.g., fcX) in the dictionary and convert 'Weights' and 'Bias' from numpy.ndarray to torch.Tensor
    ...

def load_parameters_from_mat(mat_file_path):
    """
    Load the 'parameters' variable from a .mat file and convert it into a Python dict with PyTorch tensors.
    Returns py_parameters that can be directly used in model(...).
    """
    mat_data = scipy.io.loadmat(mat_file_path)
    mat_parameters = mat_data['parameters']  # Assume the variable stored in the .mat file is named 'parameters'
    
    py_parameters = matlab_struct_to_dict(mat_parameters)
    py_parameters = convert_weights_bias_to_tensors(py_parameters)
    return py_parameters
