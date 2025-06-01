import torch
import torch.nn as nn
import numpy as np
from utils.common_utils import *

def fill_noise(x, noise_type):
    """
    Fills the input tensor `x` with random noise based on the specified `noise_type`.

    Args:
        x (torch.Tensor): The tensor to be filled with noise.
        noise_type (str): The type of noise to fill the tensor with.
            - 'u' : Uniform noise (random values between 0 and 1).
            - 'n' : Normal noise (random values from a normal distribution).

    Raises:
        ValueError: If an invalid `noise_type` is provided.
    """

    if noise_type == 'u' : 
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else :
        raise ValueError(f"Invalid noise_type '{noise_type}'. Use 'u' for uniform or 'n' for normal.")

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """
    Generates a PyTorch tensor with dimensions (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`), 
    initialized based on the specified method and noise type.
    
    Args:
        input_depth (int): Number of channels in the tensor.
        method (str): Method for tensor initialization. 
                      'noise' fills the tensor with random noise, 'meshgrid' initializes using a meshgrid.
        spatial_size (int or tuple): Height and width of the tensor. If a single integer is provided, 
                                     the tensor will be square.
        noise_type (str, optional): Type of noise to generate, 'u' for uniform or 'n' for normal. 
                                    Defaults to 'u'.
        var (float, optional): Variance scaling factor for the noise. Affects the standard deviation. 
                               Defaults to 1./10.
    
    Returns:
        torch.Tensor: A 4D tensor initialized as specified.
    
    Raises:
        ValueError: If an unsupported method is provided or if input_depth is invalid for 'meshgrid'.
    """
    
    # Ensure spatial_size is a tuple
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    
    # Handle 'noise' method: fill tensor with noise
    if method == 'noise':
        tensor_shape = (1, input_depth, spatial_size[0], spatial_size[1])
        net_input = torch.zeros(tensor_shape)
        
        # Apply noise using fill_noise function based on the noise_type
        fill_noise(net_input, noise_type)
        
        # Scale the noise by variance
        net_input *= var
    
    # Handle 'meshgrid' method: generate 2D grid coordinates
    elif method == 'meshgrid':
        if input_depth != 2:
            raise ValueError(f"'meshgrid' method requires input_depth to be 2, but got {input_depth}")
        
        # Create meshgrid for 2D coordinates normalized to the range [0, 1]
        X, Y = np.meshgrid(
            np.linspace(0, 1, spatial_size[1]), 
            np.linspace(0, 1, spatial_size[0])
        )
        
        # Concatenate and convert to PyTorch tensor
        meshgrid = np.stack([X, Y], axis=0)
        net_input = np_to_torch(meshgrid)
    
    # Raise error for unsupported method
    else:
        raise ValueError(f"Unsupported method '{method}'. Choose 'noise' or 'meshgrid'.")
    
    return net_input


def get_params(opt_targets, net, net_input, downsampler=None):
    """
    Gathers the parameters to be optimized based on the specified targets.

    Args:
        opt_targets (str): Comma-separated string indicating what to optimize, e.g., "net,input" or "net".
        net (torch.nn.Module): The neural network.
        net_input (torch.Tensor): The input tensor `z` that can be set to require gradients.
        downsampler (torch.nn.Module, optional): Optional downsampler module for optimization.

    Returns:
        list: A list of parameters to be optimized.
    """
    # Split the targets into a list
    target_list = opt_targets.split(',')
    parameters_to_optimize = []
    
    # Loop through the specified targets
    for target in target_list:
        if target == 'net':
            # Add network parameters if specified
            parameters_to_optimize += list(net.parameters())
        elif target == 'downsampler':
            # Ensure downsampler exists, then add its parameters
            if downsampler is None:
                raise ValueError("Downsampler is not provided!")
            parameters_to_optimize += list(downsampler.parameters())
        elif target == 'input':
            # Set net_input to require gradients and add it to the parameters list
            net_input.requires_grad = True
            parameters_to_optimize.append(net_input)
        else:
            # Raise an error if an unknown target is encountered
            raise ValueError(f"Unrecognized optimization target: {target}")
    
    return parameters_to_optimize

















