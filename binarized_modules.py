import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd.function import Function, InplaceFunction

import numpy as np


class Binarize1(InplaceFunction):
    """
    Binarize1 class for binarizing input tensors.
    """

    def forward(ctx, input, quant_mode='det', allow_scale=False, inplace=False):
        """
        Forward pass of the binarization function.

        Args:
            input (Tensor): The input tensor to be binarized.
            quant_mode (str): The quantization mode. Default is 'det'.
            allow_scale (bool): Whether to allow scaling. Default is False.
            inplace (bool): Whether to perform the operation in-place. Default is False.

        Returns:
            Tensor: The binarized output tensor.
        """
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        scale = output.abs().max() if allow_scale else 1

        if quant_mode == 'det':
            return output.div(scale).sign().mul(scale)
        else:
            return (output.div(scale).add_(1).div_(2).add_(torch.rand(output.size()).add(-0.5)).clamp_(0, 1).round().
                    mul_(2).add_(-1).mul(scale))

    def backward(ctx, grad_output):
        """
        Backward pass of the binarization function.

        Args:
            grad_output (Tensor): The gradient of the output tensor.

        Returns:
            Tensor: The gradient of the input tensor.
        """
        # STE
        grad_input = grad_output
        return grad_input, None, None, None


# Class for binarizing input tensors
class Binarize2(InplaceFunction):
    """
    Binarize2 class for binarizing input tensors.
    """
    def forward(ctx, input, quant_mode='det', allow_scale=False, inplace=False):
        """
        Forward pass of the binarization function.

        Args:
            input (Tensor): The input tensor to be binarized.
            quant_mode (str): The quantization mode. Default is 'det'.
            allow_scale (bool): Whether to allow scaling. Default is False.
            inplace (bool): Whether to perform the operation in-place. Default is False.

        Returns:
            Tensor: The binarized output tensor.
        """
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        scale = output.abs().max() if allow_scale else 1

        if quant_mode == 'det':
            return output.div(scale).sign().mul(scale)
        else:
            return output.div(scale).sign().mul(scale)

    def backward(ctx, grad_output):
        """
        Backward pass of the binarization function.

        Args:
            grad_output (Tensor): The gradient of the output tensor.

        Returns:
            Tensor: The gradient of the input tensor.
        """
        # Straight Through Estimator (STE)
        grad_input = grad_output
        return grad_input, None, None, None
    
# Function for binarizing input tensors using Binarize1 class
def binarized1(input, quant_mode='det'):
    """
    Binarizes the input tensor using Binarize1 class.

    Args:
        input (Tensor): The input tensor to be binarized.
        quant_mode (str): The quantization mode. Default is 'det'.

    Returns:
        Tensor: The binarized output tensor.
    """
    return Binarize1.apply(input, quant_mode)

# Function for binarizing input tensors using Binarize2 class
def binarized2(input, quant_mode='det'):
    """
    Binarizes the input tensor using Binarize2 class.

    Args:
        input (Tensor): The input tensor to be binarized.
        quant_mode (str): The quantization mode. Default is 'det'.

    Returns:
        Tensor: The binarized output tensor.
    """
    return Binarize2.apply(input, quant_mode)


# def quantize(input, quant_mode, numBits):
#     return Quantize.apply(input, quant_mode, numBits)


class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        if input.size(1) != 784:
            input_b = binarized2(input)
            print("#####################################################################################")
            print("The binarized outcome of this layer: ", input_b)
            print("#####################################################################################")
        else:

            input_b = input
            print("#####################################################################################")
            print("The binarized outcome of the input layer: ", input_b)
            print("#####################################################################################")
        weight_b = binarized2(self.weight)
        out = nn.functional.linear(input_b, weight_b)
        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out 