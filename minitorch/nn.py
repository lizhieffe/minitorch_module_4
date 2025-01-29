from typing import Tuple
from minitorch.tensor_data import index_to_position, to_index

import numpy as np
from numpy._core.defchararray import title

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling with an average operation.

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    input_storage, _, input_strides = input.tuple()

    out_h = height // kh
    out_w = width // kw


    out_tensor = input.zeros((batch, channel, out_h, out_w))
    out, out_shape, out_strides = out_tensor.tuple()
    out_size = batch * channel * out_h * out_w


    for i in range(out_size):
        out_idx = np.empty_like(out_shape)
        to_index(i, out_shape, out_idx)
        bi, ci, ohi, owi = out_idx

        total = 0.0
        for khi in range(kh):
            for kwi in range(kw):
                input_idx = np.array((bi, ci, ohi * kh + khi, owi * kw + kwi))
                input_pos = index_to_position(input_idx, input_strides)
                input_val = input_storage[input_pos]

                total += input_val
        total = total / kh / kw

        out_pos_raw = out_idx[0] * out_strides[0] + out_idx[1] * out_strides[1] + out_idx[2] * out_strides[2] + out_idx[3] * out_strides[3]
        out_pos = index_to_position(out_idx, out_strides)
        out[out_pos] = total
    return out_tensor, out_h, out_w

class AvgPool2d(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, kernel: Tensor) -> Tensor:
        kernel = np.array(kernel.to_numpy(), dtype=np.int64)
        ctx.save_for_backward(t1, kernel)
        return tile(t1, kernel)[0]

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (t1, kernel) = ctx.saved_tensors
        
        # Out here means the emit to previous layer.
        out = t1.zeros(t1.shape)
        out_storage, out_shape, out_strides = out.tuple()
        
        print(f"===lizhi {grad_output=}")
        for i in range(out_storage.shape[0]):
            out_idx = np.empty_like(out_shape)
            to_index(i, out_shape, out_idx)

            grad_storage, grad_shape, grad_strides = grad_output.tuple()
            grad_idx = np.array(out_idx)
            grad_idx[-2] = out_idx[-2] // kernel[0]
            grad_idx[-1] = out_idx[-1] // kernel[1]
            grad_pos = index_to_position(grad_idx, grad_strides)
            grad_val = grad_storage[grad_pos]

            out_pos = index_to_position(out_idx, out_strides)
            out_storage[out_pos] = grad_val / kernel[0] / kernel[1]

        return out


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    kernel_tensor = input.make(kernel, (2,), backend=input.backend)
    return AvgPool2d.apply(input, kernel_tensor)

# TODO: Implement for Task 4.3.
