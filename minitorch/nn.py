from typing import Tuple
from minitorch.tensor_data import index_to_position

import numpy as np

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

    out_h = height / kh
    out_w = width / kw

    out_tensor = input.zeros((batch, channel, out_h, out_w))
    out, out_shape, out_strides = out_tensor.tuple()
    out_size = batch * channel * out_h * out_w

    for i in range(out_size):
        out_idx = np.empty_like(out_shape)
        bi, ci, ohi, owi = out_idx

        total = 0.0
        for khi in range(kh):
            for kwi in range(kw):
                input_idx = np.array((bi, ci, ohi * kh + khi, owi * kw + kwi))
                input_pos = index_to_position(input_idx, input_strides)
                input_val = input_storage[input_pos]

                total += input_val
        total = total / kh / kw

        out_pos = index_to_position(out_idx, out_shape)
        out[out_pos] = total

# TODO: Implement for Task 4.3.
