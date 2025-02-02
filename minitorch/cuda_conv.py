# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs) -> Fn:
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn, **kwargs) -> FakeCUDAKernel:
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32

def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """

  
    The grid x, y, z axis corresponds to the [B, T, COUT] dim of the output.


    """

    # Get shapes
    #
    # input: `batch, in_channels, width`
    # weight: `out_channels, in_channels, k_width`
    # output: `batch, out_channels, width`
    #
    batch, in_channels, width = input_shape
    out_channels, _, k_width = weight_shape
    assert batch == out_shape[0]
    assert out_channels == out_shape[1]
    assert width == out_shape[2]

    # TODO: why
    input_batch_stride = input_strides[0] if input_shape[0] > 1 else 0

    # get the index in the grid
    # [B, T, COUT]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z

    if i >= batch or j >= width:
        return

    # Allocate shared memory.
    BLOCK_DIM = 32
    input_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM, BLOCK_DIM), numba.float64) # [B, T, KW * C_IN]
    weight_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64) # [KW * C_IN, C_OUT]

    for conv_i in range(0, k_width * in_channels, BLOCK_DIM):
        k_width_i = conv_i // in_channels
        in_channels_i = conv_i % in_channels

        input_index = (batch, in_channels_i, width + k_width_i)
        input_shared[i][j][conv_i] = input[input_batch_stride * batch + input_strides[1] * in_channels_i + input_strides[2] * (width + k_width_i)]
        weight_shared[conv_i][k] = weight[weight_strides[0] * k + weight_strides[1] * in_channels_i + weight_strides[2] * k_width_i]

        if i < batch and j < width and k < k_width * in_channels:
            input_shared[i][j]

    


