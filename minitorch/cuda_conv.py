# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any, Tuple

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .autodiff import Context
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
from .tensor_functions import Function

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

# Cuda has limit of max 1024 threads per block. Here we set this to 8 because we are building 3D block.
THREADS_PER_BLOCK = 8

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

    # out[0] = 1.23
    # out[1] = 2.34
    # out[2] = 2.34
    # out[3] = 2.34
    # out[6] = 2.34
    
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

    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y
    pk = cuda.threadIdx.z

    # out[0] = 1.23
    # out[1] = 2.34
    # out[2] = 2.34
    # out[3] = 2.34
    # out[6] = 2.34

    if i >= batch or j >= width or k >= out_channels:
        return
    # if i >= batch or j >= out_channels or k >= width:
    #     return

    # out[0] = 1.23
    # out[1] = 2.34
    # out[2] = 2.34
    # out[3] = 2.34
    # out[6] = 2.34

    # Allocate shared memory.
    BLOCK_DIM = 8
    input_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM, BLOCK_DIM), numba.float64) # [B, T, KW * C_IN]
    weight_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64) # [KW * C_IN, C_OUT]

    total = 0.0
    # total = 1.25
    for conv_i in range(0, k_width * in_channels, BLOCK_DIM):
        # assert conv_i == 1
        k_width_i = (pk + conv_i) // in_channels
        in_channels_i = (pk + conv_i) % in_channels

        if pk + conv_i < k_width * in_channels:
            input_shared[i][j][pk] = input[input_batch_stride * i + input_strides[1] * in_channels_i + input_strides[2] * (j + k_width_i)]
            weight_pos = weight_strides[0] * k + weight_strides[1] * in_channels_i + weight_strides[2] * k_width_i
            # print(f"===lizhi {k=} {in_channels_i=} {weight_strides.shape=} {weight_strides[0]=} {weight_strides[1]=} {weight_strides[2]=} {weight_pos=}")
            weight_shared[pk][k] = weight[weight_pos]
            input_shared[i][j][pk] = input[input_batch_stride * i + input_strides[1] * 0 + input_strides[2] * (j + k_width_i)]
            # input_shared[i][j][pk] = 1.2


        # numba.cuda.syncthreads()

        for iii in range(BLOCK_DIM):
            if iii + conv_i >= k_width * in_channels:
                break
            total += input_shared[i][j][iii] * weight_shared[iii][k]

    out_pos = out_strides[0] * i + out_strides[1] * k + out_strides[2] * j
    print(f"===lizhi {out_shape[1]=} {out_shape[2]=} {i=} {j=} {k=} {out_strides[0]=} {out_strides[1]=} {out_strides[2]=} {out_pos=}")
    out[out_pos] = total
    
    # out[0] = 1.23
    # out[1] = 2.34
    # out[2] = 2.34
    # out[3] = 2.34
    # out[6] = 2.34

tensor_conv1d = jit(_tensor_conv1d)

class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 1D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
        -------
            batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))



        # One block per batch, extra rows, extra col
        blockspergrid = (
            (output.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (output.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (output.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK)
        # threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
        # threadsperblock = (4, 4, 4)
        # threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK * THREADS_PER_BLOCK)
        # threadsperblock = (32, 32, 1)
        tensor_conv1d[blockspergrid, threadsperblock](
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )

        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)

        blockspergrid = (
            (grad_weight.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (grad_weight.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (grad_weight.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK)
        # threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
        # tensor_conv1d[blockspergrid, threadsperblock](
        #     *grad_weight.tuple(), grad_weight.size, *new_input.tuple(), *new_grad_output.tuple(), False
        # )



        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d[blockspergrid, threadsperblock](
            *grad_input.tuple(), grad_input.size, *grad_output.tuple(), *new_weight.tuple(), True
        )
        # tensor_conv1d(  # type: ignore
        #     *grad_input.tuple(),
        #     grad_input.size,  # type: ignore
        #     *grad_output.tuple(),
        #     *new_weight.tuple(),
        #     True,  # type: ignore
        # )
        return grad_input, grad_weight

        

    


