from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        # The input matrix are either 2 or 3 dims. If 3, it is for the batch dim. If it is 2, it is extended to 3.
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # TODO(lizhi): This commented out code doesn't work for some reason. Figure it out!!
        #
        # # When `out` and `in` are stride-aligned, avoid indexing
        # if np.array_equal(in_strides, out_strides) and np.array_equal(in_shape, out_shape):
        #     for out_pos in range(out.shape[0]):
        #         out[out_pos] = fn(in_storage[out_pos])
        #     return


        # Not stride-aligned.
        for out_pos in prange(out.shape[0]):
            out_idx = np.zeros(out_shape.shape, dtype=np.int64)
            in_idx = np.zeros(in_shape.shape, dtype=np.int64)

            # out_pos -> out_idx
            to_index(out_pos, out_shape, out_idx)
            # out_idx -> in_idx
            broadcast_index(out_idx, out_shape, in_shape, in_idx)
            # in_idx -> in_pos
            in_pos = index_to_position(in_idx, in_strides)

            real_out_pos = index_to_position(out_idx, out_strides)
            out[real_out_pos] = fn(in_storage[in_pos])

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # When `out`, `a_storage`, `b_storage` are stride-aligned, avoid indexing
        if np.array_equal(a_strides, b_strides) and np.array_equal(out_strides, a_strides) and np.array_equal(a_shape, b_shape) and np.array_equal(a_shape, out_shape):
            for i in prange(out.shape[0]):
                out[i] = fn(a_storage[i], b_storage[i])
            return

        # When not stride-aligned.
        for out_pos in prange(out.shape[0]):
        # for out_pos in range(out.shape[0]):
            # Note: dtype cannot use python int type which is incompatible with numba.
            out_idx = np.zeros(out_shape.shape, dtype=np.int64)
            # out_pos -> out_idx
            to_index(out_pos, out_shape, out_idx)

            a_idx = np.zeros(a_shape.shape, dtype=np.int64)
            # out_idx -> a_idx
            broadcast_index(out_idx, out_shape, a_shape, a_idx)
            # a_idx -> a_pos
            a_pos = index_to_position(a_idx, a_strides)
            a_val = a_storage[a_pos]

            b_idx = np.zeros(b_shape.shape, dtype=np.int64)
            # out_idx -> b_idx
            broadcast_index(out_idx, out_shape, b_shape, b_idx)
            # b_idx -> b_pos
            b_pos = index_to_position(b_idx, b_strides)
            b_val = b_storage[b_pos]

            out[out_pos] = fn(a_val, b_val)

    # return _zip

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        for i in prange(out.shape[0]):
            # Generate out_index
            out_index = np.zeros(out_shape.shape)
            to_index(i, out_shape, out_index)

            reduced = 0.0
            for j in range(a_shape[reduce_dim]):
                a_index = out_index
                a_index[reduce_dim] = j
                a_pos = index_to_position(a_index, a_strides)
                a_val = a_storage[a_pos]

                if j == 0:
                    reduced = a_val
                else:
                    reduced = fn(reduced, a_val)

            out[i] = reduced

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    The dim of a, b, out are all 3. See the matrix_multiply() fn in this file for details.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    assert a_shape[-1] == b_shape[-2]
    assert a_shape.shape[0] == 3
    assert len(a_shape.shape) == len(b_shape.shape)
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    for i in prange(out.shape[0]):
        out_idx = np.zeros(out_shape.shape)
        to_index(i, out_shape, out_idx)

        inner_sum = 0
        for j in range(a_shape[-1]):
            # Get a_value
            a_big_idx = out_idx.copy()
            a_big_idx[2] = j
            # There might be broadcast on the dim 0 (e.g. 1 element extended to N elements). So cast the a_idx back to a's original shape.
            a_big_shape = a_shape.copy()
            a_big_shape[0] = out_shape[0]
            a_idx = np.zeros(a_shape.shape)
            broadcast_index(a_big_idx, a_big_shape, a_shape, a_idx)
            a_pos = index_to_position(a_idx, a_strides)
            a_val = a_storage[a_pos]

            # Get b_value
            b_big_idx = out_idx.copy()
            b_big_idx[1] = j
            # There might be broadcast on the dim 0 (e.g. 1 element extended to N elements). So cast the b_idx back to b's original shape.
            b_big_shape = b_shape.copy()
            b_big_shape[0] = out_shape[0]
            b_idx = np.zeros(b_shape.shape)
            broadcast_index(b_big_idx, b_big_shape, b_shape, b_idx)
            b_pos = index_to_position(b_idx, b_strides)
            b_val = b_storage[b_pos]

            
            inner_sum += (a_val * b_val)

        out[i] = inner_sum


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
