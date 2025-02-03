import pytest
from hypothesis import given, settings

import minitorch
from minitorch import Tensor

from .tensor_strategies import assert_close_tensor, shaped_tensors, tensors
from .strategies import assert_close, small_floats


@pytest.mark.task4_1
def test_conv1d_simple() -> None:
    # input: `batch, in_channels, width`
    # weight: `out_channels, in_channels, k_width`
    # output: `batch, out_channels, width`
    
    t = minitorch.tensor([0, 1, 2, 3]).view(1, 1, 4)
    t.requires_grad_(True)
    t2 = minitorch.tensor([[1, 2, 3]]).view(1, 1, 3)
    print(f"===lizhi {t=} {t2=}")
    _, t2_shape, t2_strides = t2.tuple()
    print(f"===lizhi {t2_shape=} {t2_strides=}")
    out = minitorch.Conv1dCudaFun.apply(t, t2)

    print(f"===lizhi {out[0,0,0]=}")
    assert out[0, 0, 0] == 0 * 1 + 1 * 2 + 2 * 3
    assert out[0, 0, 1] == 1 * 1 + 2 * 2 + 3 * 3
    assert out[0, 0, 2] == 2 * 1 + 3 * 2
    assert out[0, 0, 3] == 3 * 1

@pytest.mark.task4_1
@given(tensors(shape=(2, 2, 6)), tensors(shape=(3, 2, 2)))
@settings(max_examples=50)
def test_conv1d_simple_2(input: Tensor, weight: Tensor) -> None:
    out = minitorch.Conv1dCudaFun.apply(input, weight)
    assert_close(out[0,0,0], input[0, 0, 0] * weight[0, 0, 0] + input[0, 0, 1] * weight[0, 0, 1] + input[0, 1, 0] * weight[0, 1, 0] + input[0, 1, 1] * weight[0, 1, 1])

@pytest.mark.task4_1
@given(tensors(shape=(1, 1, 6)), tensors(shape=(1, 1, 4)))
def test_conv1d(input: Tensor, weight: Tensor) -> None:
    print(input, weight)
    minitorch.grad_check(minitorch.Conv1dFun.apply, input, weight)


@pytest.mark.task4_1
@given(tensors(shape=(2, 2, 6)), tensors(shape=(3, 2, 2)))
@settings(max_examples=50)
def test_conv1d_channel(input: Tensor, weight: Tensor) -> None:
    minitorch.grad_check(minitorch.Conv1dFun.apply, input, weight)


@pytest.mark.task4_2
@given(tensors(shape=(1, 1, 6, 6)), tensors(shape=(1, 1, 2, 4)))
def test_conv(input: Tensor, weight: Tensor) -> None:
    minitorch.grad_check(minitorch.Conv2dFun.apply, input, weight)


@pytest.mark.task4_2
@given(tensors(shape=(2, 1, 6, 6)), tensors(shape=(1, 1, 2, 4)))
@settings(max_examples=10)
def test_conv_batch(input: Tensor, weight: Tensor) -> None:
    minitorch.grad_check(minitorch.Conv2dFun.apply, input, weight)


@pytest.mark.task4_2
@given(tensors(shape=(2, 2, 6, 6)), tensors(shape=(3, 2, 2, 4)))
@settings(max_examples=10)
def test_conv_channel(input: Tensor, weight: Tensor) -> None:
    minitorch.grad_check(minitorch.Conv2dFun.apply, input, weight)


@pytest.mark.task4_2
def test_conv2() -> None:
    t = minitorch.tensor([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]).view(
        1, 1, 4, 4
    )
    t.requires_grad_(True)

    t2 = minitorch.tensor([[1, 1], [1, 1]]).view(1, 1, 2, 2)
    t2.requires_grad_(True)
    out = minitorch.Conv2dFun.apply(t, t2)
    out.sum().backward()

    minitorch.grad_check(minitorch.Conv2dFun.apply, t, t2)
