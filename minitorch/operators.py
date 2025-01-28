"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Any, Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:

EPS = 1e-6


# - mul
def mul(a: float, b: float) -> float:
    # assert False
    return a * b


# - id
def id(a) -> float:
    return a


# - add
def add(a, b):
    return a + b


# - neg
def neg(a: float) -> float:
    return -1.0 * a


# - lt
def lt(a: float, b: float) -> float:
    return 1.0 if a < b else 0.0


# - eq
def eq(a, b):
    return 1.0 if a == b else 0.0


# - max
def max(a, b):
    if a > b:
        return a
    else:
        return b


# - is_close
def is_close(a, b) -> bool:
    return abs(a - b) < 1e-2


# - sigmoid
def sigmoid(a):
    return 1 / (1 + math.exp(-1 * a))

def sigmoid_back(a):
    # For some reason, calling sigmoid() directly failed in Numba.
    def _sigmoid(a):
      return 1 / (1 + math.exp(-1 * a))
    return _sigmoid(a) * (1 - _sigmoid(a))

# - relu
def relu(a):
    if a > 0:
        return a
    else:
        return 0.0


# - log
def log(a):
    ret = math.log(a)
    return ret


# - exp
def exp(a):
    return math.exp(a)


# - log_back
def log_back(a, b):
    return b / (a + EPS)


# - inv
def inv(a: float) -> float:
    ret = 1.0 / a
    return ret


# - inv_back
def inv_back(a, b_output):
    """The backprop eq for inv().

    Args:
      a: the value in the forward inv().
      b_output: the derivative that to backprop from the next layer.

    Returns:
      The derivative that to backprop to the previous layer. 
    """
    return -(1.0 / a**2) * b_output


# - relu_back
# TODO: is this right?
def relu_back(x, y):
    """:math:`f(x) =` y if x is greater than 0 else 0"""
    # TODO: Implement for Task 0.1.
    if x > 0:
        return y
    else:
        return 0


#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions


# - map
def map(fn: Callable[[Any], Any], a: list[Any]) -> list[Any]:
    return [fn(x) for x in a]


# - zipWith
def zipWith(fn: Callable[[Any, Any], Any], a: list[Any], b: list[Any]) -> list[Any]:
    return [fn(x, y) for x, y in zip(a, b)]


# - reduce
def reduce(fn: Callable[[Any, Any], Any], a: Iterable) -> Any:
    reduced = None
    for x in a:
        if reduced is None:
            reduced = x
        else:
            reduced = fn(reduced, x)

    return 0.0 if reduced is None else reduced


#
# Use these to implement


# - negList : negate a list
def negList(a: list[Any]) -> list[Any]:
    return map(neg, a)


# - addLists : add two lists together
def addLists(a: list[float], b: list[float]) -> list[float]:
    return zipWith(add, a, b)


# - sum: sum lists
def sum(a: list[Any]) -> Any:
    return reduce(add, iter(a))


# - prod: take the product of lists
def prod(a: list[Any]) -> Any:
    return reduce(mul, iter(a))


# TODO: Implement for Task 0.3.