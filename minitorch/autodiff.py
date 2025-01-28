from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

import copy

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals_lower = copy.deepcopy(list(vals))
    vals_lower[arg] -= epsilon

    vals_higher = copy.deepcopy(list(vals))
    vals_higher[arg] += epsilon

    ret = (f(*vals_higher) - f(*vals_lower)) / 2 / epsilon
    return ret


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass



def dfs(variable: Variable, order: list[Variable], visited: set[int]) -> None:
    visited.add(variable.unique_id)

    if not variable.is_constant():
    # if True:
      for p in variable.parents:
          if not p.unique_id in visited:
              dfs(p, order, visited)

    order.append(variable)

def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    order = []
    visited = set()
    dfs(variable, order, visited)

    return reversed(order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    order = list(topological_sort(variable))
    # print(f"===lizhi autodiff backprop {[n.unique_id for n in order]}")
    # for n in order:
    #     print(f"===lizhi autodiff backprop {n.unique_id} {n.is_leaf()=} {n.is_constant()=} {n._tensor._storage=}")
    

    acc_derivs = {variable.unique_id: deriv}

    for v in order:
      v_deriv = acc_derivs[v.unique_id]
      if v.is_leaf():
        v.accumulate_derivative(v_deriv)
        continue

      if v.is_constant():
        continue

      parents_and_derivs = v.chain_rule(v_deriv)
      for p, p_deriv in parents_and_derivs:
        if p.unique_id in acc_derivs:
          acc_derivs[p.unique_id] += p_deriv
        else:
          acc_derivs[p.unique_id] = p_deriv


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False

    # Note, the saved_vallues here are the **raw value** of the local inputs.
    # The Scalar.history.inputs are the **Scalars** of the inputs.
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values