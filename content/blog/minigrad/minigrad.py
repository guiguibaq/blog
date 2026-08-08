"""Minigrad: a simplified version of PyTorch autograd.

It follows the same philosophy as Karpathy's micrograd, but
gets closer to real PyTorch in two ways: it follows PyTorch's API (one class
for the data, one for the logic), and it supports numpy inputs.

  Data      <-> torch.Tensor
  Function  <-> torch.autograd.Function
  Layer     <-> torch.nn.Module
  SGD       <-> torch.optim.SGD
"""

from __future__ import annotations

from typing import Any, List, Set, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Broadcasting
# ---------------------------------------------------------------------------

def unbroadcast(grad: np.ndarray, original_shape: Tuple[int, ...]) -> np.ndarray:
  """Undo the broadcast numpy silently applied during the forward.

  A gradient flowing back to a node must have the shape of that node, so we
  sum over every dimension that broadcasting added or stretched.
  """
  while grad.ndim > len(original_shape):
    grad = grad.sum(axis=0)
  for i, dim in enumerate(original_shape):
    if dim == 1 and grad.shape[i] != 1:
      grad = grad.sum(axis=i, keepdims=True)
  return grad


# ---------------------------------------------------------------------------
# Function: the base class holding the forward / backward logic
# ---------------------------------------------------------------------------

class Function:
  def __init__(self):
    self.parents: List['Data'] = []
    self._saved: Tuple[Any, ...] = ()

  def save_for_backward(self, *tensors: Any):
    self._saved = tensors

  @staticmethod
  def forward(ctx: Any, *args: np.ndarray, **kwargs: Any) -> np.ndarray:
    raise NotImplementedError()  # should be implemented by the child class

  @staticmethod
  def backward(ctx: Any, grad_in: np.ndarray) -> Tuple[np.ndarray, ...]:
    raise NotImplementedError()  # should be implemented by the child class

  @classmethod
  def apply(cls, *args: 'Data', **kwargs: Any) -> 'Data':
    # args are data, kwargs are extra parameters, like the exponent in a power

    # create instance of operator
    operation_instance = cls()
    operation_instance.parents = list(args)

    # create Data to be returned
    data_out = operation_instance.forward(
        operation_instance, *[arg.data for arg in args], **kwargs
    )
    requires_grad = any([arg.requires_grad for arg in args])
    return Data(data_out, requires_grad=requires_grad, grad_fn=operation_instance)

  def __repr__(self) -> str:
    return f"{type(self).__name__}_{id(self)}"


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

class MatMul(Function):
  @staticmethod
  def forward(ctx: Any, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ctx.save_for_backward(a, b)
    return a @ b

  @staticmethod
  def backward(ctx: Any, grad_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a, b = ctx._saved
    return grad_in @ b.T, a.T @ grad_in


class Add(Function):
  @staticmethod
  def forward(ctx: Any, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ctx.save_for_backward(a, b)
    return a + b

  @staticmethod
  def backward(ctx: Any, grad_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return grad_in, grad_in


class Sub(Function):
  @staticmethod
  def forward(ctx: Any, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ctx.save_for_backward(a, b)
    return a - b

  @staticmethod
  def backward(ctx: Any, grad_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return grad_in, -grad_in


class Mul(Function):
  @staticmethod
  def forward(ctx: Any, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ctx.save_for_backward(a, b)
    return a * b

  @staticmethod
  def backward(ctx: Any, grad_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a, b = ctx._saved
    return grad_in * b, grad_in * a


class Div(Function):
  @staticmethod
  def forward(ctx: Any, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ctx.save_for_backward(a, b)
    return a / b

  @staticmethod
  def backward(ctx: Any, grad_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a, b = ctx._saved
    return grad_in / b, - grad_in * a / b**2


class Neg(Function):
  @staticmethod
  def forward(ctx: Any, a: np.ndarray) -> np.ndarray:
    ctx.save_for_backward(a)
    return -a

  @staticmethod
  def backward(ctx: Any, grad_in: np.ndarray) -> Tuple[np.ndarray]:
    return (-grad_in,)


class Pow(Function):
  @staticmethod
  def forward(ctx: Any, a: np.ndarray, exponent: int) -> np.ndarray:
    ctx.save_for_backward(a)
    ctx.exponent = exponent
    return a**exponent

  @staticmethod
  def backward(ctx: Any, grad_in: np.ndarray) -> Tuple[np.ndarray]:
    a = ctx._saved[0]
    return (grad_in * ctx.exponent * a ** (ctx.exponent - 1),)


class Exp(Function):
  @staticmethod
  def forward(ctx: Any, a: np.ndarray) -> np.ndarray:
    ctx.save_for_backward(a)
    return np.exp(a)

  @staticmethod
  def backward(ctx: Any, grad_in: np.ndarray) -> Tuple[np.ndarray]:
    a = ctx._saved[0]
    return (grad_in * np.exp(a),)


class Log(Function):
  @staticmethod
  def forward(ctx: Any, a: np.ndarray) -> np.ndarray:
    ctx.save_for_backward(a)
    return np.log(a)

  @staticmethod
  def backward(ctx: Any, grad_in: np.ndarray) -> Tuple[np.ndarray]:
    a = ctx._saved[0]
    return (grad_in / a,)


class ReLu(Function):
  @staticmethod
  def forward(ctx: Any, a: np.ndarray) -> np.ndarray:
    ctx.save_for_backward(a)
    return np.maximum(a, 0)

  @staticmethod
  def backward(ctx: Any, grad_in: np.ndarray) -> Tuple[np.ndarray]:
    a = ctx._saved[0]
    return (grad_in * (a > 0),)


class Sum(Function):
  @staticmethod
  def forward(ctx: Any, a: np.ndarray,
              axis: Union[int, Tuple[int, ...], None] = None,
              keepdims: bool = False) -> np.ndarray:
    ctx.in_shape = a.shape
    ctx.axis = axis
    ctx.keepdims = keepdims
    return a.sum(axis=axis, keepdims=keepdims)

  @staticmethod
  def backward(ctx: Any, grad_in: np.ndarray) -> Tuple[np.ndarray]:
    grad_in = np.asarray(grad_in, dtype=np.float64)
    if ctx.axis is not None and not ctx.keepdims:
      grad_in = np.expand_dims(grad_in, ctx.axis)
    return (np.broadcast_to(grad_in, ctx.in_shape).copy(),)


class Mean(Function):
  @staticmethod
  def forward(ctx: Any, a: np.ndarray,
              axis: Union[int, Tuple[int, ...], None] = None,
              keepdims: bool = False) -> np.ndarray:
    ctx.in_shape = a.shape
    ctx.axis = axis
    ctx.keepdims = keepdims
    if axis is None:
      ctx.n = a.size
    else:
      axes = (axis,) if isinstance(axis, int) else tuple(axis)
      ctx.n = int(np.prod([a.shape[ax] for ax in axes]))
    return a.mean(axis=axis, keepdims=keepdims)

  @staticmethod
  def backward(ctx: Any, grad_in: np.ndarray) -> Tuple[np.ndarray]:
    grad_in = np.asarray(grad_in, dtype=np.float64) / ctx.n
    if ctx.axis is not None and not ctx.keepdims:
      grad_in = np.expand_dims(grad_in, ctx.axis)
    return (np.broadcast_to(grad_in, ctx.in_shape).copy(),)


class Reshape(Function):
  @staticmethod
  def forward(ctx: Any, x: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    ctx.in_shape = x.shape
    return x.reshape(shape)

  @staticmethod
  def backward(ctx: Any, grad_in: np.ndarray) -> Tuple[np.ndarray]:
    return (grad_in.reshape(ctx.in_shape),)


class Transpose(Function):
  @staticmethod
  def forward(ctx: Any, x: np.ndarray) -> np.ndarray:
    return x.T

  @staticmethod
  def backward(ctx: Any, grad_in: np.ndarray) -> Tuple[np.ndarray]:
    return (grad_in.T,)


class Softmax(Function):
  """
  See https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
  """

  @staticmethod
  def forward(ctx: Any, x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    s = e / e.sum(axis=axis, keepdims=True)
    ctx.save_for_backward(s)
    ctx.axis = axis
    return s

  @staticmethod
  def backward(ctx: Any, grad_in: np.ndarray) -> Tuple[np.ndarray]:
    (s,) = ctx._saved
    return (s * (grad_in - (grad_in * s).sum(axis=ctx.axis, keepdims=True)),)


class CrossEntropy(Function):
  @staticmethod
  def forward(ctx: Any, x: np.ndarray, targets: np.ndarray) -> np.ndarray:
    t = np.asarray(targets).astype(np.int64)
    z = x - x.max(axis=1, keepdims=True)  # stability shift
    logsumexp = np.log(np.exp(z).sum(axis=1, keepdims=True))
    logp = z - logsumexp
    n = x.shape[0]
    loss = -logp[np.arange(n), t].mean()
    ctx.save_for_backward(np.exp(logp))  # softmax probs
    ctx.t = t
    ctx.n = n
    return loss

  @staticmethod
  def backward(ctx: Any, grad_in: np.ndarray) -> Tuple[np.ndarray]:
    (probs,) = ctx._saved
    grad_logits = probs.copy()
    grad_logits[np.arange(ctx.n), ctx.t] -= 1.0  # softmax - onehot
    return (grad_in * grad_logits / ctx.n,)


def cross_entropy(logits: 'Data', targets: 'Data') -> 'Data':
  return CrossEntropy.apply(logits, targets)


# ---------------------------------------------------------------------------
# Graph traversal
# ---------------------------------------------------------------------------

def get_topological_order(data_node: 'Data') -> List['Data']:
  topo_order: List['Data'] = []
  visited: Set['Data'] = set([])
  visiting: Set['Data'] = set([])

  def dfs(data_node: 'Data'):
    visiting.add(data_node)
    if data_node.grad_fn is None:
      topo_order.append(data_node)
      visiting.remove(data_node)
      visited.add(data_node)
      return

    for parent_node in data_node.grad_fn.parents:
      if parent_node in visiting:
        raise ValueError("Circular dependency found")
      if parent_node not in visited:
        dfs(parent_node)

    visited.add(data_node)
    topo_order.append(data_node)
    visiting.remove(data_node)
    return

  dfs(data_node)
  return topo_order


# ---------------------------------------------------------------------------
# Data: the node of the computation graph
# ---------------------------------------------------------------------------

class Data:
  def __init__(self, data: np.ndarray, requires_grad: bool = False,
               grad_fn: 'Function' = None):
    self.data = data
    self.grad: np.ndarray | None = None
    self.requires_grad = requires_grad
    self.grad_fn = grad_fn

  def __repr__(self) -> str:
    return (f"Data(data={self.data}, grad={self.grad}, "
            f"requires_grad={self.requires_grad}, grad_fn={self.grad_fn})")

  # --- ops ---
  def __add__(self, other: 'Data') -> 'Data':
    return Add.apply(self, other)

  def __radd__(self, other: 'Data') -> 'Data':
    return Add.apply(other, self)

  def __matmul__(self, other: 'Data') -> 'Data':
    return MatMul.apply(self, other)

  def __sub__(self, other: 'Data') -> 'Data':
    return Sub.apply(self, other)

  def __rsub__(self, other: 'Data') -> 'Data':
    return Sub.apply(other, self)

  def __mul__(self, other: 'Data') -> 'Data':
    return Mul.apply(self, other)

  def __rmul__(self, other: 'Data') -> 'Data':
    return Mul.apply(other, self)

  def __truediv__(self, other: 'Data') -> 'Data':
    return Div.apply(self, other)

  def __rtruediv__(self, other: 'Data') -> 'Data':
    return Div.apply(other, self)

  def __neg__(self) -> 'Data':
    return Neg.apply(self)

  def __pow__(self, exponent: int) -> 'Data':
    return Pow.apply(self, exponent=exponent)

  def exp(self) -> 'Data':
    return Exp.apply(self)

  def log(self) -> 'Data':
    return Log.apply(self)

  def relu(self) -> 'Data':
    return ReLu.apply(self)

  def softmax(self, axis: int = -1) -> 'Data':
    return Softmax.apply(self, axis=axis)

  def sum(self, axis: Union[int, Tuple[int, ...], None] = None,
          keepdims: bool = False) -> 'Data':
    return Sum.apply(self, axis=axis, keepdims=keepdims)

  def mean(self, axis: Union[int, Tuple[int, ...], None] = None,
           keepdims: bool = False) -> 'Data':
    return Mean.apply(self, axis=axis, keepdims=keepdims)

  def reshape(self, shape: Tuple[int, ...]) -> 'Data':
    # `shape` must be passed as a kwarg: apply() treats positional args as Data
    return Reshape.apply(self, shape=shape)

  @property
  def T(self) -> 'Data':
    return Transpose.apply(self)

  # --- backward ---
  def backward(self) -> None:
    """Backpropagate through the graph, accumulating gradients in the leaves.

    Gradients of the intermediate nodes live in `grad_dict` and are discarded
    at the end of the call; only the leaves keep (and accumulate) a `.grad`.
    """
    grad_dict = {}
    # Initialize the gradient for the start_node based on its data shape
    grad_dict[id(self)] = np.ones_like(self.data)

    for data_node in reversed(get_topological_order(self)):
      if not data_node.requires_grad:  # no need to explore
        continue

      # If the current node is a leaf node, accumulate its gradient
      if data_node.requires_grad and not data_node.grad_fn:
        if data_node.grad is not None:
          data_node.grad += grad_dict[id(data_node)]
        else:
          data_node.grad = grad_dict[id(data_node)]
        continue

      # Pass down the gradient to parents
      data_node_grad = grad_dict[id(data_node)]
      grads_out = type(data_node.grad_fn).backward(data_node.grad_fn, data_node_grad)

      for grad_out, data_node_parent in zip(grads_out, data_node.grad_fn.parents):
        if data_node_parent.requires_grad:
          parent_id = id(data_node_parent)
          if parent_id in grad_dict:
            grad_dict[parent_id] += unbroadcast(grad_out, data_node_parent.data.shape)
          else:
            grad_dict[parent_id] = unbroadcast(grad_out, data_node_parent.data.shape)
    return


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class Layer:
  def parameters(self) -> List['Data']:
    """Recursively collect every learnable Data reachable from this layer."""
    seen: Set['Data'] = set()
    params: List['Data'] = []

    def visit(node: Any):
      if isinstance(node, Data):
        if node.requires_grad and node not in seen:
          seen.add(node)
          params.append(node)

      elif isinstance(node, Layer):
        for next_node in node.__dict__.values():
          visit(next_node)

      elif isinstance(node, (list, tuple)):
        for next_node in node:
          visit(next_node)

      elif isinstance(node, dict):
        for next_node in node.values():
          visit(next_node)

    visit(self)

    return params

  def zero_grad(self) -> None:
    for data in self.parameters():
      data.grad = None

  def __call__(self, *a: Any, **kw: Any) -> 'Data':
    return self.forward(*a, **kw)


class Linear(Layer):
  def __init__(self, dim_in: int, dim_out: int):
    self.W = Data(np.random.randn(dim_out, dim_in) * np.sqrt(2.0 / dim_in),
                  requires_grad=True)
    self.b = Data(np.zeros(dim_out), requires_grad=True)

  def forward(self, X: 'Data', **kw: Any) -> 'Data':
    return X @ self.W.T + self.b


class MLP(Layer):
  def __init__(self, sizes: List[int]):
    self.linear_layers = [
        Linear(size_in, size_out)
        for size_in, size_out in zip(sizes[:-1], sizes[1:])
    ]

  def forward(self, x: 'Data') -> 'Data':
    h = x
    for i, linear_layer in enumerate(self.linear_layers):
      if i < len(self.linear_layers) - 1:
        h = linear_layer(h).relu()
      else:
        h = linear_layer(h)
    return h


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------

class SGD:
  def __init__(self, params: List['Data'], lr: float, momentum: float = 0.0):
    self.params = params
    self.lr = lr
    self.momentum = momentum
    self.buf = [np.zeros_like(p.data) for p in self.params]

  def step(self) -> None:
    for i, p in enumerate(self.params):
      if p.grad is None:
        continue
      if self.momentum:
        self.buf[i] = self.momentum * self.buf[i] + p.grad
        p.data -= self.lr * self.buf[i]
      else:
        p.data -= self.lr * p.grad

  def zero_grad(self) -> None:
    for p in self.params:
      p.grad = None
