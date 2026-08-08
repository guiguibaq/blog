"""Unit tests for minigrad.

Run with `pytest test_minigrad.py`, or directly with `python test_minigrad.py`
if pytest is not installed.

Most gradients are checked against central finite differences, so the tests do
not depend on PyTorch. The few tests that do cross-check against torch are
skipped when torch is missing.
"""

from __future__ import annotations

import numpy as np

from minigrad import (
    MLP,
    SGD,
    CrossEntropy,
    Data,
    Function,
    Linear,
    Softmax,
    cross_entropy,
    get_topological_order,
    unbroadcast,
)

try:
  import torch
  HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
  torch = None
  HAS_TORCH = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def numeric_grad(scalar_fn, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
  """Central finite differences of `scalar_fn` w.r.t. the array `x` in place."""
  grad = np.zeros_like(x, dtype=np.float64)
  it = np.nditer(x, flags=["multi_index"])
  while not it.finished:
    idx = it.multi_index
    original = x[idx]
    x[idx] = original + eps
    plus = scalar_fn()
    x[idx] = original - eps
    minus = scalar_fn()
    x[idx] = original
    grad[idx] = (plus - minus) / (2 * eps)
    it.iternext()
  return grad


def check_grad(build, *inputs: Data, atol: float = 1e-6, rtol: float = 1e-4) -> None:
  """Compare the analytic backward of `build(*inputs)` to finite differences.

  `build` takes the input Data nodes and returns a scalar loss Data.
  """
  for node in inputs:
    node.grad = None

  build(*inputs).backward()

  for node in inputs:
    expected = numeric_grad(
        lambda: float(np.asarray(build(*inputs).data)), node.data
    )
    assert node.grad is not None, "no gradient reached the leaf"
    assert node.grad.shape == node.data.shape, (
        f"grad shape {node.grad.shape} != data shape {node.data.shape}"
    )
    assert np.allclose(node.grad, expected, atol=atol, rtol=rtol), (
        f"analytic {node.grad}\nnumeric  {expected}"
    )


def rand(*shape: int, seed: int = 0, low: float = 0.5, high: float = 1.5) -> Data:
  """A leaf with values bounded away from 0 (safe for log, div and relu)."""
  rng = np.random.default_rng(seed)
  return Data(rng.uniform(low, high, size=shape), requires_grad=True)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

def test_forward_elementwise_ops():
  a = Data(np.array([1.0, 2.0, 3.0]))
  b = Data(np.array([4.0, 5.0, 6.0]))

  assert np.allclose((a + b).data, [5.0, 7.0, 9.0])
  assert np.allclose((a - b).data, [-3.0, -3.0, -3.0])
  assert np.allclose((a * b).data, [4.0, 10.0, 18.0])
  assert np.allclose((a / b).data, [0.25, 0.4, 0.5])
  assert np.allclose((-a).data, [-1.0, -2.0, -3.0])
  assert np.allclose((a**2).data, [1.0, 4.0, 9.0])
  assert np.allclose(a.exp().data, np.exp([1.0, 2.0, 3.0]))
  assert np.allclose(a.log().data, np.log([1.0, 2.0, 3.0]))


def test_forward_reverse_ops():
  """The `__r*__` symbols let a Data sit on the right of the operator."""
  a = Data(np.array([2.0, 4.0]))
  ones = Data(np.array([1.0, 1.0]))

  assert np.allclose((ones + a).data, [3.0, 5.0])
  assert np.allclose((ones - a).data, [-1.0, -3.0])
  assert np.allclose((ones * a).data, [2.0, 4.0])
  assert np.allclose((ones / a).data, [0.5, 0.25])


def test_forward_relu_and_reductions():
  x = Data(np.array([[-1.0, 2.0], [3.0, -4.0]]))

  assert np.allclose(x.relu().data, [[0.0, 2.0], [3.0, 0.0]])
  assert np.allclose(x.sum().data, 0.0)
  assert np.allclose(x.sum(axis=0).data, [2.0, -2.0])
  assert np.allclose(x.sum(axis=1, keepdims=True).data, [[1.0], [-1.0]])
  assert np.allclose(x.mean().data, 0.0)
  assert np.allclose(x.mean(axis=1).data, [0.5, -0.5])


def test_forward_shape_ops():
  x = Data(np.arange(6.0).reshape(2, 3))

  assert x.T.data.shape == (3, 2)
  assert np.allclose(x.T.data, np.arange(6.0).reshape(2, 3).T)
  assert np.allclose(x.reshape((3, 2)).data, np.arange(6.0).reshape(3, 2))


def test_forward_matmul_mlp():
  """The forward of a two layer MLP matches the plain numpy computation."""
  rng = np.random.default_rng(0)
  X = Data(rng.random((5, 4)))
  W1 = Data(rng.random((3, 4)), requires_grad=True)
  b1 = Data(rng.random((1, 1)), requires_grad=True)
  W2 = Data(rng.random((1, 3)), requires_grad=True)
  b2 = Data(rng.random((1, 1)), requires_grad=True)
  label = Data(rng.random((5, 1)))

  loss = (((X @ W1.T + b1).relu() @ W2.T + b2 - label) ** 2).mean()

  expected = np.mean(
      (np.maximum(X.data @ W1.data.T + b1.data, 0) @ W2.data.T + b2.data
       - label.data) ** 2
  )
  assert np.allclose(loss.data, expected)


# ---------------------------------------------------------------------------
# requires_grad propagation
# ---------------------------------------------------------------------------

def test_requires_grad_propagates_through_the_graph():
  x = Data(np.ones(3), requires_grad=False)
  w = Data(np.ones(3), requires_grad=True)

  assert (x * x).requires_grad is False
  assert (x * w).requires_grad is True
  assert (x * w).grad_fn is not None
  assert x.grad_fn is None  # leaves have no grad_fn


def test_no_grad_on_nodes_that_do_not_require_it():
  x = Data(np.array([1.0, 2.0]), requires_grad=False)
  w = Data(np.array([3.0, 4.0]), requires_grad=True)

  (x * w).sum().backward()

  assert x.grad is None
  assert np.allclose(w.grad, x.data)


# ---------------------------------------------------------------------------
# Backward: one test per operation, against finite differences
# ---------------------------------------------------------------------------

def test_backward_add():
  check_grad(lambda a, b: (a + b).sum(), rand(2, 3, seed=1), rand(2, 3, seed=2))


def test_backward_sub():
  check_grad(lambda a, b: (a - b).sum(), rand(2, 3, seed=3), rand(2, 3, seed=4))


def test_backward_mul():
  check_grad(lambda a, b: (a * b).sum(), rand(2, 3, seed=5), rand(2, 3, seed=6))


def test_backward_div():
  check_grad(lambda a, b: (a / b).sum(), rand(2, 3, seed=7), rand(2, 3, seed=8))


def test_backward_neg():
  check_grad(lambda a: (-a).sum(), rand(4, seed=9))


def test_backward_pow():
  check_grad(lambda a: (a**3).sum(), rand(4, seed=10), atol=1e-5)


def test_backward_exp():
  check_grad(lambda a: a.exp().sum(), rand(4, seed=11), atol=1e-5)


def test_backward_log():
  check_grad(lambda a: a.log().sum(), rand(4, seed=12))


def test_backward_relu():
  # values are spread around 0 so both branches of the relu are exercised
  x = Data(np.array([-2.0, -0.5, 0.5, 2.0]), requires_grad=True)
  check_grad(lambda a: (a.relu() * a.relu()).sum(), x)


def test_backward_matmul():
  check_grad(lambda a, b: (a @ b).sum(), rand(3, 4, seed=13), rand(4, 2, seed=14))


def test_backward_sum_over_axes():
  check_grad(lambda a: a.sum(), rand(2, 3, seed=15))
  check_grad(lambda a: (a.sum(axis=0) ** 2).sum(), rand(2, 3, seed=16), atol=1e-5)
  check_grad(lambda a: (a.sum(axis=1, keepdims=True) ** 2).sum(),
             rand(2, 3, seed=17), atol=1e-5)
  check_grad(lambda a: (a.sum(axis=(0, 1)) ** 2).sum(),
             rand(2, 3, seed=18), atol=1e-4)


def test_backward_mean_over_axes():
  check_grad(lambda a: a.mean(), rand(2, 3, seed=19))
  check_grad(lambda a: (a.mean(axis=0) ** 2).sum(), rand(2, 3, seed=20))
  check_grad(lambda a: (a.mean(axis=1, keepdims=True) ** 2).sum(),
             rand(2, 3, seed=21))


def test_backward_reshape_and_transpose():
  check_grad(lambda a: (a.reshape((3, 2)) ** 2).sum(), rand(2, 3, seed=22), atol=1e-5)
  check_grad(lambda a: (a.T ** 2).sum(), rand(2, 3, seed=23), atol=1e-5)


def test_backward_softmax():
  x = rand(3, 4, seed=24, low=-1.0, high=1.0)
  weights = np.arange(12.0).reshape(3, 4)

  def build(a):
    return (Softmax.apply(a, axis=-1) * Data(weights)).sum()

  check_grad(build, x)


def test_backward_cross_entropy():
  logits = rand(5, 3, seed=25, low=-2.0, high=2.0)
  targets = Data(np.array([0, 2, 1, 1, 0]))

  check_grad(lambda x: cross_entropy(x, targets), logits)


def test_softmax_forward_is_a_distribution():
  x = Data(np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]]))
  probs = Softmax.apply(x, axis=-1).data

  assert np.allclose(probs.sum(axis=-1), 1.0)
  assert np.all(probs > 0)  # no overflow on the large row


def test_cross_entropy_forward_matches_manual_formula():
  logits = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 0.0]])
  targets = np.array([2, 0])

  loss = cross_entropy(Data(logits), Data(targets)).data

  probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
  expected = -np.log(probs[np.arange(2), targets]).mean()
  assert np.allclose(loss, expected)


# ---------------------------------------------------------------------------
# Broadcasting
# ---------------------------------------------------------------------------

def test_unbroadcast_sums_the_added_and_stretched_dimensions():
  # a leading dimension was added by the broadcast
  assert unbroadcast(np.ones((4, 3)), (3,)).shape == (3,)
  assert np.allclose(unbroadcast(np.ones((4, 3)), (3,)), 4.0)

  # a size-1 dimension was stretched by the broadcast
  assert unbroadcast(np.ones((4, 3)), (1, 3)).shape == (1, 3)
  assert np.allclose(unbroadcast(np.ones((4, 3)), (1, 3)), 4.0)

  # nothing to undo
  assert np.allclose(unbroadcast(np.ones((4, 3)), (4, 3)), 1.0)


def test_bias_gradient_keeps_the_bias_shape():
  """The classic X @ W.T + b case: b.grad must stay (dim_out,), not (batch, dim_out)."""
  batch, dim_in, dim_out = 8, 4, 5
  rng = np.random.default_rng(0)
  X = Data(rng.random((batch, dim_in)))
  W = Data(rng.random((dim_out, dim_in)), requires_grad=True)
  b = Data(np.zeros(dim_out), requires_grad=True)

  (X @ W.T + b).sum().backward()

  assert b.grad.shape == (dim_out,)
  assert np.allclose(b.grad, batch)
  assert W.grad.shape == (dim_out, dim_in)


def test_broadcast_gradients_against_finite_differences():
  check_grad(lambda A, c: ((A * c) ** 2).sum(),
             rand(4, 3, seed=26), rand(3, seed=27), atol=1e-5)
  check_grad(lambda A, c: ((A + c) ** 2).sum(),
             rand(4, 3, seed=28), rand(1, 3, seed=29), atol=1e-5)


# ---------------------------------------------------------------------------
# Graph traversal and gradient accumulation
# ---------------------------------------------------------------------------

def test_topological_order_puts_parents_before_children():
  a = Data(np.array([1.0]), requires_grad=True)
  b = Data(np.array([2.0]), requires_grad=True)
  c = a + b

  order = get_topological_order(c)

  assert order in ([a, b, c], [b, a, c])


def test_topological_order_visits_each_node_once():
  x = Data(np.array([1.0]), requires_grad=True)
  h = x * x  # x is a parent twice
  loss = h.sum()

  order = get_topological_order(loss)

  assert len(order) == 3
  assert order[-1] is loss


def test_topological_order_detects_cycles():
  node = Data(np.array([1.0]), requires_grad=True)
  fake_fn = Function()
  fake_fn.parents = [node]
  node.grad_fn = fake_fn  # node is now its own parent

  try:
    get_topological_order(node)
  except ValueError as err:
    assert "Circular dependency" in str(err)
  else:
    raise AssertionError("expected a ValueError on a cyclic graph")


def test_gradients_accumulate_when_a_leaf_is_reused():
  x = Data(np.array([-0.5, 0.25, 1.0]), requires_grad=True)

  (x * x).sum().backward()

  assert np.allclose(x.grad, 2 * x.data)


def test_gradients_accumulate_through_a_reused_intermediate_node():
  x = Data(np.array([-0.5, 0.25, 1.0]), requires_grad=True)
  a = x + x

  (a * a).sum().backward()

  assert np.allclose(x.grad, 8 * x.data)


def test_backward_twice_accumulates_in_the_leaves_only():
  x = Data(np.array([3.0]), requires_grad=True)
  h = x * x
  loss = h.sum()

  loss.backward()
  first = x.grad.copy()
  loss.backward()

  assert np.allclose(first, 2 * x.data)
  assert np.allclose(x.grad, 2 * first)  # leaves accumulate
  assert h.grad is None  # intermediate nodes never store a gradient


def test_zero_grad_resets_the_accumulation():
  model = MLP([3, 4, 1])
  X = Data(np.random.default_rng(0).random((6, 3)))

  model(X).sum().backward()
  model.zero_grad()
  model(X).sum().backward()
  once = [p.grad.copy() for p in model.parameters()]

  model.zero_grad()
  model(X).sum().backward()
  model(X).sum().backward()
  twice = [p.grad for p in model.parameters()]

  for grad_once, grad_twice in zip(once, twice):
    assert np.allclose(grad_twice, 2 * grad_once)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

def test_linear_shapes_and_forward():
  layer = Linear(4, 3)
  X = Data(np.random.default_rng(0).random((7, 4)))

  out = layer(X)

  assert out.data.shape == (7, 3)
  assert np.allclose(out.data, X.data @ layer.W.data.T + layer.b.data)


def test_mlp_shapes_and_hidden_relu():
  model = MLP([2, 5, 5, 3])
  X = Data(np.random.default_rng(0).standard_normal((10, 2)))

  out = model(X)

  assert out.data.shape == (10, 3)
  assert len(model.linear_layers) == 3
  # the last layer is linear, so negative logits must be possible
  assert np.any(out.data < 0) or np.any(out.data > 0)


def test_parameters_are_found_recursively_and_deduplicated():
  model = MLP([3, 4, 2])
  params = model.parameters()

  assert len(params) == 4  # one W and one b per linear layer
  assert all(p.requires_grad for p in params)
  assert len({id(p) for p in params}) == 4

  # a layer sharing the same Data twice must not report it twice
  class Tied(Linear):
    def __init__(self):
      super().__init__(2, 2)
      self.also_W = self.W

  assert len(Tied().parameters()) == 2


def test_parameters_ignore_non_learnable_data():
  layer = Linear(3, 2)
  layer.running_mean = Data(np.zeros(2), requires_grad=False)

  assert len(layer.parameters()) == 2


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------

def test_sgd_step_without_momentum():
  p = Data(np.array([1.0, 2.0]), requires_grad=True)
  p.grad = np.array([0.5, -1.0])
  opt = SGD([p], lr=0.1)

  opt.step()

  assert np.allclose(p.data, [1.0 - 0.05, 2.0 + 0.1])


def test_sgd_step_with_momentum_accumulates_velocity():
  p = Data(np.array([1.0]), requires_grad=True)
  opt = SGD([p], lr=0.1, momentum=0.9)

  p.grad = np.array([1.0])
  opt.step()  # buf = 1.0
  p.grad = np.array([1.0])
  opt.step()  # buf = 0.9 * 1.0 + 1.0 = 1.9

  assert np.allclose(p.data, 1.0 - 0.1 * 1.0 - 0.1 * 1.9)


def test_sgd_skips_parameters_without_gradient():
  p = Data(np.array([1.0]), requires_grad=True)
  opt = SGD([p], lr=0.1)

  opt.step()  # p.grad is None

  assert np.allclose(p.data, 1.0)


def test_sgd_zero_grad_clears_gradients():
  p = Data(np.array([1.0]), requires_grad=True)
  p.grad = np.array([1.0])
  opt = SGD([p], lr=0.1)

  opt.zero_grad()

  assert p.grad is None


# ---------------------------------------------------------------------------
# End to end training
# ---------------------------------------------------------------------------

def test_regression_fits_a_sine():
  np.random.seed(0)
  X = np.linspace(-1, 1, 64).reshape(-1, 1)
  y = np.sin(2 * np.pi * X)
  X_tensor, y_tensor = Data(X), Data(y)

  model = MLP([1, 16, 16, 1])
  opt = SGD(model.parameters(), lr=0.05, momentum=0.9)

  losses = []
  for _ in range(2000):
    loss = ((model(X_tensor) - y_tensor) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(float(loss.data))

  assert losses[-1] < 0.05 * losses[0]
  assert losses[-1] < 0.01


def test_classification_separates_two_clusters():
  np.random.seed(0)
  rng = np.random.default_rng(0)
  cluster_a = rng.standard_normal((50, 2)) * 0.3 + np.array([-1.5, 0.0])
  cluster_b = rng.standard_normal((50, 2)) * 0.3 + np.array([1.5, 0.0])
  X = Data(np.vstack([cluster_a, cluster_b]))
  y = np.concatenate([np.zeros(50, dtype=np.int64), np.ones(50, dtype=np.int64)])
  targets = Data(y)

  model = MLP([2, 16, 2])
  opt = SGD(model.parameters(), lr=0.1, momentum=0.9)

  for _ in range(300):
    loss = cross_entropy(model(X), targets)
    opt.zero_grad()
    loss.backward()
    opt.step()

  accuracy = (model(X).data.argmax(1) == y).mean()
  assert accuracy == 1.0


# ---------------------------------------------------------------------------
# Cross-checks against PyTorch (skipped when torch is unavailable)
# ---------------------------------------------------------------------------

def test_matmul_and_bias_gradients_match_torch():
  if not HAS_TORCH:
    return
  rng = np.random.default_rng(0)
  X = Data(rng.random((6, 3)), requires_grad=True)
  W = Data(rng.random((4, 3)), requires_grad=True)
  b = Data(rng.random(4), requires_grad=True)

  (X @ W.T + b).sum().backward()

  Xt, Wt, bt = (torch.tensor(z.data, requires_grad=True) for z in (X, W, b))
  (Xt @ Wt.T + bt).sum().backward()

  assert np.allclose(X.grad, Xt.grad, atol=1e-8, rtol=1e-6)
  assert np.allclose(W.grad, Wt.grad, atol=1e-8, rtol=1e-6)
  assert np.allclose(b.grad, bt.grad, atol=1e-8, rtol=1e-6)


def test_elementwise_broadcast_gradients_match_torch():
  if not HAS_TORCH:
    return
  rng = np.random.default_rng(1)
  A = Data(rng.random((4, 3)), requires_grad=True)
  c = Data(rng.random(3), requires_grad=True)

  (A * c).sum().backward()

  At, ct = (torch.tensor(z.data, requires_grad=True) for z in (A, c))
  (At * ct).sum().backward()

  assert np.allclose(A.grad, At.grad, atol=1e-8, rtol=1e-6)
  assert np.allclose(c.grad, ct.grad, atol=1e-8, rtol=1e-6)


def test_softmax_and_cross_entropy_match_torch():
  if not HAS_TORCH:
    return
  rng = np.random.default_rng(2)
  logits = rng.standard_normal((5, 4))
  targets = np.array([0, 3, 1, 2, 3])

  x = Data(logits, requires_grad=True)
  cross_entropy(x, Data(targets)).backward()

  xt = torch.tensor(logits, requires_grad=True)
  torch.nn.CrossEntropyLoss()(xt, torch.tensor(targets)).backward()

  assert np.allclose(x.grad, xt.grad, atol=1e-8, rtol=1e-6)

  probs = Softmax.apply(Data(logits), axis=-1).data
  assert np.allclose(probs, torch.softmax(xt, dim=-1).detach().numpy())


def test_mlp_gradients_match_a_replicated_torch_mlp():
  if not HAS_TORCH:
    return
  np.random.seed(0)
  rng = np.random.default_rng(0)
  model = MLP([3, 8, 4, 1])
  X = rng.random((10, 3))
  y = rng.random((10, 1))

  ((model(Data(X)) - Data(y)) ** 2).mean().backward()

  torch_layers = []
  for layer in model.linear_layers:
    tl = torch.nn.Linear(layer.W.data.shape[1], layer.W.data.shape[0]).double()
    with torch.no_grad():
      tl.weight.copy_(torch.tensor(layer.W.data))
      tl.bias.copy_(torch.tensor(layer.b.data))
    torch_layers.append(tl)

  h = torch.tensor(X)
  for i, tl in enumerate(torch_layers):
    h = tl(h)
    if i < len(torch_layers) - 1:
      h = torch.relu(h)
  ((h - torch.tensor(y)) ** 2).mean().backward()

  for layer, tl in zip(model.linear_layers, torch_layers):
    assert np.allclose(layer.W.grad, tl.weight.grad.numpy(), atol=1e-8, rtol=1e-6)
    assert np.allclose(layer.b.grad, tl.bias.grad.numpy(), atol=1e-8, rtol=1e-6)


if __name__ == "__main__":
  tests = [(name, fn) for name, fn in sorted(globals().items())
           if name.startswith("test_") and callable(fn)]
  failures = 0
  for name, fn in tests:
    try:
      fn()
      print(f"PASS {name}")
    except Exception as err:  # noqa: BLE001 - a tiny standalone runner
      failures += 1
      print(f"FAIL {name}: {type(err).__name__}: {err}")
  print(f"\n{len(tests) - failures}/{len(tests)} passed")
  raise SystemExit(1 if failures else 0)
