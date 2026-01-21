import pytest
import torch

from torchrdit.interface import _compile_vars


@pytest.mark.parametrize(
    "expr, x_value",
    [
        ("abs($x)", 0.5),
        ("real($x)", 0.5),
        ("imag($x)", 0.5),
        ("sin($x)", 0.5),
        ("cos($x)", 0.5),
        ("tan($x)", 0.1),
        ("sqrt($x)", 0.5),
        ("exp($x)", 0.5),
        ("log($x)", 0.5),
    ],
)
def test_var_expression_functions_are_differentiable(expr, x_value):
    x = torch.tensor(x_value, requires_grad=True)
    compiled = _compile_vars({"$x": x, "$y": expr}, device="cpu")
    out = compiled.evaluate()
    y = out["$y"]
    assert torch.is_tensor(y)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize(
    "expr, expected_grad",
    [
        ("sum([$x, $x * 2])", 3.0),
        ("mean([$x, $x * 2])", 1.5),
        ("min([$x, $x * 2])", 1.0),
        ("max([$x, $x * 2])", 2.0),
    ],
)
def test_var_expression_reductions_have_gradients(expr, expected_grad):
    x = torch.tensor(0.5, requires_grad=True)
    compiled = _compile_vars({"$x": x, "$y": expr}, device="cpu")
    out = compiled.evaluate()
    y = out["$y"]
    assert torch.is_tensor(y)
    y.backward()
    assert x.grad is not None
    assert float(x.grad) == pytest.approx(expected_grad)
