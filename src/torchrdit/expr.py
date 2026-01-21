from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple


class ExprNode:
    pass


@dataclass(frozen=True)
class Literal(ExprNode):
    value: Any


@dataclass(frozen=True)
class Name(ExprNode):
    ident: str


@dataclass(frozen=True)
class BinOp(ExprNode):
    op: str
    left: ExprNode
    right: ExprNode


@dataclass(frozen=True)
class UnaryOp(ExprNode):
    op: str
    operand: ExprNode


@dataclass(frozen=True)
class Call(ExprNode):
    func: str
    args: Tuple[ExprNode, ...]
    kwargs: Tuple[Tuple[str, ExprNode], ...]


@dataclass(frozen=True)
class Subscript(ExprNode):
    value: ExprNode
    key: ExprNode


@dataclass(frozen=True)
class SliceNode(ExprNode):
    lower: Optional[ExprNode]
    upper: Optional[ExprNode]
    step: Optional[ExprNode]


@dataclass(frozen=True)
class ListNode(ExprNode):
    elts: Tuple[ExprNode, ...]


@dataclass(frozen=True)
class TupleNode(ExprNode):
    elts: Tuple[ExprNode, ...]


@dataclass(frozen=True)
class DictNode(ExprNode):
    keys: Tuple[ExprNode, ...]
    values: Tuple[ExprNode, ...]


@dataclass(frozen=True)
class ParseRules:
    context: str
    allow_constants: bool
    allow_names: bool
    allow_binops: bool
    allow_unaryops: bool
    allow_calls: bool
    allow_call_kwargs: bool
    allow_subscript: bool
    allow_lists: bool
    allow_tuples: bool
    allow_dicts: bool
    allowed_functions: Optional[Iterable[str]] = None
    constant_predicate: Optional[Callable[[Any], bool]] = None
    call_arity: Optional[int] = None


_BIN_OPS: Dict[type, str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Pow: "**",
}

_UNARY_OPS: Dict[type, str] = {
    ast.UAdd: "+",
    ast.USub: "-",
}


def parse_expression(expr: str, *, path: str, rules: ParseRules, error_cls: type[Exception]) -> ExprNode:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise error_cls(f"Invalid {rules.context} at {path}: {exc.msg}") from exc
    return _parse_node(tree.body, path=path, rules=rules, error_cls=error_cls)


def _parse_node(node: ast.AST, *, path: str, rules: ParseRules, error_cls: type[Exception]) -> ExprNode:
    if isinstance(node, ast.Constant):
        if not rules.allow_constants:
            raise error_cls(f"Unsupported syntax in {rules.context} at {path}: Constant")
        if rules.constant_predicate is not None and not rules.constant_predicate(node.value):
            raise error_cls(f"Unsupported constant in {rules.context} at {path}: {node.value!r}")
        return Literal(node.value)

    if isinstance(node, ast.Name):
        if not rules.allow_names:
            raise error_cls(f"Unsupported syntax in {rules.context} at {path}: Name")
        return Name(node.id)

    if isinstance(node, ast.BinOp):
        if not rules.allow_binops:
            raise error_cls(f"Unsupported syntax in {rules.context} at {path}: BinOp")
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise error_cls(f"Unsupported binary operator in {rules.context} at {path}: {op_type}")
        return BinOp(
            _BIN_OPS[op_type],
            _parse_node(node.left, path=path, rules=rules, error_cls=error_cls),
            _parse_node(node.right, path=path, rules=rules, error_cls=error_cls),
        )

    if isinstance(node, ast.UnaryOp):
        if not rules.allow_unaryops:
            raise error_cls(f"Unsupported syntax in {rules.context} at {path}: UnaryOp")
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise error_cls(f"Unsupported unary operator in {rules.context} at {path}: {op_type}")
        return UnaryOp(
            _UNARY_OPS[op_type],
            _parse_node(node.operand, path=path, rules=rules, error_cls=error_cls),
        )

    if isinstance(node, ast.Call):
        if not rules.allow_calls:
            raise error_cls(f"Unsupported syntax in {rules.context} at {path}: Call")
        if not isinstance(node.func, ast.Name):
            raise error_cls(f"Only simple calls are allowed in {rules.context} at {path}")
        func_name = node.func.id
        allowed = set(rules.allowed_functions or [])
        if rules.allowed_functions is not None and func_name not in allowed:
            raise error_cls(f"Unknown function in {rules.context} at {path}: {func_name!r}")
        if not rules.allow_call_kwargs and node.keywords:
            raise error_cls(f"Keyword args are not allowed in {rules.context} at {path}")
        if rules.call_arity is not None and len(node.args) != rules.call_arity:
            raise error_cls(f"Calls in {rules.context} must have {rules.call_arity} positional args at {path}")
        args = tuple(_parse_node(a, path=path, rules=rules, error_cls=error_cls) for a in node.args)
        kwargs = tuple((kw.arg, _parse_node(kw.value, path=path, rules=rules, error_cls=error_cls)) for kw in node.keywords)
        return Call(func_name, args=args, kwargs=kwargs)

    if isinstance(node, ast.Subscript):
        if not rules.allow_subscript:
            raise error_cls(f"Unsupported syntax in {rules.context} at {path}: Subscript")
        value = _parse_node(node.value, path=path, rules=rules, error_cls=error_cls)
        key = _parse_slice(node.slice, path=path, rules=rules, error_cls=error_cls)
        return Subscript(value=value, key=key)

    if isinstance(node, ast.List):
        if not rules.allow_lists:
            raise error_cls(f"Unsupported syntax in {rules.context} at {path}: List")
        return ListNode(tuple(_parse_node(e, path=path, rules=rules, error_cls=error_cls) for e in node.elts))

    if isinstance(node, ast.Tuple):
        if not rules.allow_tuples:
            raise error_cls(f"Unsupported syntax in {rules.context} at {path}: Tuple")
        return TupleNode(tuple(_parse_node(e, path=path, rules=rules, error_cls=error_cls) for e in node.elts))

    if isinstance(node, ast.Dict):
        if not rules.allow_dicts:
            raise error_cls(f"Unsupported syntax in {rules.context} at {path}: Dict")
        keys = tuple(_parse_node(k, path=path, rules=rules, error_cls=error_cls) for k in node.keys)
        vals = tuple(_parse_node(v, path=path, rules=rules, error_cls=error_cls) for v in node.values)
        return DictNode(keys=keys, values=vals)

    raise error_cls(f"Unsupported syntax in {rules.context} at {path}: {type(node).__name__}")


def _parse_slice(node: ast.AST, *, path: str, rules: ParseRules, error_cls: type[Exception]) -> ExprNode:
    if isinstance(node, ast.Slice):
        lower = _parse_node(node.lower, path=path, rules=rules, error_cls=error_cls) if node.lower is not None else None
        upper = _parse_node(node.upper, path=path, rules=rules, error_cls=error_cls) if node.upper is not None else None
        step = _parse_node(node.step, path=path, rules=rules, error_cls=error_cls) if node.step is not None else None
        return SliceNode(lower=lower, upper=upper, step=step)
    return _parse_node(node, path=path, rules=rules, error_cls=error_cls)


def evaluate_expression(
    node: ExprNode,
    env: Mapping[str, Any],
    *,
    functions: Mapping[str, Callable[..., Any]],
    path: str,
    context: str,
    error_cls: type[Exception],
    name_label: str = "name",
) -> Any:
    if isinstance(node, Literal):
        return node.value

    if isinstance(node, Name):
        if node.ident in env:
            return env[node.ident]
        raise error_cls(f"Unknown {name_label} in {context} at {path}: {node.ident!r}")

    if isinstance(node, BinOp):
        left = evaluate_expression(
            node.left, env, functions=functions, path=path, context=context, error_cls=error_cls, name_label=name_label
        )
        right = evaluate_expression(
            node.right, env, functions=functions, path=path, context=context, error_cls=error_cls, name_label=name_label
        )
        if node.op == "+":
            return left + right
        if node.op == "-":
            return left - right
        if node.op == "*":
            return left * right
        if node.op == "/":
            return left / right
        if node.op == "**":
            return left**right
        raise error_cls(f"Unsupported binary operator in {context} at {path}: {node.op!r}")

    if isinstance(node, UnaryOp):
        operand = evaluate_expression(
            node.operand, env, functions=functions, path=path, context=context, error_cls=error_cls, name_label=name_label
        )
        if node.op == "+":
            return +operand
        if node.op == "-":
            return -operand
        raise error_cls(f"Unsupported unary operator in {context} at {path}: {node.op!r}")

    if isinstance(node, Call):
        if node.func not in functions:
            raise error_cls(f"Unknown function in {context} at {path}: {node.func!r}")
        func = functions[node.func]
        args = [
            evaluate_expression(a, env, functions=functions, path=path, context=context, error_cls=error_cls, name_label=name_label)
            for a in node.args
        ]
        kwargs = {
            key: evaluate_expression(
                val, env, functions=functions, path=path, context=context, error_cls=error_cls, name_label=name_label
            )
            for key, val in node.kwargs
        }
        return func(*args, **kwargs)

    if isinstance(node, Subscript):
        value = evaluate_expression(
            node.value, env, functions=functions, path=path, context=context, error_cls=error_cls, name_label=name_label
        )
        key = evaluate_expression(
            node.key, env, functions=functions, path=path, context=context, error_cls=error_cls, name_label=name_label
        )
        return value[key]

    if isinstance(node, SliceNode):
        lower = (
            evaluate_expression(
                node.lower, env, functions=functions, path=path, context=context, error_cls=error_cls, name_label=name_label
            )
            if node.lower is not None
            else None
        )
        upper = (
            evaluate_expression(
                node.upper, env, functions=functions, path=path, context=context, error_cls=error_cls, name_label=name_label
            )
            if node.upper is not None
            else None
        )
        step = (
            evaluate_expression(
                node.step, env, functions=functions, path=path, context=context, error_cls=error_cls, name_label=name_label
            )
            if node.step is not None
            else None
        )
        return slice(lower, upper, step)

    if isinstance(node, ListNode):
        return [
            evaluate_expression(e, env, functions=functions, path=path, context=context, error_cls=error_cls, name_label=name_label)
            for e in node.elts
        ]

    if isinstance(node, TupleNode):
        return tuple(
            evaluate_expression(
                e, env, functions=functions, path=path, context=context, error_cls=error_cls, name_label=name_label
            )
            for e in node.elts
        )

    if isinstance(node, DictNode):
        keys = [
            evaluate_expression(k, env, functions=functions, path=path, context=context, error_cls=error_cls, name_label=name_label)
            for k in node.keys
        ]
        values = [
            evaluate_expression(v, env, functions=functions, path=path, context=context, error_cls=error_cls, name_label=name_label)
            for v in node.values
        ]
        return dict(zip(keys, values))

    raise error_cls(f"Unsupported syntax in {context} at {path}: {type(node).__name__}")
