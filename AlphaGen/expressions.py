from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from .config import CONSTANTS, DELTA_TIMES, FEATURES


class ExpressionError(ValueError):
    pass


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    arity: int
    kind: str
    qlib_name: str | None = None

    @property
    def out_name(self) -> str:
        return self.qlib_name or self.name


OPERATORS: list[OperatorSpec] = [
    OperatorSpec("Abs", 1, "unary"),
    OperatorSpec("Log", 1, "unary"),
    OperatorSpec("CSRank", 1, "unary"),
    OperatorSpec("Add", 2, "binary"),
    OperatorSpec("Sub", 2, "binary"),
    OperatorSpec("Mul", 2, "binary"),
    OperatorSpec("Div", 2, "binary"),
    OperatorSpec("Pow", 2, "binary", "Power"),
    OperatorSpec("Greater", 2, "binary"),
    OperatorSpec("Less", 2, "binary"),
    OperatorSpec("Ref", 2, "rolling"),
    OperatorSpec("Mean", 2, "rolling"),
    OperatorSpec("Sum", 2, "rolling"),
    OperatorSpec("Std", 2, "rolling"),
    OperatorSpec("Var", 2, "rolling"),
    OperatorSpec("Max", 2, "rolling"),
    OperatorSpec("Min", 2, "rolling"),
    OperatorSpec("Med", 2, "rolling"),
    OperatorSpec("Mad", 2, "rolling"),
    OperatorSpec("Delta", 2, "rolling"),
    OperatorSpec("WMA", 2, "rolling"),
    OperatorSpec("EMA", 2, "rolling"),
    OperatorSpec("Cov", 3, "pair_rolling"),
    OperatorSpec("Corr", 3, "pair_rolling"),
]
SEARCH_OPERATOR_EXCLUDES = {"Pow"}
OPERATORS_NO_CSRANK = [op for op in OPERATORS if op.name != "CSRank" and op.name not in SEARCH_OPERATOR_EXCLUDES]
SEARCH_OPERATORS_WITH_CSRANK = [op for op in OPERATORS if op.name not in SEARCH_OPERATOR_EXCLUDES]
OPERATOR_BY_NAME = {op.name.lower(): op for op in OPERATORS}
OPERATOR_BY_NAME["power"] = OPERATOR_BY_NAME["pow"]


@dataclass(frozen=True)
class Expr:
    kind: str
    value: str | float | int | OperatorSpec
    args: tuple["Expr", ...] = ()

    @property
    def is_featured(self) -> bool:
        if self.kind == "feature":
            return True
        if self.kind in {"constant", "delta"}:
            return False
        return any(arg.is_featured for arg in self.args)

    @property
    def is_delta(self) -> bool:
        return self.kind == "delta"

    def __str__(self) -> str:
        return self.to_qlib()

    def to_qlib(self) -> str:
        if self.kind == "feature":
            return f"${self.value}"
        if self.kind == "constant":
            return format_number(float(self.value))
        if self.kind == "delta":
            return str(int(self.value))
        op = self.value
        assert isinstance(op, OperatorSpec)
        return f"{op.out_name}({','.join(arg.to_qlib() for arg in self.args)})"


def format_number(value: float) -> str:
    if not math.isfinite(value):
        raise ExpressionError(f"Non-finite numeric value is not supported: {value}")
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def feature_expr(name: str) -> Expr:
    name = name.lower()
    if name not in FEATURES:
        raise ExpressionError(f"Unsupported feature: {name}")
    return Expr("feature", name)


def constant_expr(value: float) -> Expr:
    return Expr("constant", float(value))


def delta_expr(value: int) -> Expr:
    if int(value) <= 0:
        raise ExpressionError(f"Delta time must be positive: {value}")
    return Expr("delta", int(value))


def call_expr(op: OperatorSpec, args: Sequence[Expr]) -> Expr:
    if len(args) != op.arity:
        raise ExpressionError(f"{op.name} expects {op.arity} args, got {len(args)}")
    if op.kind == "unary":
        if not args[0].is_featured or args[0].is_delta:
            raise ExpressionError(f"{op.name} expects one featured expression")
    elif op.kind == "binary":
        if not any(arg.is_featured for arg in args) or any(arg.is_delta for arg in args):
            raise ExpressionError(f"{op.name} expects normal expressions")
    elif op.kind == "rolling":
        if not args[0].is_featured or not args[1].is_delta:
            raise ExpressionError(f"{op.name} expects expression and delta time")
    elif op.kind == "pair_rolling":
        if not args[0].is_featured or not args[1].is_featured or not args[2].is_delta:
            raise ExpressionError(f"{op.name} expects two expressions and delta time")
    return Expr("call", op, tuple(args))


@dataclass(frozen=True)
class Action:
    kind: str
    value: int | float | str | OperatorSpec | None = None

    def label(self) -> str:
        if isinstance(self.value, OperatorSpec):
            return self.value.name
        if self.kind == "feature":
            return f"${self.value}"
        return str(self.value if self.value is not None else self.kind)


class ActionCodec:
    def __init__(self, include_csrank: bool = False) -> None:
        operators = SEARCH_OPERATORS_WITH_CSRANK if include_csrank else OPERATORS_NO_CSRANK
        self.actions: list[Action] = []
        self.actions.extend(Action("operator", op) for op in operators)
        self.actions.extend(Action("feature", feature) for feature in FEATURES)
        self.actions.extend(Action("constant", value) for value in CONSTANTS)
        self.actions.extend(Action("delta", value) for value in DELTA_TIMES)
        self.actions.append(Action("stop"))

    def __len__(self) -> int:
        return len(self.actions)

    @property
    def stop_index(self) -> int:
        return len(self.actions) - 1

    def action(self, idx: int) -> Action:
        return self.actions[idx]


class ExpressionBuilder:
    def __init__(self) -> None:
        self.stack: list[Expr] = []

    def copy(self) -> "ExpressionBuilder":
        other = ExpressionBuilder()
        other.stack = list(self.stack)
        return other

    def is_valid(self) -> bool:
        return len(self.stack) == 1 and self.stack[0].is_featured

    def tree(self) -> Expr:
        if not self.is_valid():
            raise ExpressionError(f"Invalid expression stack: {self.stack}")
        return self.stack[0]

    def valid_action_mask(self, codec: ActionCodec) -> list[bool]:
        return [self.is_action_valid(action) for action in codec.actions]

    def is_action_valid(self, action: Action) -> bool:
        if action.kind == "stop":
            return self.is_valid()
        if action.kind == "feature":
            return not (self.stack and self.stack[-1].is_delta)
        if action.kind == "constant":
            return len(self.stack) == 0 or self.stack[-1].is_featured
        if action.kind == "delta":
            return len(self.stack) > 0 and self.stack[-1].is_featured
        if action.kind == "operator":
            op = action.value
            assert isinstance(op, OperatorSpec)
            if len(self.stack) < op.arity:
                return False
            args = self.stack[-op.arity :]
            try:
                call_expr(op, args)
            except ExpressionError:
                return False
            return True
        return False

    def apply(self, action: Action) -> None:
        if not self.is_action_valid(action):
            raise ExpressionError(f"Invalid action {action.label()} for stack {self.stack}")
        if action.kind == "feature":
            self.stack.append(feature_expr(str(action.value)))
        elif action.kind == "constant":
            self.stack.append(constant_expr(float(action.value)))
        elif action.kind == "delta":
            self.stack.append(delta_expr(int(action.value)))
        elif action.kind == "operator":
            op = action.value
            assert isinstance(op, OperatorSpec)
            args = tuple(self.stack[-op.arity :])
            del self.stack[-op.arity :]
            self.stack.append(call_expr(op, args))
        elif action.kind == "stop":
            return
        else:
            raise ExpressionError(f"Unknown action: {action}")


_TOKEN_RE = re.compile(
    r"\s*([A-Za-z_]\w*|\$|[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[,()])"
)
_NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


class ExpressionParser:
    def __init__(self, expr: str) -> None:
        self.expr = expr
        self.tokens = self._tokenize(expr)
        self.pos = 0

    @staticmethod
    def _tokenize(expr: str) -> list[str]:
        tokens: list[str] = []
        pos = 0
        while pos < len(expr):
            match = _TOKEN_RE.match(expr, pos)
            if match is None:
                raise ExpressionError(f"Cannot tokenize expression near {expr[pos:pos + 24]!r}")
            tokens.append(match.group(1))
            pos = match.end()
        return tokens

    def parse(self) -> Expr:
        node = self._parse_one()
        if self.pos != len(self.tokens):
            raise ExpressionError(f"Unexpected trailing tokens: {' '.join(self.tokens[self.pos:])}")
        return node

    def _parse_one(self) -> Expr:
        token = self._pop()
        if token == "$":
            return feature_expr(self._pop())
        if _NUMBER_RE.fullmatch(token):
            value = float(token)
            if self._peek() and self._peek().lower() == "d":
                self._pop()
                return delta_expr(int(value))
            return constant_expr(value)
        if self._peek() != "(":
            if token.lower() in FEATURES:
                return feature_expr(token)
            raise ExpressionError(f"Unknown token: {token}")
        self._pop()
        if token.lower() == "constant":
            value = self._parse_one()
            self._expect(")")
            if value.kind != "constant":
                raise ExpressionError("Constant(...) expects a number")
            return value
        op = OPERATOR_BY_NAME.get(token.lower())
        if op is None:
            raise ExpressionError(f"Unknown operator: {token}")
        args: list[Expr] = []
        if self._peek() != ")":
            while True:
                args.append(self._parse_one())
                if self._peek() == ",":
                    self._pop()
                    continue
                break
        self._expect(")")
        if op.kind in {"rolling", "pair_rolling"} and args and args[-1].kind == "constant":
            args[-1] = delta_expr(int(float(args[-1].value)))
        return call_expr(op, args)

    def _peek(self) -> str | None:
        return None if self.pos >= len(self.tokens) else self.tokens[self.pos]

    def _pop(self) -> str:
        token = self._peek()
        if token is None:
            raise ExpressionError("Unexpected end of expression")
        self.pos += 1
        return token

    def _expect(self, token: str) -> None:
        actual = self._pop()
        if actual != token:
            raise ExpressionError(f"Expected {token!r}, got {actual!r}")


def parse_expression(expr: str) -> Expr:
    return ExpressionParser(expr).parse()


def parse_many(exprs: Iterable[str]) -> list[Expr]:
    return [parse_expression(expr) for expr in exprs]
