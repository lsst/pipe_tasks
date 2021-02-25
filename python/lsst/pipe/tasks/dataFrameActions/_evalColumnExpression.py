from __future__ import annotations

__all__ = ("makeColumnExpressionAction", )

import ast
import operator as op

from typing import Type

from numpy import log10 as log
from numpy import (cos, sin, cosh, sinh)

from ._actions import SingleColumnAction, MultiColumnAction
from ._baseDataFrameActions import DataFrameAction


OPERATORS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
             ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
             ast.USub: op.neg}

EXTRA_MATH = {"cos": cos, "sin": sin, "cosh": cosh, "sinh": sinh, "log": log}


class ExpressionParser(ast.NodeVisitor):
    def __init__(self, **kwargs):
        self.variables = kwargs
        self.variables['log'] = log

    def visit_Name(self, node):
        if node.id in self.variables:
            return self.variables[node.id]
        else:
            return None

    def visit_Num(self, node):
        return node.n

    def visit_NameConstant(self, node):
        return node.value

    def visit_UnaryOp(self, node):
        val = self.visit(node.operand)
        return OPERATORS[type(node.op)](val)

    def visit_BinOp(self, node):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        return OPERATORS[type(node.op)](lhs, rhs)

    def visit_Call(self, node):
        if node.func.id in self.variables:
            function = self.visit(node.func)
            return function(self.visit(node.args[0]))
        else:
            raise ValueError("String not recognized")

    def generic_visit(self, node):
        raise ValueError("String not recognized")


def makeColumnExpressionAction(expr: str, exprDefaults=None) -> Type[DataFrameAction]:
    node = ast.parse(expr, mode='eval')

    # gather the specified names
    names = set()
    for elm in ast.walk(node):
        if isinstance(elm, ast.Name):
            names.add(elm.id)

    # remove the known Math names
    names -= EXTRA_MATH.keys()

    fields = {name: SingleColumnAction for name in names}

    class ColumnExpressionAction(MultiColumnAction):
        def __call__(self, df, **kwargs):
            values_map = {}
            for name in fields:
                values_map[name] = getattr(self, name)(df, **kwargs)

            parser = ExpressionParser(**values_map)
            return parser.visit(node.body)

        def setDefaults(self):
            super().setDefaults()
            self.actions = fields

            if exprDefaults is not None:
                for name, value in exprDefaults:
                    getattr(self.actions, name).column = value

    return ColumnExpressionAction
