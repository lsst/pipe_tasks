# This file is part of pipe_tasks.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

__all__ = ("makeColumnExpressionAction", )

import ast
import operator as op

from typing import Mapping, MutableMapping, Set, Type, Union, Optional, Any, Iterable

from numpy import log10 as log
from numpy import (cos, sin, cosh, sinh)
import pandas as pd

from lsst.pex.config.configurableActions import ConfigurableActionField
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


def makeColumnExpressionAction(className: str, expr: str,
                               exprDefaults: Optional[Mapping[str, Union[DataFrameAction,
                                                                         Type[DataFrameAction]]]] = None,
                               docstring: str = None
                               ) -> Type[DataFrameAction]:
    """Factory function for producing ConfigurableAction classes which are
    realizations of arithmetic operations.

    Parameters
    ----------
    className : `str`
        The name of the class that will be produced
    expr : `str`
        An arithmetic expression that will be parsed to produce the output
        ConfigurableAction. Individual variable names will be the name of
        individual `ConfigActions` inside the expression (i.e. "x+y" will
        produce an action with configAction.actions.x and
        configAction.actions.y). Expression can contain arithmatic python
        operators as well as; sin, cos, sinh, cosh, log (which is base 10).
    exprDefaults : `Mapping` of `str` to `DataFrameAction` optional
        A mapping of strings which correspond to the names in the expression to
        values which are default `ConfigurableActions` to assign in the
        expression. If no default for a action is supplied `SingleColumnAction`
        is set as the default.
    docstring : `str`
        A string that is assigned as the resulting classes docstring

    Returns
    -------
    action : `Type` of `DataFrameAction`
        A `DataFrameAction` class that was programatically constructed from the
        input expression.
    """
    # inspect is used because this is a factory function used to produce classes
    # and it is desireable that the classes generated appear to be in the
    # module of the calling frame, instead of something defined within the
    # scope of this function call.
    import inspect
    new_module = inspect.stack()[1].frame.f_locals['__name__']
    node = ast.parse(expr, mode='eval')

    # gather the specified names
    names: Set[str] = set()
    for elm in ast.walk(node):
        if isinstance(elm, ast.Name):
            names.add(elm.id)

    # remove the known Math names
    names -= EXTRA_MATH.keys()

    fields: Mapping[str, ConfigurableActionField] = {}
    for name in sorted(names):
        if exprDefaults is not None and (value := exprDefaults.get(name)) is not None:
            kwargs = {"default": value}
        else:
            kwargs = {}
        fields[name] = ConfigurableActionField(doc=f"expression action {name}", **kwargs)

    # skip flake8 on N807 because this is a stand alone function, but it is
    # intended to be patched in as a method on a dynamically generated class
    def __call__(self, df: pd.DataFrame, **kwargs) -> pd.Series:  # noqa: N807
        values_map = {}
        for name in fields:
            values_map[name] = getattr(self, name)(df, **kwargs)

        parser = ExpressionParser(**values_map)
        return parser.visit(node.body)

    # create the function to look up the columns for the dynamically created action
    def columns(self) -> Iterable[str]:
        for name in fields:
            yield from getattr(self, name).columns

    dct: MutableMapping[str, Any] = {"__call__": __call__, "columns": property(columns)}
    if docstring is not None:
        dct['__doc__'] = docstring
    dct.update(**fields)
    dct['__module__'] = new_module

    return type(className, (DataFrameAction, ), dct)
