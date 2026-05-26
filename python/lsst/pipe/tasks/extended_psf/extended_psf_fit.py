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

__all__ = (
    "ExtendedPsfFit",
    "ExtendedPsfMoffatFit",
)

from pydantic import BaseModel


class ExtendedPsfFit(BaseModel):
    """Base class for ExtendedPsfImage fit information.

    Attributes
    ----------
    chi2 : `float`
        The chi-squared value of the fit.
    dof : `int`, optional
        Number of degrees of freedom in the fit.
    reduced_chi2 : `float`, optional
        The reduced chi-squared value of the fit.
    """

    chi2: float
    dof: int | None = None
    reduced_chi2: float | None = None

    def __str__(self) -> str:
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"ExtendedPsfFit({attrs})"

    __repr__ = __str__


class ExtendedPsfMoffatFit(ExtendedPsfFit):
    """Moffat-model fit information for an `ExtendedPsfImage`.

    Attributes
    ----------
    amplitude : `float`
        Best-fit Moffat amplitude.
    x_0 : `float`
        Best-fit x-center.
    y_0 : `float`
        Best-fit y-center.
    gamma : `float`
        Best-fit Moffat gamma parameter.
    alpha : `float`
        Best-fit Moffat alpha parameter.
    """

    amplitude: float
    x_0: float
    y_0: float
    gamma: float
    alpha: float

    def __str__(self) -> str:
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"ExtendedPsfMoffatFit({attrs})"

    __repr__ = __str__
