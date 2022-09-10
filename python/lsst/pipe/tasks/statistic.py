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

__all__ = ['Statistic', 'Count', 'Median', 'Percentile', 'StandardDeviation', 'SigmaIQR', 'SigmaMAD',
           'Statistics']

from abc import ABCMeta, abstractmethod
from astropy.stats import mad_std
from dataclasses import dataclass
import numpy as np
from scipy.stats import iqr


class Statistic(metaclass=ABCMeta):
    """Compute a statistic from a list of values.
    """
    # TODO: Make this a property after upgrade to Python 3.9
    @classmethod
    @abstractmethod
    def name(cls):
        pass

    @abstractmethod
    def value(self, values):
        """Return the value of the statistic given a set of values.

        Parameters
        ----------
        values : `Collection` [`float`]
            A set of values to compute the statistic for.
        Returns
        -------
        statistic : `float`
            The value of the statistic.
        """
        pass


class Count(Statistic):
    @classmethod
    def name(cls):
        return "count"

    """The median of a set of values."""
    def value(self, values):
        return len(values)


class Median(Statistic):
    @classmethod
    def name(cls):
        return "median"

    """The median of a set of values."""
    def value(self, values):
        return np.median(values)


@dataclass(frozen=True)
class Percentile(Statistic):
    """An arbitrary percentile.

    Parameters
    ----------
    percentile : `float`
        A valid percentile (0 <= p <= 100).
    """
    percentile: float

    @classmethod
    def name(cls):
        return "percentile"

    def value(self, values):
        return np.percentile(values, self.percentile)


class StandardDeviation(Statistic):
    """The standard deviation (sigma)."""
    @classmethod
    def name(cls):
        return "std"

    def value(self, values):
        return np.std(values)


class SigmaIQR(Statistic):
    """The re-scaled inter-quartile range (sigma equivalent)."""
    @classmethod
    def name(cls):
        return "sigma_iqr"

    def value(self, values):
        return iqr(values, scale='normal')


class SigmaMAD(Statistic):
    """The re-scaled median absolute deviation (sigma equivalent)."""
    @classmethod
    def name(cls):
        return "sigma_mad"

    def value(self, values):
        return mad_std(values)


Statistics = {
    stat.name(): stat
    for stat in (Count, Median, Percentile, StandardDeviation, SigmaIQR, SigmaMAD)
}
