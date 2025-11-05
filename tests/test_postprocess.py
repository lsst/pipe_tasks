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

import pytest
import unittest

import lsst.utils.tests
from lsst.pipe.tasks.postprocess import ModelExtendednessColumnAction

from astropy.table import Table
import numpy as np


class ModelExtendednessColumnActionTestCase(lsst.utils.tests.TestCase):
    """Demo test case."""

    def setUp(self):
        self.bands = ("g", "r", "i")
        action = ModelExtendednessColumnAction(bands=self.bands, min_n_good_to_shift_flux_ratio=1)
        self.action = action
        data = {
            action.size_column.format(axis="x"): [1., 3., 0.01, 0.4, 0.02],
            action.size_column.format(axis="y"): [0.2, 6, 0.01, 0.2, 0.01],
        }
        model = action.model_flux_name
        factors = np.array([1.01, 1.12, 0.995, 0.998, 1.6])
        fluxes = np.array([1.5e3, 2.5e3, 6.8e3, 3.4e3, 5.5e3])
        for column_flux, column_flux_err, factors in (
            (action.model_column_flux, action.model_column_flux_err, factors),
            (action.psf_column_flux, action.psf_column_flux_err, None),
        ):
            for idx, band in enumerate(action.bands):
                flux = np.sqrt(idx + 1.)*fluxes
                if factors is not None:
                    flux *= factors
                data[column_flux.format(band=band, model=model)] = flux
                data[column_flux_err.format(band=band, model=model)] = np.sqrt(flux)
        self.data = Table(data)

    def testExtendednessColumnAction(self):
        action = self.action
        with pytest.raises(ValueError):
            action.validate()
        action.bands_combined = {"gri": "g,r,i"}
        action.validate()
        schema = action.getInputSchema()
        n_values = len(self.data[schema[0][0]])
        assert all([len(self.data[col]) == n_values for col, _ in schema[1:]])

        result = self.action(self.data)
        columns_expected = [
            action.output_column.format(band=band)
            for band in list(action.bands) + list(action.bands_combined.keys())
        ]
        assert list(result.keys()) == columns_expected

        for column, values in result.items():
            assert len(values) == n_values
            assert all((values >= 0) & (values <= 1))


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
