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

import unittest
import lsst.utils.tests

from lsst.pipe.tasks.postprocess import ExtendednessColumnAction

from astropy.table import Table
import numpy as np


class TableAddColumnsActionTestCase(lsst.utils.tests.TestCase):
    """Demo test case."""

    def setUp(self):
        self.bands = ("g", "r", "i")
        action = ExtendednessColumnAction(bands=self.bands)
        self.action = action
        data = {
            f"{action.reff_model}_reff_x": [1., 3., 0.01, 0.4, 0.02],
            f"{action.reff_model}_reff_y": [0.2, 6, 0.01, 0.2, 0.01],
            f"{action.reff_model}_reff_rho": [0.5, 0.2, 0., 0.1, -0.02],
            action.lnlike_ps_diff_column: [0.8, 50, 0., -0.05, 0.1],
        }
        for column, kwargs in (
            (action.model_column_flux, {"model": action.reff_model}),
            (action.sn_column_flux, {}),
            (action.model_column_flux_err, {"model": action.reff_model}),
            (action.sn_column_flux_err, {}),
        ):
            for idx, band in enumerate(action.bands):
                data[column.format(band=band, **kwargs)] = np.sqrt(idx + 1.)*np.array(
                    [1.5e3, 2.5e3, 6.8e3, 3.4e3, 5.5e3])
        self.data = Table(data)


    def testExtendednessColumnAction(self):
        result = tuple(self.action(self.data))
        assert len(result) == 1
        assert result[0] == self.action.output_column
        values = self.data[result[0]]
        assert len(values) == len(self.data[self.action.lnlike_ps_diff_column])
        assert all((values >= 0) & (values <= 1))


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
