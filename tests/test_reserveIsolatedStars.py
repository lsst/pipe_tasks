#
# LSST Data Management System
# Copyright 2008-2022 AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
"""Test ReserveIsolatedStarsTask.
"""
import unittest
import numpy as np

import lsst.utils.tests

from lsst.pipe.tasks.reserveIsolatedStars import (ReserveIsolatedStarsConfig,
                                                  ReserveIsolatedStarsTask)


class ReserveIsolatedStarsTestCase(lsst.utils.tests.TestCase):
    """Test ReserveIsolatedStarsTask."""
    def test_reserve(self):
        """Test running the reserve isolated stars task."""
        config = ReserveIsolatedStarsConfig()
        reserve_task = ReserveIsolatedStarsTask(config=config)

        # Check we get the correct number of reserved stars.
        nstar = 1000
        reserved = reserve_task.run(nstar, extra='r_100')
        self.assertEqual(reserved.sum(),
                         int(config.reserve_fraction*nstar))

        # Confirm we get the same list with the same run.
        reserved2 = reserve_task.run(nstar, extra='r_100')
        np.testing.assert_array_equal(reserved2, reserved)

        # Confirm we get a different list with a different run.
        reserved3 = reserve_task.run(nstar, extra='r_101')
        self.assertFalse(np.all(reserved3 == reserved))

    def test_reserve_none(self):
        """Test running the reserve isolated stars task with no reserved stars."""
        config = ReserveIsolatedStarsConfig()
        config.reserve_fraction = 0.0
        reserve_task = ReserveIsolatedStarsTask(config=config)

        # Check we get the correct number of reserved stars.
        nstar = 1000
        reserved = reserve_task.run(nstar)
        self.assertEqual(reserved.sum(), 0)


class MyMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
