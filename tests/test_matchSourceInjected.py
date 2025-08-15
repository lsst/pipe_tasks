#
# This file is part of ap_pipe.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import numpy as np
import pandas as pd
import unittest

from astropy.table import Table
from lsst.afw.table import SourceCatalog, SourceTable
from lsst.afw.image import ExposureF
from lsst.afw.geom import makeSkyWcs
from lsst.geom import SpherePoint, degrees, Point2D, Extent2I
import lsst.utils.tests

from lsst.pipe.tasks.matchDiffimSourceInjected import (
    MatchInjectedToDiaSourceTask,
    MatchInjectedToAssocDiaSourceTask,
    MatchInjectedToDiaSourceConfig,
    MatchInjectedToAssocDiaSourceConfig,
)


class BaseTestMatchInjected(lsst.utils.tests.TestCase):
    def setUp(self):
        rng = np.random.RandomState(6)
        # 0.35 arcsec = 0.5/np.sqrt(2) arcsec (half of the diagonal of a 0.5 arcsec square)
        offsetFactor = 1./3600 * 0.35

        # Create a mock injected catalog
        self.injectedCat = Table(
            {
                "injection_id": [1, 2, 3, 5, 6, 7, 12, 21, 31, 49],
                "injection_flag": np.repeat(0, 10),
                # random positions with 10 arcmin size
                # ra centered at 0, dec centered at -30
                "ra": (1/6.) * rng.random(size=10),
                "dec": -30 + (1/6.) * rng.random(size=10),
                "mag": 20.25 - 0.5 * rng.random(size=10),  # random magnitudes
                "source_type": np.repeat("DeltaFunction", 10)
            }
        )

        # Create a mock diaSources catalog
        schema = SourceTable.makeMinimalSchema()
        self.diaSources = SourceCatalog(schema)
        for i in range(5):
            record = self.diaSources.addNew()
            record.setId(100 + i)
            record.setCoord(
                # Random posisions of diaSources
                SpherePoint((1/6.) * rng.random(), -30 + (1/6.) * rng.random(), degrees)
            )

        for i, (ra, dec) in enumerate(self.injectedCat[['ra', 'dec']][:8]):
            record = self.diaSources.addNew()
            sign = rng.choice([-1, 1], size=2)
            record.setId(i + 200)
            record.setCoord(
                SpherePoint(
                    ra+sign[0]*rng.random()*offsetFactor,
                    dec+sign[1]*rng.random()*offsetFactor,
                    degrees
                )
            )

        # Create a mock difference image
        self.diffIm = ExposureF(Extent2I(4096, 4096))
        crpix = Point2D(0, 0)
        crval = SpherePoint(0, -30, degrees)
        cdMatrix = np.array([[5.19513851e-05, 2.81124812e-07],
                            [3.25186974e-07, 5.19112119e-05]])
        wcs = makeSkyWcs(crpix, crval, cdMatrix)
        self.diffIm.setWcs(wcs)

        # add a fake source outside of the image
        self.injectedCatForTrimming = self.injectedCat.copy()
        self.injectedCatForTrimming.add_row(
            {
                'injection_id': 50,
                'injection_flag': 0,
                'ra': 360. - 0.1/6.,
                'dec': -30 - 0.1/6.,
                'mag': 20.0,
                'source_type': 'DeltaFunction'
            }
        )

        # only 4 injected sources are associated
        self.assocDiaSources = pd.DataFrame(
            {
                "diaSourceId": [101, 102, 103, 201, 202, 205, 207],
                "band": np.repeat("r", 7),
                "visit": np.repeat(410001, 7),
                "detector": np.repeat(0, 7),
                "diaObjectId": np.arange(7),
            }
        )


class TestMatchInjectedToDiaSourceTask(BaseTestMatchInjected):

    def test_run(self):
        config = MatchInjectedToDiaSourceConfig()
        config.matchDistanceArcseconds = 0.5
        config.doMatchVisit = False
        config.doForcedMeasurement = False

        task = MatchInjectedToDiaSourceTask(config=config)

        result = task.run(self.injectedCat, self.diffIm, self.diaSources)
        self.assertEqual(len(result.matchDiaSources), len(self.injectedCat))
        self.assertEqual(np.sum(result.matchDiaSources['diaSourceId'] > 0), 8)
        self.assertEqual(np.sum(result.matchDiaSources['dist_diaSrc'] > 0), 8)
        self.assertEqual(
            np.sum(np.abs(result.matchDiaSources['dist_diaSrc']) < config.matchDistanceArcseconds), 8
        )

    def test_run_trimming(self):
        config = MatchInjectedToDiaSourceConfig()
        config.matchDistanceArcseconds = 0.5
        config.doMatchVisit = True
        config.doForcedMeasurement = False

        task = MatchInjectedToDiaSourceTask(config=config)
        result = task.run(self.injectedCatForTrimming, self.diffIm, self.diaSources)

        self.assertEqual(len(result.matchDiaSources), len(self.injectedCatForTrimming) - 1)
        self.assertEqual(np.sum(result.matchDiaSources['diaSourceId'] > 0), 8)
        self.assertEqual(np.sum(result.matchDiaSources['dist_diaSrc'] > 0), 8)
        self.assertEqual(
            np.sum(np.abs(result.matchDiaSources['dist_diaSrc']) < config.matchDistanceArcseconds), 8
        )

    def test_getVectors(self):
        config = MatchInjectedToDiaSourceConfig()
        config.matchDistanceArcseconds = 0.5
        config.doMatchVisit = False
        config.doForcedMeasurement = False

        task = MatchInjectedToDiaSourceTask(config=config)

        ras = np.radians(self.injectedCat['ra'])
        decs = np.radians(self.injectedCat['dec'])
        vectors = task._getVectors(ras, decs)
        self.assertEqual(vectors.shape, (10, 3))


class TestMatchInjectedToAssocDiaSourceTask(BaseTestMatchInjected):

    def test_run(self):
        config = MatchInjectedToDiaSourceConfig()
        config.matchDistanceArcseconds = 0.5
        config.doMatchVisit = False
        config.doForcedMeasurement = False

        task = MatchInjectedToDiaSourceTask(config=config)

        result = task.run(self.injectedCat, self.diffIm, self.diaSources)

        configAssoc = MatchInjectedToAssocDiaSourceConfig()
        taskAssoc = MatchInjectedToAssocDiaSourceTask(config=configAssoc)
        resultAssoc = taskAssoc.run(self.assocDiaSources, result.matchDiaSources)
        self.assertEqual(len(resultAssoc.matchAssocDiaSources), len(self.injectedCat))
        self.assertEqual(np.sum(resultAssoc.matchAssocDiaSources['isAssocDiaSource']), 4)


if __name__ == "__main__":
    unittest.main()
