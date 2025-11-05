# This file is part of {{ cookiecutter.package_name }}.

# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np
import pandas as pd
import unittest

import lsst.geom as geom
from lsst.pipe.tasks.coaddBase import makeSkyInfo
from lsst.pipe.tasks.drpAssociationPipe import DrpAssociationPipeTask
import lsst.skymap as skyMap
import lsst.utils.tests


class TestDrpAssociationPipe(lsst.utils.tests.TestCase):

    def setUp(self):
        simpleMapConfig = skyMap.discreteSkyMap.DiscreteSkyMapConfig()
        simpleMapConfig.raList = [10, 11]
        simpleMapConfig.decList = [-1, -1]
        simpleMapConfig.radiusList = [0.1, 0.1]

        self.simpleMap = skyMap.DiscreteSkyMap(simpleMapConfig)
        self.tractId = 0
        self.patchId = 10

        self.skyInfo = makeSkyInfo(self.simpleMap, self.tractId, self.patchId)
        self.innerPatchBox = geom.Box2D(self.skyInfo.patchInfo.getInnerBBox())
        self.innerTractSkyRegion = self.skyInfo.tractInfo.getInnerSkyRegion()

        self.nSources = 100
        xs = np.linspace(self.innerPatchBox.getMinX() + 1,
                         self.innerPatchBox.getMaxX() - 1,
                         self.nSources)
        ys = np.linspace(self.innerPatchBox.getMinY() + 1,
                         self.innerPatchBox.getMaxY() - 1,
                         self.nSources)

        dataIn = []
        dataOut = []
        for x, y in zip(xs, ys):
            coordIn = self.skyInfo.wcs.pixelToSky(x, y)
            coordOut = self.skyInfo.wcs.pixelToSky(
                x + 10 * self.innerPatchBox.getWidth(),
                y + 10 * self.innerPatchBox.getHeight())
            dataIn.append({"ra": coordIn.getRa().asDegrees(),
                           "dec": coordIn.getDec().asDegrees()})
            dataOut.append({"ra": coordOut.getRa().asDegrees(),
                            "dec": coordOut.getDec().asDegrees()})

        self.diaSrcCatIn = pd.DataFrame(data=dataIn)
        self.diaSrcCatOut = pd.DataFrame(data=dataOut)

    def tearDown(self):
        pass

    def testTrimToPatch(self):
        """Test that points inside and outside the patch are correctly
        identified as such.
        """
        dpaTask = DrpAssociationPipeTask()

        self.assertEqual(
            np.sum(dpaTask._trimToPatch(self.diaSrcCatIn,
                                        self.innerPatchBox,
                                        self.skyInfo.wcs,
                                        innerTractSkyRegion=self.innerTractSkyRegion)),
            self.nSources)

        self.assertEqual(
            np.sum(dpaTask._trimToPatch(self.diaSrcCatOut,
                                        self.innerPatchBox,
                                        self.skyInfo.wcs,
                                        innerTractSkyRegion=self.innerTractSkyRegion)),
            0)


def setup_module(module):
    lsst.utils.tests.init()


class MatchMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
