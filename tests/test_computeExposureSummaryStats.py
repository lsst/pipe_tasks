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

"""Test ComputeExposureSummaryStatsTask.
"""
import unittest

import numpy as np

import lsst.utils.tests
from lsst.afw.detection import GaussianPsf
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
from lsst.daf.base import DateTime
from lsst.afw.coord import Observatory
from lsst.afw.geom import makeCdMatrix, makeSkyWcs
from lsst.pipe.tasks.computeExposureSummaryStats import ComputeExposureSummaryStatsTask


class ComputeExposureSummaryTestCase(lsst.utils.tests.TestCase):

    def testComputeExposureSummary(self):
        """Make a fake exposure and background and compute summary.
        """
        np.random.seed(12345)

        # Make an exposure with a noise image
        exposure = afwImage.ExposureF(100, 100)
        skySigma = 10.0
        exposure.getImage().getArray()[:, :] = np.random.normal(0.0, skySigma, size=(100, 100))
        exposure.getVariance().getArray()[:, :] = skySigma**2.

        # Set the visitInfo
        date = DateTime(date=59234.7083333334, system=DateTime.DateSystem.MJD)
        observatory = Observatory(-70.7366*lsst.geom.degrees, -30.2407*lsst.geom.degrees,
                                  2650.)
        visitInfo = afwImage.VisitInfo(exposureTime=10.0,
                                       date=date,
                                       observatory=observatory)
        exposure.getInfo().setVisitInfo(visitInfo)

        # Install a Gaussian PSF
        psfSize = 2.0
        psf = GaussianPsf(5, 5, psfSize)
        exposure.setPsf(psf)

        # Install a simple WCS
        scale = 0.2*lsst.geom.arcseconds
        raCenter = 300.0*lsst.geom.degrees
        decCenter = 0.0*lsst.geom.degrees
        cdMatrix = makeCdMatrix(scale=scale)
        skyWcs = makeSkyWcs(crpix=exposure.getBBox().getCenter(),
                            crval=lsst.geom.SpherePoint(raCenter, decCenter),
                            cdMatrix=cdMatrix)
        exposure.setWcs(skyWcs)

        # Install a simple photoCalib
        photoCalib = afwImage.PhotoCalib(calibrationMean=0.3)
        zp = 2.5*np.log10(photoCalib.getInstFluxAtZeroMagnitude())
        exposure.setPhotoCalib(photoCalib)

        # Compute the background image
        bgGridSize = 10
        bctrl = afwMath.BackgroundControl(afwMath.Interpolate.NATURAL_SPLINE)
        bctrl.setNxSample(int(exposure.getMaskedImage().getWidth()/bgGridSize) + 1)
        bctrl.setNySample(int(exposure.getMaskedImage().getHeight()/bgGridSize) + 1)
        backobj = afwMath.makeBackground(exposure.getMaskedImage().getImage(), bctrl)
        background = afwMath.BackgroundList()
        background.append(backobj)

        # Run the task
        expSummaryTask = ComputeExposureSummaryStatsTask()
        summary = expSummaryTask.run(exposure, None, background)

        # Test the outputs
        self.assertFloatsAlmostEqual(summary.psfSigma, psfSize)
        self.assertFloatsAlmostEqual(summary.psfIxx, psfSize**2.)
        self.assertFloatsAlmostEqual(summary.psfIyy, psfSize**2.)
        self.assertFloatsAlmostEqual(summary.psfIxy, 0.0)
        self.assertFloatsAlmostEqual(summary.psfArea, 23.088975164455444)

        delta = (scale*50).asDegrees()
        for a, b in zip(summary.raCorners,
                        [raCenter.asDegrees() + delta, raCenter.asDegrees() - delta,
                         raCenter.asDegrees() - delta, raCenter.asDegrees() + delta]):
            self.assertFloatsAlmostEqual(a, b, atol=1e-10)
        for a, b in zip(summary.decCorners,
                        [decCenter.asDegrees() - delta, decCenter.asDegrees() - delta,
                         decCenter.asDegrees() + delta, decCenter.asDegrees() + delta]):
            self.assertFloatsAlmostEqual(a, b, atol=1e-10)

        self.assertFloatsAlmostEqual(summary.ra, raCenter.asDegrees(), atol=1e-10)
        self.assertFloatsAlmostEqual(summary.dec, decCenter.asDegrees(), atol=1e-10)

        self.assertFloatsAlmostEqual(summary.zeroPoint, zp)

        # Need to compare background level and noise
        # These are only approximately 0+/-10 because of the small image
        self.assertFloatsAlmostEqual(summary.skyBg, -0.079, atol=1e-3)

        self.assertFloatsAlmostEqual(summary.meanVar, skySigma**2.)

        self.assertFloatsAlmostEqual(summary.zenithDistance, 30.57112, atol=1e-5)


class MyMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
