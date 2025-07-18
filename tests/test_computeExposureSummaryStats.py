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
from lsst.daf.base import DateTime, PropertyList
from lsst.afw.coord import Observatory
from lsst.afw.geom import makeCdMatrix, makeSkyWcs
from lsst.afw.cameraGeom.testUtils import DetectorWrapper
from lsst.pipe.tasks.computeExposureSummaryStats import ComputeExposureSummaryStatsTask
from lsst.pipe.tasks.computeExposureSummaryStats import ComputeExposureSummaryStatsConfig
from lsst.pipe.tasks.computeExposureSummaryStats import compute_magnitude_limit


class ComputeExposureSummaryTestCase(lsst.utils.tests.TestCase):

    def testComputeExposureSummary(self):
        """Make a fake exposure and background and compute summary.
        """
        np.random.seed(12345)

        # Make an exposure with a noise image
        exposure = afwImage.ExposureF(100, 100)

        band = "i"
        physical_filter = "test-i"
        exposure.setFilter(afwImage.FilterLabel(band=band, physical=physical_filter))

        readNoise = 5.0
        gain = 1.0
        detector = DetectorWrapper(numAmps=1).detector
        metadata = PropertyList()
        metadata.add("LSST ISR UNITS", "electron")
        for amp in detector.getAmplifiers():
            metadata.add(f"LSST ISR READNOISE {amp.getName()}", readNoise)
            metadata.add(f"LSST ISR GAIN {amp.getName()}", gain)
        exposure.setDetector(detector)
        exposure.setMetadata(metadata)

        skyMean = 100.0
        skySigma = 10.0
        exposure.getImage().getArray()[:, :] = np.random.normal(skyMean, skySigma, size=(100, 100))
        exposure.getVariance().getArray()[:, :] = skySigma**2.

        # Set the visitInfo
        date = DateTime(date=59234.7083333334, system=DateTime.DateSystem.MJD)
        observatory = Observatory(-70.7366*lsst.geom.degrees, -30.2407*lsst.geom.degrees,
                                  2650.)
        expTime = 10.0
        visitInfo = afwImage.VisitInfo(exposureTime=expTime,
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
        pixelScale = skyWcs.getPixelScale().asArcseconds()
        exposure.setWcs(skyWcs)

        # Install a simple photoCalib
        photoCalib = afwImage.PhotoCalib(calibrationMean=0.3)
        zp = 2.5*np.log10(photoCalib.getInstFluxAtZeroMagnitude())
        exposure.setPhotoCalib(photoCalib)

        # Install a simple apCorrMap
        apCorrMap = afwImage.ApCorrMap()
        apCorrMap.set(
            "base_PsfFlux_instFlux", afwMath.ChebyshevBoundedField(exposure.getBBox(), np.zeros((3, 3)))
        )
        exposure.setApCorrMap(apCorrMap)

        # Compute the background image
        bgGridSize = 10
        bctrl = afwMath.BackgroundControl(afwMath.Interpolate.NATURAL_SPLINE)
        bctrl.setNxSample(int(exposure.getMaskedImage().getWidth()/bgGridSize) + 1)
        bctrl.setNySample(int(exposure.getMaskedImage().getHeight()/bgGridSize) + 1)
        backobj = afwMath.makeBackground(exposure.getMaskedImage().getImage(), bctrl)
        background = afwMath.BackgroundList()
        background.append(backobj)

        # Configure and run the task
        expSummaryTask = ComputeExposureSummaryStatsTask()
        # Configure nominal values for effective time calculation (normalized to 1s exposure)
        expSummaryTask.config.fiducialZeroPoint = {band: float(zp - 2.5*np.log10(expTime))}
        expSummaryTask.config.fiducialPsfSigma = {band: float(psfSize)}
        expSummaryTask.config.fiducialSkyBackground = {band: float(skyMean/expTime)}
        # Run the task
        summary = expSummaryTask.run(exposure, None, background)

        # Test the outputs
        self.assertFloatsAlmostEqual(summary.psfSigma, psfSize)
        self.assertFloatsAlmostEqual(summary.psfIxx, psfSize**2.)
        self.assertFloatsAlmostEqual(summary.psfIyy, psfSize**2.)
        self.assertFloatsAlmostEqual(summary.psfIxy, 0.0)
        self.assertFloatsAlmostEqual(summary.psfArea, 23.088975164455444)

        self.assertFloatsAlmostEqual(summary.psfTraceRadiusDelta, 0.0)
        self.assertFloatsAlmostEqual(summary.psfApFluxDelta, 0.0)
        self.assertFloatsAlmostEqual(summary.psfApCorrSigmaScaledDelta, 0.0)

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

        self.assertFloatsAlmostEqual(summary.pixelScale, pixelScale)

        self.assertFloatsAlmostEqual(summary.zeroPoint, zp)
        self.assertFloatsAlmostEqual(summary.expTime, expTime)

        # Need to compare background level and noise
        # These are only approximately 0+/-10 because of the small image
        self.assertFloatsAlmostEqual(summary.skyBg, skyMean, rtol=1e-3)
        self.assertFloatsAlmostEqual(summary.meanVar, skySigma**2.)

        self.assertFloatsAlmostEqual(summary.zenithDistance, 30.57112, atol=1e-5)

        # Effective exposure time and depth
        self.assertFloatsAlmostEqual(summary.effTime, expTime, rtol=1e-3)
        self.assertFloatsAlmostEqual(summary.magLim, 26.584, rtol=1e-3)

    def testComputeMagnitudeLimit(self):
        """Test the magnitude limit calculation using fiducials from SMTN-002
        and syseng_throughputs."""

        # Values from syseng_throughputs notebook assuming 30s exposure
        # consisting of 2x15s snaps each with readnoise of 9e-
        fwhm_eff_fid = {'g': 0.87, 'r': 0.83, 'i': 0.80}
        skycounts_fid = {'g': 463.634122, 'r': 988.626863, 'i': 1588.280513}
        zeropoint_fid = {'g': 28.508375, 'r': 28.360838, 'i': 28.171396}
        readnoise_fid = {'g': 12.73, 'r': 12.73, 'i': 12.73}
        # Assumed values from SMTN-002
        snr = 5
        gain = 1.0
        exposure_time = 30
        pixel_scale = 0.2
        # Output magnitude limit from syseng_throughputs notebook
        m5_fid = {'g': 24.90, 'r': 24.48, 'i': 24.10}

        for band in ['g', 'r', 'i']:
            # Translate into DM quantities
            psfArea = 2.266 * (fwhm_eff_fid[band] / pixel_scale)**2
            skyBg = skycounts_fid[band]
            zeroPoint = zeropoint_fid[band] + 2.5*np.log10(exposure_time)
            readNoise = readnoise_fid[band]

            # Calculate the M5 values
            m5 = compute_magnitude_limit(psfArea, skyBg, zeroPoint, readNoise, gain, snr)
            self.assertFloatsAlmostEqual(m5, m5_fid[band], atol=1e-2)

        # Check that input NaN lead to output NaN
        nan = float('nan')
        m5 = compute_magnitude_limit(nan, skyBg, zeroPoint, readNoise, gain, snr)
        self.assertFloatsAlmostEqual(m5, nan, ignoreNaNs=True)
        m5 = compute_magnitude_limit(psfArea, nan, zeroPoint, readNoise, gain, snr)
        self.assertFloatsAlmostEqual(m5, nan, ignoreNaNs=True)
        m5 = compute_magnitude_limit(psfArea, skyBg, nan, readNoise, gain, snr)
        self.assertFloatsAlmostEqual(m5, nan, ignoreNaNs=True)
        m5 = compute_magnitude_limit(psfArea, skyBg, zeroPoint, nan, gain, snr)
        self.assertFloatsAlmostEqual(m5, nan, ignoreNaNs=True)

    def testComputeFiducialMagnitudeLimit(self):
        """Test the magnitude limit calculation using fiducials from SMTN-002
        and syseng_throughputs."""

        # Setup the config
        config = ComputeExposureSummaryStatsConfig()
        # Configure fiducial values.
        # These come from obs_lsst/config/lsstCam
        config.fiducialZeroPoint = {
            "u": 26.52, "g": 28.51, "r": 28.36, "i": 28.17, "z": 27.78, "y": 26.82,
        }
        config.fiducialPsfSigma = {
            "u": 1.72, "g": 1.63, "r": 1.56, "i": 1.51, "z": 1.47, "y": 1.44,
        }
        config.fiducialSkyBackground = {
            "u": 1.51, "g": 15.45, "r": 32.95, "i": 52.94, "z": 79.40, "y": 90.57,
        }
        config.fiducialReadNoise = {
            "u": 9.0, "g": 9.0, "r": 9.0, "i": 9.0, "z": 9.0, "y": 9.0,
        }
        # Other properties
        exposureTime = 30  # second
        pixelScale = 0.2  # arcsec/pix
        gain = 1.0  # [e-/ADU]

        # Fiducial m5 from SMTN-002
        fiducialMagLim = {
            "u": 23.70, "g": 24.97, "r": 24.52, "i": 24.13, "z": 23.56, "y": 22.55
        }

        # Compare fiducial m5 calculated from config to SMTN-002 values
        for band in fiducialMagLim.keys():
            print(f"\n{band}")
            m5fid = config.fiducialMagnitudeLimit(band, exposureTime, pixelScale, gain)
            print(f"{fiducialMagLim[band]:.2f} {m5fid:.3f}")
            self.assertFloatsAlmostEqual(m5fid, fiducialMagLim[band], atol=1e-2)

        # Check that passing an unknown band outputs NaN
        nan = float('nan')
        m5fid = config.fiducialMagnitudeLimit("block", exposureTime, pixelScale, gain)
        self.assertFloatsAlmostEqual(m5fid, nan, ignoreNaNs=True)


class MyMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
