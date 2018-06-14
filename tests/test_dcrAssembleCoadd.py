# This file is part of pipe_tasks.
#
# LSST Data Management System
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
# See COPYRIGHT file at the top of the source tree.
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
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.

from astropy import units as u
import numpy as np
import unittest

from lsst.afw.coord import Observatory, Weather
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
from lsst.geom import arcseconds, degrees, radians
from lsst.meas.algorithms.testUtils import plantSources
import lsst.utils.tests
from lsst.pipe.tasks.dcrAssembleCoadd import DcrAssembleCoaddTask, DcrAssembleCoaddConfig


class DcrAssembleCoaddTestTask(lsst.utils.tests.TestCase):
    """A test case for the DCR-aware image coaddition algorithm.

    Attributes
    ----------
    bbox : `lsst.afw.geom.Box2I`
        Bounding box of the test model.
    bufferSize : `int`
        Distance from the inner edge of the bounding box
        to avoid placing test sources in the model images.
    config : `lsst.pipe.tasks.dcrAssembleCoadd.DcrAssembleCoaddConfig`
        Configuration parameters to initialize the task.
    filterInfo : `lsst.afw.image.Filter`
        Dummy filter object for testing.
    mask : `lsst.afw.image.Mask`
        Reference mask of the unshifted model.
    """

    def setUp(self):
        """Define the filter, DCR parameters, and the bounding box for the tests.
        """
        self.config = DcrAssembleCoaddConfig()
        self.config.dcrNumSubfilters = 3
        lambdaEff = 476.31  # Use LSST g band values for the test.
        lambdaMin = 405
        lambdaMax = 552
        afwImage.utils.defineFilter("gTest", lambdaEff, lambdaMin=lambdaMin, lambdaMax=lambdaMax)
        self.bufferSize = 5
        badMaskPlanes = self.config.badMaskPlanes[:]
        badMaskPlanes.append("CLIPPED")
        xSize = 40
        ySize = 42
        x0 = 12345
        y0 = 67890
        self.bbox = afwGeom.Box2I(afwGeom.Point2I(x0, y0), afwGeom.Extent2I(xSize, ySize))

    def makeTestImages(self):
        """Make reproduceable PSF-convolved masked images for testing.

        Returns
        -------
        dcrModels : `list` of `lsst.afw.image.maskedImage`
            A list of masked images, each containing the model for one subfilter
        """
        seed = 5
        rng = np.random.RandomState(seed)
        psfSize = 2
        nSrc = 5
        noiseLevel = 5
        detectionSigma = 5.
        sourceSigma = 20.
        fluxRange = 2.
        x0, y0 = self.bbox.getBegin()
        xSize, ySize = self.bbox.getDimensions()
        xLoc = rng.rand(nSrc)*(xSize - 2*self.bufferSize) + self.bufferSize + x0
        yLoc = rng.rand(nSrc)*(ySize - 2*self.bufferSize) + self.bufferSize + y0
        dcrModels = []

        imageSum = np.zeros((ySize, xSize))
        for subfilter in range(self.config.dcrNumSubfilters):
            flux = (rng.rand(nSrc)*(fluxRange - 1.) + 1.)*sourceSigma*noiseLevel
            sigmas = [psfSize for src in range(nSrc)]
            coordList = list(zip(xLoc, yLoc, flux, sigmas))
            model = plantSources(self.bbox, 10, 0, coordList, addPoissonNoise=False)
            model.image.array += rng.rand(ySize, xSize)*noiseLevel
            imageSum += model.image.array
            model.mask.addMaskPlane("CLIPPED")
            dcrModels.append(model.maskedImage)
        maskVals = np.zeros_like(imageSum)
        maskVals[imageSum > detectionSigma*noiseLevel] = afwImage.Mask.getPlaneBitMask('DETECTED')
        for model in dcrModels:
            model.mask.array[:] = maskVals
        self.mask = dcrModels[0].mask
        return dcrModels

    def makeDummyWcs(self, rotAngle, pixelScale, crval):
        """Make a World Coordinate System object for testing.

        Parameters
        ----------
        rotAngle : `lsst.geom.Angle`
            rotation of the CD matrix, East from North
        pixelScale : `lsst.geom.Angle`
            Pixel scale of the projection.
        crval : `lsst.afw.geom.SpherePoint`
            Coordinates of the reference pixel of the wcs.

        Returns
        -------
        `lsst.afw.geom.skyWcs.SkyWcs`
            A wcs that matches the inputs.
        """
        crpix = afwGeom.Box2D(self.bbox).getCenter()
        cdMatrix = afwGeom.makeCdMatrix(scale=pixelScale, orientation=rotAngle, flipX=True)
        wcs = afwGeom.makeSkyWcs(crpix=crpix, crval=crval, cdMatrix=cdMatrix)
        return wcs

    def makeDummyVisitInfo(self, azimuth, elevation):
        """Make a self-consistent visitInfo object for testing.

        For simplicity, the simulated observation is assumed
        to be taken on the local meridian.

        Parameters
        ----------
        azimuth : `lsst.geom.Angle`
            Azimuth angle of the simulated observation.
        elevation : `lsst.geom.Angle`
            Elevation angle of the simulated observation.

        Returns
        -------
        `lsst.afw.image.VisitInfo`
            VisitInfo for the exposure.
        """
        lsstLat = -30.244639*degrees
        lsstLon = -70.749417*degrees
        lsstAlt = 2663.
        lsstTemperature = 20.*u.Celsius  # in degrees Celcius
        lsstHumidity = 40.  # in percent
        lsstPressure = 73892.*u.pascal
        lsstWeather = Weather(lsstTemperature.value, lsstPressure.value, lsstHumidity)
        lsstObservatory = Observatory(lsstLon, lsstLat, lsstAlt)
        airmass = 1.0/np.sin(elevation.asRadians())
        era = 0.*radians  # on the meridian
        zenithAngle = 90.*degrees - elevation
        ra = lsstLon + np.sin(azimuth.asRadians())*zenithAngle/np.cos(lsstLat.asRadians())
        dec = lsstLat + np.cos(azimuth.asRadians())*zenithAngle
        visitInfo = afwImage.VisitInfo(era=era,
                                       boresightRaDec=afwGeom.SpherePoint(ra, dec),
                                       boresightAzAlt=afwGeom.SpherePoint(azimuth, elevation),
                                       boresightAirmass=airmass,
                                       boresightRotAngle=0.*radians,
                                       observatory=lsstObservatory,
                                       weather=lsstWeather
                                       )
        return visitInfo

    def testDcrCalculation(self):
        """Test that the shift in pixels due to DCR is consistently computed.

        The shift is compared to pre-computed values.
        """
        dcrAssembleCoaddTask = DcrAssembleCoaddTask(self.config)
        dcrAssembleCoaddTask.filterInfo = afwImage.Filter("gTest")
        rotAngle = 0.*radians
        azimuth = 30.*degrees
        elevation = 65.*degrees
        pixelScale = 0.2*arcseconds
        visitInfo = self.makeDummyVisitInfo(azimuth, elevation)
        wcs = self.makeDummyWcs(rotAngle, pixelScale, crval=visitInfo.getBoresightRaDec())
        dcrShift = dcrAssembleCoaddTask.calculateDcr(visitInfo, wcs)
        refShift = [afwGeom.Extent2D(-0.5363512808, -0.3103517169),
                    afwGeom.Extent2D(0.001887293861, 0.001092054612),
                    afwGeom.Extent2D(0.3886592703, 0.2248919247)]
        for shiftOld, shiftNew in zip(refShift, dcrShift):
            self.assertFloatsAlmostEqual(shiftOld.getX(), shiftNew.getX(), rtol=1e-6, atol=1e-8)
            self.assertFloatsAlmostEqual(shiftOld.getY(), shiftNew.getY(), rtol=1e-6, atol=1e-8)

    def testRotationAngle(self):
        """Test that the sky rotation angle is consistently computed.

        The rotation is compared to pre-computed values.
        """
        dcrAssembleCoaddTask = DcrAssembleCoaddTask(self.config)
        cdRotAngle = 0.*radians
        azimuth = 130.*afwGeom.degrees
        elevation = 70.*afwGeom.degrees
        pixelScale = 0.2*afwGeom.arcseconds
        visitInfo = self.makeDummyVisitInfo(azimuth, elevation)
        wcs = self.makeDummyWcs(cdRotAngle, pixelScale, crval=visitInfo.getBoresightRaDec())
        rotAngle = dcrAssembleCoaddTask.calculateRotationAngle(visitInfo, wcs)
        refAngle = -0.9344289857053072*radians
        self.assertAnglesAlmostEqual(refAngle, rotAngle, maxDiff=1e-6*radians)

    def testConditionDcrModelNoChange(self):
        """Conditioning should not change the model if it equals the reference.

        This additionally tests that the variance and mask planes do not change.
        """
        dcrAssembleCoaddTask = DcrAssembleCoaddTask(self.config)
        dcrModels = self.makeTestImages()
        refModels = [model.clone() for model in dcrModels]
        dcrAssembleCoaddTask.conditionDcrModel(refModels, dcrModels, self.bbox, gain=1.)
        for model, refModel in zip(dcrModels, refModels):
            self.assertMaskedImagesEqual(model, refModel)

    def testConditionDcrModelWithChange(self):
        """Verify conditioning when the model changes by a known amount.

        This additionally tests that the variance and mask planes do not change.
        """
        dcrAssembleCoaddTask = DcrAssembleCoaddTask(self.config)
        dcrModels = self.makeTestImages()
        refModels = [model.clone() for model in dcrModels]
        for model in dcrModels:
            model.image.array[:] *= 3.
        dcrAssembleCoaddTask.conditionDcrModel(refModels, dcrModels, self.bbox, gain=1.)
        for model, refModel in zip(dcrModels, refModels):
            refModel.image.array[:] *= 2.
            self.assertMaskedImagesEqual(model, refModel)

    def testRegularizationLargeClamp(self):
        """Frequency regularization should leave the models unchanged if the clamp factor is large.

        This also tests that noise-like pixels are not regularized.
        """
        self.config.regularizeSigma = 1.
        self.config.clampFrequency = 3.
        dcrAssembleCoaddTask = DcrAssembleCoaddTask(self.config)
        dcrModels = self.makeTestImages()
        statsCtrl = afwMath.StatisticsControl()
        modelRefs = [model.clone() for model in dcrModels]
        dcrAssembleCoaddTask.regularizeModel(dcrModels, self.bbox, self.mask, statsCtrl)
        for model, modelRef in zip(dcrModels, modelRefs):
            self.assertMaskedImagesEqual(model, modelRef)

    def testRegularizationSmallClamp(self):
        """Test that large variations between model planes are reduced.

        This also tests that noise-like pixels are not regularized.
        """
        self.config.regularizeSigma = 1.
        self.config.clampFrequency = 1.1
        dcrAssembleCoaddTask = DcrAssembleCoaddTask(self.config)
        dcrModels = self.makeTestImages()
        statsCtrl = afwMath.StatisticsControl()
        modelRefs = [model.clone() for model in dcrModels]
        templateImage = np.mean([model[self.bbox, afwImage.PARENT].image.array
                                 for model in dcrModels], axis=0)

        dcrAssembleCoaddTask.regularizeModel(dcrModels, self.bbox, self.mask, statsCtrl)
        for model, modelRef in zip(dcrModels, modelRefs):
            noiseLevel = dcrAssembleCoaddTask.calculateNoiseCutoff(modelRef, statsCtrl)
            # The mask and variance planes should be unchanged
            self.assertFloatsEqual(model.mask.array, modelRef.mask.array)
            self.assertFloatsEqual(model.variance.array, modelRef.variance.array)
            # Make sure the test parameters do reduce the outliers
            self.assertGreater(np.max(modelRef.image.array - templateImage),
                               np.max(model.image.array - templateImage))
            highThreshold = templateImage*self.config.clampFrequency + noiseLevel*self.config.regularizeSigma
            self.assertTrue(np.all(model.image.array <= highThreshold))
            lowThreshold = templateImage/self.config.clampFrequency - noiseLevel
            self.assertTrue(np.all(model.image.array >= lowThreshold))

    def testModelClamp(self):
        """Test that large amplitude changes between iterations are restricted.

        This also tests that noise-like pixels are not regularized.
        """
        seed = 5
        rng = np.random.RandomState(seed)
        self.config.modelClampFactor = 2.
        self.config.dcrNumSubfilters = 1
        dcrAssembleCoaddTask = DcrAssembleCoaddTask(self.config)
        oldModel = self.makeTestImages()[0]
        xSize, ySize = self.bbox.getDimensions()
        statsCtrl = afwMath.StatisticsControl()
        residual = oldModel.clone()
        residual.image.array[:] = rng.rand(ySize, xSize)*np.max(oldModel.image.array)
        residualRef = residual.clone()

        newModel = dcrAssembleCoaddTask.clampModel(residual, oldModel, self.bbox, statsCtrl)

        # The mask and variance planes should be unchanged
        self.assertFloatsEqual(newModel.mask.array, oldModel.mask.array)
        self.assertFloatsEqual(newModel.variance.array, oldModel.variance.array)
        # Make sure the test parameters do reduce the outliers
        self.assertGreater(np.max(residualRef.image.array),
                           np.max(newModel.image.array - oldModel.image.array))
        # Check that all of the outliers are clipped
        noiseLevel = dcrAssembleCoaddTask.calculateNoiseCutoff(oldModel, statsCtrl)
        highThreshold = (oldModel.image.array*self.config.modelClampFactor +
                         noiseLevel*self.config.regularizeSigma)
        self.assertTrue(np.all(newModel.image.array <= highThreshold))
        lowThreshold = oldModel.image.array/self.config.modelClampFactor - noiseLevel
        self.assertTrue(np.all(newModel.image.array >= lowThreshold))


class MyMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
