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
from lsst.pipe.tasks.dcrModel import DcrModel, calculateDcr, calculateImageParallacticAngle


class DcrModelTestTask(lsst.utils.tests.TestCase):
    """A test case for the DCR-aware image coaddition algorithm.

    Attributes
    ----------
    bbox : `lsst.afw.geom.Box2I`
        Bounding box of the test model.
    bufferSize : `int`
        Distance from the inner edge of the bounding box
        to avoid placing test sources in the model images.
    dcrNumSubfilters : int
        Number of sub-filters used to model chromatic effects within a band.
    lambdaEff : `float`
        Effective wavelength of the full band.
    lambdaMax : `float`
        Maximum wavelength where the relative throughput
        of the band is greater than 1%.
    lambdaMin : `float`
        Minimum wavelength where the relative throughput
        of the band is greater than 1%.
    mask : `lsst.afw.image.Mask`
        Reference mask of the unshifted model.
    """

    def setUp(self):
        """Define the filter, DCR parameters, and the bounding box for the tests.
        """
        self.dcrNumSubfilters = 3
        self.lambdaEff = 476.31  # Use LSST g band values for the test.
        self.lambdaMin = 405.
        self.lambdaMax = 552.
        self.bufferSize = 5
        xSize = 40
        ySize = 42
        x0 = 12345
        y0 = 67890
        self.bbox = afwGeom.Box2I(afwGeom.Point2I(x0, y0), afwGeom.Extent2I(xSize, ySize),
                                  invert=False)

    def makeTestImages(self):
        """Make reproduceable PSF-convolved masked images for testing.

        Returns
        -------
        modelImages : `list` of `lsst.afw.image.maskedImage`
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
        modelImages = []

        imageSum = np.zeros((ySize, xSize))
        for subfilter in range(self.dcrNumSubfilters):
            flux = (rng.rand(nSrc)*(fluxRange - 1.) + 1.)*sourceSigma*noiseLevel
            sigmas = [psfSize for src in range(nSrc)]
            coordList = list(zip(xLoc, yLoc, flux, sigmas))
            model = plantSources(self.bbox, 10, 0, coordList, addPoissonNoise=False)
            model.image.array += rng.rand(ySize, xSize)*noiseLevel
            imageSum += model.image.array
            model.mask.addMaskPlane("CLIPPED")
            modelImages.append(model.maskedImage)
        maskVals = np.zeros_like(imageSum)
        maskVals[imageSum > detectionSigma*noiseLevel] = afwImage.Mask.getPlaneBitMask('DETECTED')
        for model in modelImages:
            model.mask.array[:] = maskVals
        self.mask = modelImages[0].mask
        return modelImages

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
        dcrNumSubfilters = 3
        afwImage.utils.defineFilter("gTest", self.lambdaEff,
                                    lambdaMin=self.lambdaMin, lambdaMax=self.lambdaMax)
        filterInfo = afwImage.Filter("gTest")
        rotAngle = 0.*radians
        azimuth = 30.*degrees
        elevation = 65.*degrees
        pixelScale = 0.2*arcseconds
        visitInfo = self.makeDummyVisitInfo(azimuth, elevation)
        wcs = self.makeDummyWcs(rotAngle, pixelScale, crval=visitInfo.getBoresightRaDec())
        dcrShift = calculateDcr(visitInfo, wcs, filterInfo, dcrNumSubfilters)
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
        cdRotAngle = 0.*radians
        azimuth = 130.*afwGeom.degrees
        elevation = 70.*afwGeom.degrees
        pixelScale = 0.2*afwGeom.arcseconds
        visitInfo = self.makeDummyVisitInfo(azimuth, elevation)
        wcs = self.makeDummyWcs(cdRotAngle, pixelScale, crval=visitInfo.getBoresightRaDec())
        rotAngle = calculateImageParallacticAngle(visitInfo, wcs)
        refAngle = -0.9344289857053072*radians
        self.assertAnglesAlmostEqual(refAngle, rotAngle, maxDiff=1e-6*radians)

    def testConditionDcrModelNoChange(self):
        """Conditioning should not change the model if it equals the reference.

        This additionally tests that the variance and mask planes do not change.
        """
        dcrModels = DcrModel(modelImages=self.makeTestImages())
        newModels = [dcrModels[subfilter].clone() for subfilter in range(self.dcrNumSubfilters)]
        for subfilter, newModel in enumerate(newModels):
            dcrModels.conditionDcrModel(subfilter, newModel, self.bbox, gain=1.)
            self.assertMaskedImagesEqual(dcrModels[subfilter], newModel)

    def testConditionDcrModelNoChangeHighGain(self):
        """Conditioning should not change the model if it equals the reference.

        This additionally tests that the variance and mask planes do not change.
        """
        dcrModels = DcrModel(modelImages=self.makeTestImages())
        newModels = [dcrModels[subfilter].clone() for subfilter in range(self.dcrNumSubfilters)]
        for subfilter, newModel in enumerate(newModels):
            dcrModels.conditionDcrModel(subfilter, newModel, self.bbox, gain=2.5)
            self.assertMaskedImagesAlmostEqual(dcrModels[subfilter], newModel)

    def testConditionDcrModelWithChange(self):
        """Verify conditioning when the model changes by a known amount.

        This additionally tests that the variance and mask planes do not change.
        """
        dcrModels = DcrModel(modelImages=self.makeTestImages())
        newModels = [dcrModels[subfilter].clone() for subfilter in range(self.dcrNumSubfilters)]
        for model in newModels:
            model.image.array[:] *= 3.
        for subfilter, newModel in enumerate(newModels):
            dcrModels.conditionDcrModel(subfilter, newModel, self.bbox, gain=1.)
            refModel = dcrModels[subfilter]
            refModel.image.array[:] *= 2.
            self.assertMaskedImagesAlmostEqual(refModel, newModel)

    def testRegularizationLargeClamp(self):
        """Frequency regularization should leave the models unchanged if the clamp factor is large.

        This also tests that noise-like pixels are not regularized.
        """
        regularizeSigma = 1.
        clampFrequency = 3.
        dcrModels = DcrModel(modelImages=self.makeTestImages())
        statsCtrl = afwMath.StatisticsControl()
        refModels = [dcrModels[subfilter].clone() for subfilter in range(self.dcrNumSubfilters)]
        mask = refModels[0].mask
        dcrModels.regularizeModel(self.bbox, mask, statsCtrl, regularizeSigma, clampFrequency)
        for subfilter, refModel in enumerate(refModels):
            self.assertMaskedImagesEqual(dcrModels[subfilter], refModel)

    def testRegularizationSmallClamp(self):
        """Test that large variations between model planes are reduced.

        This also tests that noise-like pixels are not regularized.
        """
        regularizeSigma = 1.
        clampFrequency = 1.1
        dcrModels = DcrModel(modelImages=self.makeTestImages())
        statsCtrl = afwMath.StatisticsControl()
        refModels = [dcrModels[subfilter].clone() for subfilter in range(self.dcrNumSubfilters)]
        mask = refModels[0].mask
        templateImage = dcrModels.getReferenceImage(self.bbox)

        dcrModels.regularizeModel(self.bbox, mask, statsCtrl, regularizeSigma, clampFrequency)
        for subfilter, refModel in enumerate(refModels):
            model = dcrModels[subfilter]
            noiseLevel = dcrModels.calculateNoiseCutoff(refModel, statsCtrl, regularizeSigma)
            # The mask and variance planes should be unchanged
            self.assertFloatsEqual(model.mask.array, refModel.mask.array)
            self.assertFloatsEqual(model.variance.array, refModel.variance.array)
            # Make sure the test parameters do reduce the outliers
            self.assertGreater(np.max(refModel.image.array - templateImage),
                               np.max(model.image.array - templateImage))
            highThreshold = templateImage*clampFrequency + regularizeSigma*noiseLevel
            self.assertTrue(np.all(model.image.array <= highThreshold))
            lowThreshold = templateImage/clampFrequency - regularizeSigma*noiseLevel
            self.assertTrue(np.all(model.image.array >= lowThreshold))

    def testModelClamp(self):
        """Test that large amplitude changes between iterations are restricted.

        This also tests that noise-like pixels are not regularized.
        """
        regularizeSigma = 3.
        modelClampFactor = 2.
        subfilter = 0
        dcrModels = DcrModel(modelImages=self.makeTestImages())
        seed = 5
        rng = np.random.RandomState(seed)
        oldModel = dcrModels[0]
        xSize, ySize = self.bbox.getDimensions()
        statsCtrl = afwMath.StatisticsControl()
        newModel = oldModel.clone()
        newModel.image.array[:] += rng.rand(ySize, xSize)*np.max(oldModel.image.array)
        newModelRef = newModel.clone()

        dcrModels.clampModel(subfilter, newModel, self.bbox, statsCtrl, regularizeSigma, modelClampFactor)

        # The mask and variance planes should be unchanged
        self.assertFloatsEqual(newModel.mask.array, oldModel.mask.array)
        self.assertFloatsEqual(newModel.variance.array, oldModel.variance.array)
        # Make sure the test parameters do reduce the outliers
        self.assertGreater(np.max(newModelRef.image.array),
                           np.max(newModel.image.array - oldModel.image.array))
        # Check that all of the outliers are clipped
        noiseLevel = dcrModels.calculateNoiseCutoff(oldModel, statsCtrl, regularizeSigma)
        highThreshold = (oldModel.image.array*modelClampFactor +
                         noiseLevel*regularizeSigma)
        self.assertTrue(np.all(newModel.image.array <= highThreshold))
        lowThreshold = oldModel.image.array/modelClampFactor - noiseLevel
        self.assertTrue(np.all(newModel.image.array >= lowThreshold))

    def testIterateModel(self):
        """Test that the DcrModel is iterable, and has the right values.
        """
        testModels = self.makeTestImages()
        refVals = [np.sum(model.image.array) for model in testModels]
        dcrModels = DcrModel(modelImages=testModels)
        for refVal, model in zip(refVals, dcrModels):
            self.assertFloatsEqual(refVal, np.sum(model.image.array))
        # Negative indices are allowed, so check that those return models from the end.
        self.assertFloatsEqual(refVals[-1], np.sum(dcrModels[-1].image.array))


class MyMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
