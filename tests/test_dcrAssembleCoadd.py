from __future__ import absolute_import, division, print_function
#
# LSST Data Management System
# Copyright 2008-2017 AURA/LSST.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.

from astropy import units as u
import numpy as np
import scipy.ndimage.interpolation
from scipy.ndimage.morphology import binary_dilation as dilate
import unittest

from lsst.afw.coord import Observatory, Weather
import lsst.afw.geom as afwGeom
from lsst.afw.geom import Angle, makeCdMatrix, makeSkyWcs
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.utils.tests
from lsst.pipe.tasks.dcrAssembleCoadd import DcrAssembleCoaddTask, DcrAssembleCoaddConfig


class DcrAssembleCoaddTestTask(lsst.utils.tests.TestCase, DcrAssembleCoaddTask):
    """A test case for the DCR-aware image coaddition algorithm.
    """

    def setUp(self):
        self.config = DcrAssembleCoaddConfig()
        lambdaEff = 476.31  # Use LSST g band values for the test.
        lambdaMin = 405
        lambdaMax = 552
        afwImage.utils.defineFilter("gTest", lambdaEff, lambdaMin=lambdaMin, lambdaMax=lambdaMax)
        self.filterInfo = afwImage.Filter("gTest")
        self.config.dcrNumSubfilters = 3
        badMaskPlanes = self.config.badMaskPlanes[:]
        badMaskPlanes.append("CLIPPED")
        xSize = 40
        ySize = 42
        psfSize = 2
        nSrc = 5
        seed = 5
        self.rng = np.random
        self.rng.seed(seed)
        noiseLevel = 5
        detectionSigma = 5.
        sourceSigma = 20.
        fluxRange = 2.
        xLoc = self.rng.random(nSrc)*(xSize - 2*self.config.bufferSize) + self.config.bufferSize
        yLoc = self.rng.random(nSrc)*(ySize - 2*self.config.bufferSize) + self.config.bufferSize
        self.dcrModels = []
        self.bbox = afwGeom.Box2I(afwGeom.Point2I(12345, 67890), afwGeom.Extent2I(xSize, ySize))

        imageSum = np.zeros((ySize, xSize))
        for subfilter in range(self.config.dcrNumSubfilters):
            flux = (self.rng.random(nSrc)*(fluxRange - 1.) + 1.)*sourceSigma*noiseLevel
            model = afwImage.MaskedImageF(self.bbox)
            image = model.image.array
            image += self.rng.random((ySize, xSize))*noiseLevel
            for x, y, f in zip(xLoc, yLoc, flux):
                xVals = np.exp(-np.power(np.arange(xSize) - x, 2.)/(2*np.power(psfSize, 2.)))
                yVals = np.exp(-np.power(np.arange(ySize) - y, 2.)/(2*np.power(psfSize, 2.)))
                image += f*np.outer(yVals, xVals)
            imageSum += image
            model.variance.array[:] = image
            model.mask.addMaskPlane("CLIPPED")
            self.dcrModels.append(model)
        maskVals = np.zeros_like(imageSum)
        maskVals[imageSum > detectionSigma*noiseLevel] = afwImage.Mask.getPlaneBitMask('DETECTED')
        for model in self.dcrModels:
            model.mask.array[:] = maskVals
        self.mask = self.dcrModels[0].mask

    def makeDummyWcs(self, rotAngle, pixelScale, crval):
        """Make a World Coordinate System object for testing.

        Parameters
        ----------
        rotAngle : lsst.afw.geom.Angle
            rotation of the CD matrix, East from North
        pixelScale : lsst.afw.geom.Angle
            Pixel scale of the projection.
        crval : lsst.afw.geom.SpherePoint
            Coordinates of the reference pixel of the wcs.

        Returns
        -------
        lsst.afw.geom.skyWcs.skyWcs.SkyWcs
            A wcs that matches the inputs.
        """
        crpix = afwGeom.Box2D(self.bbox).getCenter()
        cd_matrix = makeCdMatrix(scale=pixelScale, orientation=rotAngle, flipX=True)
        wcs = makeSkyWcs(crpix=crpix, crval=crval, cdMatrix=cd_matrix)
        return wcs

    def makeDummyVisitInfo(self, azimuth, elevation):
        """Make a self-consistent visitInfo object for testing.

        For simplicity, the simulated observation is assumed to be taken on the local meridian.

        Parameters
        ----------
        azimuth : lsst.afw.geom.Angle
            Azimuth angle of the simulated observation.
        elevation : lsst.afw.geom.Angle
            Elevation angle of the simulated observation.

        Returns
        -------
        lsst.afw.image.VisitInfo
            VisitInfo for the exposure.
        """
        lsstLat = -30.244639*afwGeom.degrees
        lsstLon = -70.749417*afwGeom.degrees
        lsstAlt = 2663.
        lsstTemperature = 20.*u.Celsius  # in degrees Celcius
        lsstHumidity = 40.  # in percent
        lsstPressure = 73892.*u.pascal
        lsstWeather = Weather(lsstTemperature.value, lsstPressure.value, lsstHumidity)
        lsstObservatory = Observatory(lsstLon, lsstLat, lsstAlt)
        airmass = 1.0/np.sin(elevation.asRadians())
        era = Angle(0.)  # on the meridian
        ra = lsstLon + np.sin(azimuth.asRadians())*(Angle(np.pi/2.) - elevation)/np.cos(lsstLat.asRadians())
        dec = lsstLat + np.cos(azimuth.asRadians())*(Angle(np.pi/2.) - elevation)
        visitInfo = afwImage.VisitInfo(era=era,
                                       boresightRaDec=afwGeom.SpherePoint(ra, dec),
                                       boresightAzAlt=afwGeom.SpherePoint(azimuth, elevation),
                                       boresightAirmass=airmass,
                                       boresightRotAngle=Angle(0.),
                                       observatory=lsstObservatory,
                                       weather=lsstWeather
                                       )
        return visitInfo

    def testDcrShiftCalculation(self):
        """Test that the shift in pixels due to DCR is consistently computed.

        The shift is compared to pre-computed values.
        """
        rotAngle = Angle(0.)
        azimuth = 30.*afwGeom.degrees
        elevation = 65.*afwGeom.degrees
        pixelScale = 0.2*afwGeom.arcseconds
        visitInfo = self.makeDummyVisitInfo(azimuth, elevation)
        wcs = self.makeDummyWcs(rotAngle, pixelScale, crval=visitInfo.getBoresightRaDec())
        dcrShift = self.dcrShiftCalculate(visitInfo, wcs)
        refShift = [afwGeom.Extent2D(-0.5363512808, -0.3103517169),
                    afwGeom.Extent2D(0.001887293861, 0.001092054612),
                    afwGeom.Extent2D(0.3886592703, 0.2248919247)]
        for shiftOld, shiftNew in zip(refShift, dcrShift):
            self.assertFloatsAlmostEqual(shiftOld.getX(), shiftNew.getX(), rtol=1e-6, atol=1e-8)
            self.assertFloatsAlmostEqual(shiftOld.getY(), shiftNew.getY(), rtol=1e-6, atol=1e-8)

    def testRotationAngle(self):
        """Test that he sky rotation angle is consistently computed.

        The rotation is compared to pre-computed values.
        """
        cdRotAngle = Angle(0.)
        azimuth = 130.*afwGeom.degrees
        elevation = 70.*afwGeom.degrees
        pixelScale = 0.2*afwGeom.arcseconds
        visitInfo = self.makeDummyVisitInfo(azimuth, elevation)
        wcs = self.makeDummyWcs(cdRotAngle, pixelScale, crval=visitInfo.getBoresightRaDec())
        rotAngle = self.calculateRotationAngle(visitInfo, wcs)
        refAngle = Angle(-0.9344289857053072)
        self.assertAnglesAlmostEqual(refAngle, rotAngle, maxDiff=Angle(1e-6))

    def testConditionDcrModelNoChange(self):
        """Conditioning should not change the model if it is identical to the reference.

        This additionally tests that the variance and mask planes do not change.
        """
        refModels = [model.clone() for model in self.dcrModels]
        self.conditionDcrModel(refModels, self.dcrModels, self.bbox, gain=1.)
        for model, refModel in zip(self.dcrModels, refModels):
            self.assertMaskedImagesEqual(model, refModel)

    def testConditionDcrModelWithChange(self):
        """Verify the effect of conditioning when the model changes by a known amount.

        This additionally tests that the variance and mask planes do not change.
        """
        refModels = [model.clone() for model in self.dcrModels]
        for model in self.dcrModels:
            model.image.array[:] *= 3.
        self.conditionDcrModel(refModels, self.dcrModels, self.bbox, gain=1.)
        for model, refModel in zip(self.dcrModels, refModels):
            refModel.image.array[:] *= 2.
            self.assertMaskedImagesEqual(model, refModel)

    def testShiftImagePlane(self):
        """Verify the shift calculation for the image and variance planes.

        The shift of the mask plane is tested separately since it is calculated differently.
        """
        self.config.useFFT = False
        rotAngle = Angle(0.)
        azimuth = 200.*afwGeom.degrees
        elevation = 75.*afwGeom.degrees
        pixelScale = 0.2*afwGeom.arcseconds
        visitInfo = self.makeDummyVisitInfo(azimuth, elevation)
        wcs = self.makeDummyWcs(rotAngle, pixelScale, crval=visitInfo.getBoresightRaDec())
        dcrShift = self.dcrShiftCalculate(visitInfo, wcs)
        newMaskedImage = self.convolveDcrModelPlane(self.dcrModels[0], dcrShift[0],
                                                    bbox=self.bbox, useInverse=False)
        shift = (dcrShift[0].getY(), dcrShift[0].getX())
        refImage = scipy.ndimage.interpolation.shift(self.dcrModels[0].image.array, shift)
        refVariance = scipy.ndimage.interpolation.shift(self.dcrModels[0].variance.array, shift)
        self.assertFloatsAlmostEqual(newMaskedImage.image.array, refImage)
        self.assertFloatsAlmostEqual(newMaskedImage.variance.array, refVariance)

    def testShiftMaskPlane(self):
        """Verify the shift calculation for the mask plane for large and small shifts.

        The shift of the image and variance planes are tested separately
        since they are calculated differently.
        """
        shiftAmps = [0.3, 1., self.config.bufferSize]
        shiftPhis = [phi for phi in np.pi*self.rng.random(len(shiftAmps))]
        dcrShifts = [afwGeom.Extent2D(np.cos(phi), np.sin(phi))*amp
                     for amp, phi in zip(shiftAmps, shiftPhis)]
        model = self.dcrModels[0]
        for dcrShift in dcrShifts:
            shiftedMask = self.shiftMask(model.mask, dcrShift, useInverse=False)
            newMask = self.shiftMask(shiftedMask, dcrShift, useInverse=True)
            detectMask = afwImage.Mask.getPlaneBitMask('DETECTED')

            bufferXSize = np.ceil(np.abs(dcrShift.getX()))
            bufferYSize = np.ceil(np.abs(dcrShift.getY()))
            bboxClip = self.dcrModels[0].mask.getBBox()
            # The simple comparison will not be accurate for edge pixels that fall off the image when shifted
            bboxClip.grow(afwGeom.Extent2I(-bufferXSize*2, -bufferYSize*2))

            # Shifting the mask grows each mask plane by one pixel in the direction of the shift,
            #  so a shift followed by the reverse shift should be the same as a dilation by one pixel.
            convolutionStruct = np.array([[True, True, True], [True, True, True], [True, True, True]])
            maskRefCheck = dilate(model[bboxClip, afwImage.PARENT].mask.array == detectMask,
                                  iterations=1, structure=convolutionStruct)
            newMaskCheck = newMask[bboxClip, afwImage.PARENT].array == detectMask

            self.assertFloatsEqual(newMaskCheck, maskRefCheck)

    def testRegularizationLargeClamp(self):
        """Frequency regularization should leave the models unchanged if all the clamp factor is large.

        This also tests that noise-like pixels are not regularized.
        """
        self.config.regularizeSigma = 3.
        self.config.clampFrequency = 2.
        statsCtrl = afwMath.StatisticsControl()
        modelRefs = [model.clone() for model in self.dcrModels]
        self.regularizeModel(self.dcrModels, self.bbox, self.mask, statsCtrl)
        for model, modelRef in zip(self.dcrModels, modelRefs):
            self.assertMaskedImagesEqual(model, modelRef)

    def testRegularizationSmallClamp(self):
        """Test that large variations between model planes are reduced.

        This also tests that noise-like pixels are not regularized.
        """
        self.config.regularizeSigma = 3.
        self.config.clampFrequency = 1.1
        statsCtrl = afwMath.StatisticsControl()
        modelRefs = [model.clone() for model in self.dcrModels]
        templateImage = np.sum([model.image.array for model in self.dcrModels], axis=0)
        backgroundInds = self.mask.array == 0
        noiseLevel = self.config.regularizeSigma*np.std(templateImage[backgroundInds])

        self.regularizeModel(self.dcrModels, self.bbox, self.mask, statsCtrl)
        for model, modelRef in zip(self.dcrModels, modelRefs):
            self.assertFloatsAlmostEqual(model.mask.array, modelRef.mask.array)
            self.assertFloatsAlmostEqual(model.variance.array, modelRef.variance.array)
            imageDiffHigh = model.image.array - (templateImage*self.config.clampFrequency + noiseLevel)
            self.assertLessEqual(np.max(imageDiffHigh), 0.)
            imageDiffLow = model.image.array - (templateImage/self.config.clampFrequency - noiseLevel)
            self.assertGreaterEqual(np.max(imageDiffLow), 0.)


class MyMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
