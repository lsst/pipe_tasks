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

from collections import namedtuple
from astropy import units as u
import numpy as np
import scipy.ndimage.interpolation
from scipy.ndimage.morphology import binary_dilation as dilate
import unittest

from lsst.afw.coord import Coord, IcrsCoord, Observatory, Weather
import lsst.afw.geom as afwGeom
from lsst.afw.geom import Angle, makeCdMatrix, makeSkyWcs
import lsst.afw.image as afwImage
import lsst.utils.tests
from lsst.pipe.tasks.dcrAssembleCoadd import DcrAssembleCoaddTask, DcrAssembleCoaddConfig


class DcrAssembleCoaddTestTask(lsst.utils.tests.TestCase):

    def setUp(self):
        self.config = DcrAssembleCoaddConfig()
        self.config.dcrNSubbands = 3
        self.config.lambdaEff = 478.
        self.config.filterWidth = 147.
        self.config.filterName = 'g'
        badMaskPlanes = self.config.badMaskPlanes[:]
        badMaskPlanes.append("CLIPPED")
        size = 40
        psfSize = 2
        nSrc = 5
        seed = 5
        self.randGen = np.random
        self.randGen.seed(seed)
        noiseLevel = 5
        sourceSigma = 20.
        fluxRange = 2.
        edgeDist = self.config.bufferSize
        xLoc = self.randGen.random(nSrc)*(size - 2*edgeDist) + edgeDist
        yLoc = self.randGen.random(nSrc)*(size - 2*edgeDist) + edgeDist
        self.dcrModels = []
        self.bbox = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(size, size))
        self.pixelScale = 0.2*afwGeom.arcseconds

        imageSum = np.zeros((size, size))
        for subfilter in range(self.config.dcrNSubbands):
            flux = (self.randGen.random(nSrc)*(fluxRange - 1.) + 1.)*sourceSigma*noiseLevel
            model = afwImage.MaskedImageF(self.bbox)
            image = model.getImage().getArray()
            image += self.randGen.random((size, size))*noiseLevel
            for x, y, f in zip(xLoc, yLoc, flux):
                xVals = np.exp(-np.power(np.arange(size) - x, 2.)/(2*np.power(psfSize, 2.)))
                yVals = np.exp(-np.power(np.arange(size) - y, 2.)/(2*np.power(psfSize, 2.)))
                image += f*np.outer(yVals, xVals)
            imageSum += image
            model.getVariance().getArray()[:, :] = image
            model.mask.addMaskPlane("CLIPPED")
            self.dcrModels.append(model)
        maskVals = np.zeros_like(imageSum)
        maskVals[imageSum > 5*noiseLevel] = afwImage.Mask.getPlaneBitMask('DETECTED')
        for model in self.dcrModels:
            model.getMask().getArray()[:, :] = maskVals
        self.mask = self.dcrModels[0].getMask()

    def makeDummyWcs(self, rotAngle, visitInfo=None):
        if visitInfo is None:
            crval = IcrsCoord(Angle(0.), Angle(0.))
        else:
            crval = visitInfo.getBoresightRaDec()
        crpix = afwGeom.Box2D(self.bbox).getCenter()
        cd_matrix = makeCdMatrix(scale=self.pixelScale, orientation=rotAngle, flipX=True)
        wcs = makeSkyWcs(crpix=crpix, crval=crval, cdMatrix=cd_matrix)
        return wcs

    def makeDummyVisitInfo(self, azimuth, elevation):
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
                                       boresightRaDec=IcrsCoord(ra, dec),
                                       boresightAzAlt=Coord(azimuth, elevation),
                                       boresightAirmass=airmass,
                                       boresightRotAngle=Angle(0.),
                                       observatory=lsstObservatory,
                                       weather=lsstWeather
                                       )
        return visitInfo

    def testShift(self):
        dcr = namedtuple("dcr", ["dx", "dy"])
        rotAngle = Angle(0.)
        azimuth = 30.*afwGeom.degrees
        elevation = 65.*afwGeom.degrees
        visitInfo = self.makeDummyVisitInfo(azimuth, elevation)
        wcs = self.makeDummyWcs(rotAngle, visitInfo=visitInfo)
        dcrShift = DcrAssembleCoaddTask.dcrShiftCalculate(visitInfo, wcs,
                                                          self.config.lambdaEff,
                                                          self.config.filterWidth,
                                                          self.config.dcrNSubbands)
        refShift = [dcr(dx=-0.55832265311722795, dy=-0.32306512577396451),
                    dcr(dx=-0.018151534656568987, dy=-0.010503116422151829),
                    dcr(dx=0.36985291822812622, dy=0.21400990785188412)]
        for shiftOld, shiftNew in zip(refShift, dcrShift):
            self.assertFloatsAlmostEqual(shiftOld.dx, shiftNew.dx, rtol=1e-6, atol=1e-8)
            self.assertFloatsAlmostEqual(shiftOld.dy, shiftNew.dy, rtol=1e-6, atol=1e-8)

    def testRotationAngle(self):
        cdRotAngle = Angle(0.)
        azimuth = 130.*afwGeom.degrees
        elevation = 70.*afwGeom.degrees
        visitInfo = self.makeDummyVisitInfo(azimuth, elevation)
        wcs = self.makeDummyWcs(cdRotAngle, visitInfo=visitInfo)
        rotAngle = DcrAssembleCoaddTask.calculateRotationAngle(visitInfo, wcs)
        refAngle = Angle(-0.9344289857053072)
        self.assertAnglesNearlyEqual(refAngle, rotAngle)

    def testConditionDcrModel(self):
        refModels = [model.clone() for model in self.dcrModels]
        DcrAssembleCoaddTask.conditionDcrModel(refModels, self.dcrModels, self.bbox)
        for model, refModel in zip(self.dcrModels, refModels):
            self.assertFloatsAlmostEqual(model.getImage().getArray(), refModel.getImage().getArray())
            self.assertFloatsAlmostEqual(model.getVariance().getArray(), refModel.getVariance().getArray())
            self.assertFloatsAlmostEqual(model.getMask().getArray(), refModel.getMask().getArray())

    def testShiftImagePlane(self):
        rotAngle = Angle(0.)
        azimuth = 200.*afwGeom.degrees
        elevation = 75.*afwGeom.degrees
        visitInfo = self.makeDummyVisitInfo(azimuth, elevation)
        wcs = self.makeDummyWcs(rotAngle, visitInfo=visitInfo)
        dcrShift = DcrAssembleCoaddTask.dcrShiftCalculate(visitInfo, wcs,
                                                          self.config.lambdaEff,
                                                          self.config.filterWidth,
                                                          self.config.dcrNSubbands)
        newMaskedImage = DcrAssembleCoaddTask.convolveDcrModelPlane(self.dcrModels[0], dcrShift[0],
                                                                    bbox=self.bbox,
                                                                    useFFT=False,
                                                                    useInverse=False)
        shift = (dcrShift[0].dy, dcrShift[0].dx)
        refImage = scipy.ndimage.interpolation.shift(self.dcrModels[0].getImage().getArray(), shift)
        refVariance = scipy.ndimage.interpolation.shift(self.dcrModels[0].getVariance().getArray(), shift)
        self.assertFloatsAlmostEqual(newMaskedImage.getImage().getArray(), refImage)
        self.assertFloatsAlmostEqual(newMaskedImage.getVariance().getArray(), refVariance)

    def testShiftMaskPlane(self):
        rotAngle = Angle(0.)
        azimuth = 200.*afwGeom.degrees
        elevation = 75.*afwGeom.degrees
        visitInfo = self.makeDummyVisitInfo(azimuth, elevation)
        wcs = self.makeDummyWcs(rotAngle, visitInfo=visitInfo)
        dcrShift = DcrAssembleCoaddTask.dcrShiftCalculate(visitInfo, wcs,
                                                          self.config.lambdaEff,
                                                          self.config.filterWidth,
                                                          self.config.dcrNSubbands)
        model = self.dcrModels[0]
        shiftedMask = DcrAssembleCoaddTask.shiftMask(model.getMask(), dcrShift[0],
                                                     useInverse=False)
        newMask = DcrAssembleCoaddTask.shiftMask(shiftedMask, dcrShift[0],
                                                 useInverse=True)
        detectMask = afwImage.Mask.getPlaneBitMask('DETECTED')

        bufferXSize = np.ceil(np.abs(dcrShift[0].dx)) + 1
        bufferYSize = np.ceil(np.abs(dcrShift[0].dy)) + 1
        bboxClip = self.dcrModels[0].getMask().getBBox()
        bboxClip.grow(afwGeom.Extent2I(-bufferXSize, -bufferYSize))

        convolutionStruct = np.array([[True, True, True], [True, True, True], [True, True, True]])
        maskRefCheck = dilate(model[bboxClip].getMask().getArray() == detectMask,
                              iterations=1, structure=convolutionStruct)
        newMaskCheck = newMask[bboxClip].getArray() == detectMask

        self.assertFloatsEqual(newMaskCheck, maskRefCheck)

    def testRegularizationLargeClamp(self):
        """! Frequency regularization should leave the models unchanged if all the clamp factor is large.

        This also tests that noise-like pixels are not regularized.
        """
        modelRefs = [model.clone() for model in self.dcrModels]
        DcrAssembleCoaddTask.regularizeModel(self.dcrModels, self.bbox, self.mask, nSigma=3., clamp=2.)
        for model, modelRef in zip(self.dcrModels, modelRefs):
            self.assertMaskedImagesEqual(model, modelRef)

    def testRegularizationSmallClamp(self):
        """! Test that large variations between model planes are reduced.

        This also tests that noise-like pixels are not regularized.
        """
        nSigma = 3.
        clamp = 1.1
        modelRefs = [model.clone() for model in self.dcrModels]
        templateImage = np.sum([model.getImage().getArray() for model in self.dcrModels], axis=0)
        backgroundInds = self.mask.getArray() == 0
        noiseLevel = nSigma*np.std(templateImage[backgroundInds])

        DcrAssembleCoaddTask.regularizeModel(self.dcrModels, self.bbox, self.mask, nSigma=nSigma, clamp=clamp)
        for model, modelRef in zip(self.dcrModels, modelRefs):
            self.assertFloatsAlmostEqual(model.getMask().getArray(), modelRef.getMask().getArray())
            self.assertFloatsAlmostEqual(model.getVariance().getArray(), modelRef.getVariance().getArray())
            imageDiffHigh = model.getImage().getArray() - (templateImage*clamp + noiseLevel)
            self.assertLessEqual(np.max(imageDiffHigh), 0.)
            imageDiffLow = model.getImage().getArray() - (templateImage/clamp - noiseLevel)
            self.assertGreaterEqual(np.max(imageDiffLow), 0.)


class MyMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
