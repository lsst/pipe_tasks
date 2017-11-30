#
# LSST Data Management System
# Copyright 2008, 2009, 2010, 2011, 2012 LSST Corporation.
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
# see <http://www.lsstcorp.org/LegalNotices/>.
#

from __future__ import absolute_import, division, print_function
from builtins import range
import numpy as np

import lsst.pex.config
import lsst.afw.table
import lsst.afw.geom
from lsst.afw.cameraGeom import PIXELS, FOCAL_PLANE
import lsst.afw.image
import lsst.afw.math
import lsst.afw.detection
import lsst.pipe.base
from lsst.meas.base.apCorrRegistry import getApCorrNameSet


class MockObservationConfig(lsst.pex.config.Config):
    pixelScale = lsst.pex.config.Field(
        dtype=float, default=0.2, optional=False,
        doc="Pixel scale for mock WCSs in arcseconds/pixel"
    )
    doRotate = lsst.pex.config.Field(
        dtype=bool, default=True, optional=False,
        doc="Whether to randomly rotate observations relative to the tract Wcs"
    )
    fluxMag0 = lsst.pex.config.Field(
        dtype=float, default=1E11, optional=False,
        doc="Flux at zero magnitude used to define Calibs."
    )
    fluxMag0Sigma = lsst.pex.config.Field(
        dtype=float, default=100.0, optional=False,
        doc="Error on flux at zero magnitude used to define Calibs; used to add scatter as well."
    )
    expTime = lsst.pex.config.Field(
        dtype=float, default=60.0, optional=False,
        doc="Exposure time set in generated Calibs (does not affect flux or noise level)"
    )
    psfImageSize = lsst.pex.config.Field(
        dtype=int, default=21, optional=False,
        doc="Image width and height of generated Psfs."
    )
    psfMinSigma = lsst.pex.config.Field(
        dtype=float, default=1.5, optional=False,
        doc="Minimum radius for generated Psfs."
    )
    psfMaxSigma = lsst.pex.config.Field(
        dtype=float, default=3.0, optional=False,
        doc="Maximum radius for generated Psfs."
    )
    apCorrOrder = lsst.pex.config.Field(
        dtype=int, default=1, optional=False,
        doc="Polynomial order for aperture correction fields"
    )
    seed = lsst.pex.config.Field(dtype=int, default=1, doc="Seed for numpy random number generator")


class MockObservationTask(lsst.pipe.base.Task):
    """Task to generate mock Exposure parameters (Wcs, Psf, Calib), intended for use as a subtask
    of MockCoaddTask.

    @todo:
    - document "pa" in detail; angle of what to what?
    - document the catalog parameter of the run method
    """

    ConfigClass = MockObservationConfig

    def __init__(self, **kwds):
        lsst.pipe.base.Task.__init__(self, **kwds)
        self.schema = lsst.afw.table.ExposureTable.makeMinimalSchema()
        self.ccdKey = self.schema.addField("ccd", type=np.int32, doc="CCD number")
        self.visitKey = self.schema.addField("visit", type=np.int32, doc="visit number")
        self.pointingKey = lsst.afw.table.CoordKey.addFields(self.schema, "pointing", "center of visit")
        self.filterKey = self.schema.addField("filter", type=str, doc="Bandpass filter name", size=16)
        self.rng = np.random.RandomState(self.config.seed)

    def run(self, butler, n, tractInfo, camera, catalog=None):
        """Driver that generates an ExposureCatalog of mock observations.

        @param[in] butler: a data butler
        @param[in] n: number of pointings
        @param[in] camera: camera geometry (an lsst.afw.cameraGeom.Camera)
        @param[in] catalog: catalog to which to add observations (an ExposureCatalog);
            if None then a new catalog is created.

        @todo figure out what `pa` is and use that knowledge to set `boresightRotAng` and `rotType`
        """
        if catalog is None:
            catalog = lsst.afw.table.ExposureCatalog(self.schema)
        else:
            if not catalog.getSchema().contains(self.schema):
                raise ValueError("Catalog schema does not match Task schema")
        visit = 1

        for position, pa in self.makePointings(n, tractInfo):
            visitInfo = lsst.afw.image.VisitInfo(
                exposureTime = self.config.expTime,
                date = lsst.daf.base.DateTime.now(),
                boresightRaDec = position,
            )
            for detector in camera:
                calib = self.buildCalib()
                record = catalog.addNew()
                record.setI(self.ccdKey, detector.getId())
                record.setI(self.visitKey, visit)
                record.set(self.filterKey, 'r')
                record.set(self.pointingKey, position)
                record.setWcs(self.buildWcs(position, pa, detector))
                record.setCalib(calib)
                record.setVisitInfo(visitInfo)
                record.setPsf(self.buildPsf(detector))
                record.setApCorrMap(self.buildApCorrMap(detector))
                record.setBBox(detector.getBBox())
                detectorId = detector.getId()
                obj = butler.get("ccdExposureId", visit=visit, ccd=detectorId, immediate=True)
                record.setId(obj)
            visit += 1
        return catalog

    def makePointings(self, n, tractInfo):
        """Generate (celestial) positions and rotation angles that define field locations.

        Default implementation draws random pointings that are uniform in the tract's image
        coordinate system.

        @param[in] n: number of pointings
        @param[in] tractInfo: skymap tract (a lsst.skymap.TractInfo)
        @return a Python iterable over (coord, angle) pairs:
        - coord is an object position (an lsst.afw.coord.Coord)
        - angle is a position angle (???) (an lsst.afw.geom.Angle)

        The default implementation returns an iterator (i.e. the function is a "generator"),
        but derived-class overrides may return any iterable.
        """
        wcs = tractInfo.getWcs()
        bbox = lsst.afw.geom.Box2D(tractInfo.getBBox())
        bbox.grow(lsst.afw.geom.Extent2D(-0.1 * bbox.getWidth(), -0.1 * bbox.getHeight()))
        for i in range(n):
            x = self.rng.rand() * bbox.getWidth() + bbox.getMinX()
            y = self.rng.rand() * bbox.getHeight() + bbox.getMinY()
            pa = 0.0 * lsst.afw.geom.radians
            if self.config.doRotate:
                pa = self.rng.rand() * 2.0 * np.pi * lsst.afw.geom.radians
            yield wcs.pixelToSky(x, y), pa

    def buildWcs(self, position, pa, detector):
        """Build a simple TAN Wcs with no distortion and exactly-aligned CCDs.

        @param[in] position: object position on sky (an lsst.afw.coord.Coord)
        @param[in] pa: position angle (an lsst.afw.geom.Angle)
        @param[in] detector: detector information (an lsst.afw.cameraGeom.Detector)
        """
        crval = position
        pixelScale = (self.config.pixelScale * lsst.afw.geom.arcseconds).asDegrees()
        cd = (lsst.afw.geom.LinearTransform.makeScaling(pixelScale) *
              lsst.afw.geom.LinearTransform.makeRotation(pa))
        fpCtr = detector.makeCameraPoint(lsst.afw.geom.Point2D(0, 0), FOCAL_PLANE)
        crpix = detector.transform(fpCtr, PIXELS).getPoint()

        wcs = lsst.afw.image.makeWcs(crval, crpix, *cd.getMatrix().flatten())
        return wcs

    def buildCalib(self):
        """Build a simple Calib object with exposure time fixed by config, fluxMag0 drawn from
        a Gaussian defined by config, and mid-time set to DateTime.now().
        """
        calib = lsst.afw.image.Calib()
        calib.setFluxMag0(
            self.rng.randn() * self.config.fluxMag0Sigma + self.config.fluxMag0,
            self.config.fluxMag0Sigma
        )
        return calib

    def buildPsf(self, detector):
        """Build a simple Gaussian Psf with linearly-varying ellipticity and size.

        The Psf pattern increases sigma_x linearly along the x direction, and sigma_y
        linearly along the y direction.

        @param[in] detector: detector information (an lsst.afw.cameraGeom.Detector)
        @return a psf (an instance of lsst.meas.algorithms.KernelPsf)
        """
        bbox = detector.getBBox()
        dx = (self.config.psfMaxSigma - self.config.psfMinSigma) / bbox.getWidth()
        dy = (self.config.psfMaxSigma - self.config.psfMinSigma) / bbox.getHeight()
        sigmaXFunc = lsst.afw.math.PolynomialFunction2D(1)
        sigmaXFunc.setParameter(0, self.config.psfMinSigma - dx * bbox.getMinX() - dy * bbox.getMinY())
        sigmaXFunc.setParameter(1, dx)
        sigmaXFunc.setParameter(2, 0.0)
        sigmaYFunc = lsst.afw.math.PolynomialFunction2D(1)
        sigmaYFunc.setParameter(0, self.config.psfMinSigma)
        sigmaYFunc.setParameter(1, 0.0)
        sigmaYFunc.setParameter(2, dy)
        angleFunc = lsst.afw.math.PolynomialFunction2D(0)
        spatialFuncList = []
        spatialFuncList.append(sigmaXFunc)
        spatialFuncList.append(sigmaYFunc)
        spatialFuncList.append(angleFunc)
        kernel = lsst.afw.math.AnalyticKernel(
            self.config.psfImageSize, self.config.psfImageSize,
            lsst.afw.math.GaussianFunction2D(self.config.psfMinSigma, self.config.psfMinSigma),
            spatialFuncList
        )
        return lsst.meas.algorithms.KernelPsf(kernel)

    def buildApCorrMap(self, detector):
        """Build an ApCorrMap with random linearly-varying fields for all
        flux fields registered for aperture correction.

        These flux field names are used only as strings; there is no
        connection to any actual algorithms with those names or the PSF model.
        """
        order = self.config.apCorrOrder

        def makeRandomBoundedField():
            """Make an upper-left triangular coefficient array appropriate
            for a 2-d polynomial."""
            array = np.zeros((order + 1, order + 1), dtype=float)
            for n in range(order + 1):
                array[n, 0:order + 1 - n] = self.rng.randn(order + 1 - n)
            return lsst.afw.math.ChebyshevBoundedField(bbox, array)

        bbox = detector.getBBox()
        apCorrMap = lsst.afw.image.ApCorrMap()
        for name in getApCorrNameSet():
            apCorrMap.set(name + "_flux", makeRandomBoundedField())
            apCorrMap.set(name + "_fluxSigma", makeRandomBoundedField())
        return apCorrMap
