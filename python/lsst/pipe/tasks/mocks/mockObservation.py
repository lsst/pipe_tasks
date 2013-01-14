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

import numpy

import lsst.pex.config
import lsst.afw.table
import lsst.afw.geom
import lsst.afw.cameraGeom
import lsst.afw.image
import lsst.afw.math
import lsst.afw.detection
import lsst.pipe.base

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

class MockObservationTask(lsst.pipe.base.Task):
    """Task to generate mock Exposure parameters (Wcs, Psf, Calib), intended for use as a subtask
    of MockCoaddTask.
    """

    ConfigClass = MockObservationConfig

    def __init__(self, **kwds):
        lsst.pipe.base.Task.__init__(self, **kwds)
        self.schema = lsst.afw.table.ExposureTable.makeMinimalSchema()
        self.ccdKey = self.schema.addField("ccd", type=int, doc="CCD number")
        self.visitKey = self.schema.addField("visit", type=int, doc="visit number")
        self.pointingKey = self.schema.addField("pointing", type="Coord", doc="center of visit")

    def run(self, butler, n, tractInfo, camera, catalog=None):
        """Driver that generates an ExposureCatalog of mock observations.
        """
        if catalog is None:
            catalog = lsst.afw.table.ExposureCatalog(self.schema)
        else:
            if not catalog.getSchema().contains(self.schema):
                raise ValueError("Catalog schema does not match Task schema")
        visit = 1
        for position, pa in self.generatePointings(n, tractInfo):
            for raft in camera:
                raft = lsst.afw.cameraGeom.cast_Raft(raft)
                calib = self.buildCalib()
                for ccd in raft:
                    ccd = lsst.afw.cameraGeom.cast_Ccd(ccd)
                    record = catalog.addNew()
                    record.setI(self.ccdKey, ccd.getId().getSerial())
                    record.setI(self.visitKey, visit)
                    record.setCoord(self.pointingKey, position)
                    record.setWcs(self.buildWcs(position, pa, ccd))
                    record.setCalib(calib)
                    record.setPsf(self.buildPsf(ccd))
                    record.setBBox(ccd.getAllPixels())
                    record.setId(butler.get("ccdExposureId", visit=visit, ccd=ccd.getId().getSerial(),
                                            immediate=True))
            visit += 1
        return catalog

    def generatePointings(self, n, tractInfo):
        """A generator (iterator) that yields (celestial) positions and rotation angles that define
        field locations.

        Default implementation draws random pointings that are uniform in the tract's image
        coordinate system.
        """
        wcs = tractInfo.getWcs()
        bbox = lsst.afw.geom.Box2D(tractInfo.getBBox())
        bbox.grow(lsst.afw.geom.Extent2D(-0.1 * bbox.getWidth(), -0.1 * bbox.getHeight()))
        for i in xrange(n):
            x = numpy.random.rand() * bbox.getWidth() + bbox.getMinX()
            y = numpy.random.rand() * bbox.getHeight() + bbox.getMinY()
            pa = 0.0
            if self.config.doRotate:
                pa = numpy.random.rand() * 2.0 * numpy.pi
            yield wcs.pixelToSky(x, y), pa

    def buildWcs(self, position, pa, ccd):
        """Build a simple TAN Wcs with no distortion and exactly-aligned CCDs."""
        crval = position.getPosition(lsst.afw.geom.degrees)
        pixelScale = (self.config.pixelScale * lsst.afw.geom.arcseconds).asDegrees()
        cd = (lsst.afw.geom.LinearTransform.makeScaling(pixelScale) 
              * lsst.afw.geom.LinearTransform.makeRotation(pa))
        crpix = ccd.getPixelFromPosition(lsst.afw.cameraGeom.FpPoint(0,0))
        wcs = lsst.afw.image.Wcs(crval, crpix, cd.getMatrix())
        return wcs

    def buildCalib(self):
        """Build a simple Calib object with exposure time fixed by config, fluxMag0 drawn from
        a Gaussian defined by config, and mid-time set to DateTime.now().
        """
        calib = lsst.afw.image.Calib()
        calib.setMidTime(lsst.daf.base.DateTime.now())
        calib.setExptime(self.config.expTime)
        calib.setFluxMag0(
            numpy.random.randn() * self.config.fluxMag0Sigma + self.config.fluxMag0,
            self.config.fluxMag0Sigma
            )
        return calib

    def buildPsf(self, ccd):
        """Build a simple Gaussian Psf with linearly-varying ellipticity and size.

        The Psf pattern increases sigma_x linearly along the x direction, and sigma_y
        linearly along the y direction.
        """
        bbox = ccd.getAllPixels()
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
        spatialFuncList = lsst.afw.math.Function2DList()
        spatialFuncList.append(sigmaXFunc)
        spatialFuncList.append(sigmaYFunc)
        spatialFuncList.append(angleFunc)
        kernel = lsst.afw.math.AnalyticKernel(
            self.config.psfImageSize, self.config.psfImageSize,
            lsst.afw.math.GaussianFunction2D(self.config.psfMinSigma, self.config.psfMinSigma),
            spatialFuncList
            )
        return lsst.afw.detection.KernelPsf(kernel)
