#!/usr/bin/env python

import lsst.pex.config as pexConfig
import lsst.pipe.tasks.distortion as ptDistortion

class HscDistorterConfig(pexConfig.Config):
    elevation = pexConfig.RangeField(dtype=float, doc="Elevation to use (degrees)",
                                     default=60.0, min=0.0, max=90.0)

class HscDistorter(object):
    """Distortion using HSC distEst package"""
    ConfigClass = HscDistorterConfig
    def __init__(self, config, ccd):
        self.elevation = config.elevation
        self.ccd = ccd
        self.pixelSize = ccd.getPixelSize()
        self.transform = ccd.getGlobalTransform()
        self.inverseTransform = self.transform.invert()
        angle = ccd.getOrientation().getNQuarter() * math.pi/2
        self.cos, self.sin = math.cos(angle), math.sin(angle)

    def rotate(self, x, y, reverse=False):
        sin = - self.sin if reverse else self.sin
        return self.cos * x + sin * y, self.cos * y - sin * x

    def toCorrected(self, x, y):
        point = self.transform(afwGeom.PointD(x, y))
        x, y = point.getX() / self.pixelSize, point.getY() / self.pixelSize
        distX, distY = distest.getUndistortedPosition(x, y, self.elevation)
        return self.rotate(distX, distY, reverse=False)

    def toObserved(self, x, y):
        x, y = self.rotate(x, y, reverse=True)
        undistX, undistY = distest.getDistortedPositionIterative(x, y, self.elevation)
        point = afwGeom.PointD(undistX * self.pixelSize, undistY * self.pixelSize)
        point = self.inverseTransform(point)
        return point.getX(), point.getY()

