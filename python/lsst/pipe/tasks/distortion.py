#!/usr/bin/env python

import os, math
import numpy
import lsst.pex.config as pexConfig
import lsst.afw.detection as afwDet
import lsst.afw.cameraGeom as afwCG
import lsst.afw.geom as afwGeom

"""This module defines the CameraDistortion class, which calculates the effects of optical distortions."""

distorterRegistry = pexConfig.makeRegistry(
    '''A registry of distorter factories
    
       A distorter factory makes a class with the following API:
        
       def __init__(self, config, ccd):
           """Construct a distorter
           
           @param[in] config: an instance of pexConfig.Config that configures this algorithm
           @param[in] ccd: the CCD that will provide positions to be distorted
           """
        
       def toObserved(self, x, y):
           """Return the as-observed position (x,y) of the corrected position x,y"""

       def toCorrected(self, x, y):
           """Return the corrected position (x,y) of the as-observed position x,y"""
       '''
)


class NullDistorter(object):
    """No-op distortion"""
    ConfigClass = pexConfig.Config
    def __init__(self, config, ccd):
        pass
    def toObserved(self, x, y):
        return x, y
    def toCorrected(self, x, y):
        return x, y

distorterRegistry.register("null", NullDistorter)


class RadialPolyDistorterConfig(pexConfig.Config):
    coefficients = pexConfig.ListField(dtype=float, doc="Coefficients of distortion polynomial, from 0 to n")
    observedToCorrected = pexConfig.Field(dtype=bool, doc="Coefficients represent observed-->corrected?",
                                          default=True)


class RadialPolyDistorter(object):
    """Distortion using the RadialPolyDistortion class in afw"""
    ConfigClass = RadialPolyDistorterConfig
    def __init__(self, config, ccd):
        self.ccd = ccd
        self.distortion = afwCG.RadialPolyDistortion(config.coefficients)
        self.distort = self.distortion.distort if config.observedToCorrected else self.distortion.undistort
        self.undistort = self.distortion.undistort if config.observedToCorrected else self.distortion.distort
    def toObserved(self, x, y):
        point = self.undistort(afwGeom.Point2D(x, y), self.ccd)
        return point.getX(), point.getY()
    def toCorrected(self, x, y):
        point = self.distort(afwGeom.Point2D(x, y), self.ccd)
        return point.getX(), point.getY()

distorterRegistry.register("radial", RadialPolyDistorter)


# Ignore everything after here



class RadialDistortionConfig(pexConfig.Config):
    coeffs = pexConfig.ListField(
        dtype = float,
        doc = """Radial distortion polynomial coefficients, from highest power
        down to constant (typically ending as 1.0, 0.0)""",
        minLength = 0,
        optional = False,
    )
    actualToIdeal = pexConfig.Field(
        dtype = bool,
        doc = "Whether the coefficients are suitable for actual-to-ideal (TRUE) or ideal-to-actual (FALSE)",
        default = False,
        optional = False,
    )
    step = pexConfig.Field(
        dtype = float,
        doc = "Step size for lookup table (pixels)",
        default = 10.0,
        optional = False,
    )
    
class DistortionConfig(pexConfig.Config):
    radial = pexConfig.ConfigField(
        dtype = RadialDistortionConfig,
        doc = "Radial distortion configuration",
    )
    className = pexConfig.Field(
        dtype = str,
        doc = "dotted name of distortion computation function",
        optional = True,
    )


class CameraDistortion(object):
    """This is a base class for calculating the effects of optical distortions on a camera."""

    def _distortPosition(self, x, y, *args, **kwargs):
        """Distort/undistort a single position.

        @param x X coordinate to distort
        @param y Y coordinate to distort
        """
        raise NotImplementedError("Method for %s not implemented" % __name__)

    def _distortSources(self, sources, copy=True, *args, **kwargs):
        """Common method to distort/undistort a source or sources.

        @param inSource Source or iterable of sources to distort
        @returns Copy of source or sources with modified coordinates
        """
        if hasattr(sources, "__iter__"):
            # Presumably an iterable of Sources
            if copy:
                output = type(sources)()
            else:
                output = sources
            for index, inSource in enumerate(sources):
                if copy:
                    outSource = type(inSource)(inSource)
                else:
                    outSource = inSource
                xIn, yIn = inSource.getXAstrom(), inSource.getYAstrom()
                xOut, yOut = self._distortPosition(xIn, yIn, *args, **kwargs)
                outSource.setXAstrom(xOut)
                outSource.setYAstrom(yOut)
                if copy:
                    output.append(outSource)
        elif isinstance(sources, afwDet.Source):
            if copy:
                output = type(sources)(sources)
            else:
                output = sources
            xIn, yIn = sources.getXAstrom(), sources.getYAstrom()
            xOut, yOut = self._distortPosition(xIn, yIn, *args, **kwargs)
            output.setXAstrom(xOut)
            output.setYAstrom(yOut)
        elif isinstance(sources, afwGeom.Point2D):
            if copy:
                output = type(sources)()
                output.setX(sources.getX())
                output.setY(sources.getY())
            else:
                output = sources
            xIn, yIn = sources.getX(), sources.getY()
            xOut, yOut = self._distortPosition(xIn, yIn, *args, **kwargs)
            output.setX(xOut)
            output.setY(yOut)
        else:
            raise RuntimeError("Unrecognised type: %s" % str(type(sources)))
        return output

    def actualToIdeal(self, actual, copy=True):
        """Transform source or sources from actual coordinates to ideal coodinates.

        @param measured Source or sources with actual (detector) coordinates
        @returns Copy of source or sources with ideal coordinates
        """
        return self._distortSources(actual, copy=copy)

    def idealToActual(self, ideal, copy=True):
        """Transform source or sources from ideal coordinates to actual coodinates.

        @param measured Source or sources with ideal coordinates
        @returns Copy of source or sources with actual (detector) coordinates
        """
        return self._distortSources(ideal, copy=copy)

def createDistortion(ccd, config):
    """Create a suitable CameraDistortion object

    @param ccd Ccd for distortion (sets position relative to center)
    @param config Instance of DistortionConfig
    @returns CameraDistortion specified by ccd and configuration
    """
    if config.className:
        try:
            distMod, distClassname = config.className.rsplit('.', 1)
            _temp = __import__(distMod, globals(), locals(), [distClassname], -1)
            distClass = _temp.__dict__[distClassname]
        except Exception, e:
            raise RuntimeError('Failed to import distortion class %s: %s' % \
                               (config.className, e))
        return distClass(ccd, config)
    elif config.radial:
        return RadialDistortion(ccd, config.radial)
    else:
        return NullDistortion()

createDistortion.ConfigClass = DistortionConfig



class RadialDistortion(CameraDistortion):
    ConfigClass = RadialDistortionConfig

    def __init__(self, ccd, config):
        """Constructor

        @param ccd Ccd for distortion (sets position relative to center)
        @param config Configuration for distortion
        """
        self.coeffs = config.coeffs
        self.actualToIdeal = config.actualToIdeal
        self.step = config.step

        position = ccd.getCenter().getPixels(ccd.getPixelSize()) # Centre of CCD on focal plane
        center = ccd.getSize().getPixels(ccd.getPixelSize()) / 2.0 # Central pixel
        # Pixels from focal plane center to CCD corner
        self.x0 = position.getX() - center.getX()
        self.y0 = position.getY() - center.getY()

        bbox = ccd.getAllPixels()
        corners = ((bbox.getMinX(), bbox.getMinY()),
                   (bbox.getMinX(), bbox.getMaxY()),
                   (bbox.getMaxX(), bbox.getMinY()),
                   (bbox.getMaxX(), bbox.getMaxY()))
        cornerRadii = list()
        for c in corners:
            cornerRadii.append(math.hypot(c[0] + self.x0, c[1] + self.y0))
        self.minRadius = min(min(cornerRadii) - self.step, 0)
        self.maxRadius = max(cornerRadii) + self.step

        self._init()
        return

    def __getstate__(self):
        """Get state for pickling"""
        state = dict(self.__dict__)
        # Remove big, easily regenerated components
        del state['actual']
        del state['ideal']
        return state

    def __setstate__(self, state):
        """Restore state for unpickling"""
        for key, value in state.items():
            self.__dict__[key] = value
        self._init()
        return

    def _init(self):
        """Set up distortion lookup table"""
        poly = numpy.poly1d(self.coeffs, variable='r')
        fromRadii = numpy.arange(self.minRadius, self.maxRadius, self.step, dtype=float)
        toRadii = numpy.polyval(poly, fromRadii)

        if self.actualToIdeal:
            # Actual --> ideal
            self.actual = fromRadii
            self.ideal = toRadii
        else:
            # Ideal --> actual
            self.actual = toRadii
            self.ideal = fromRadii
            # Extend to cover minRadius --> maxRadius in actual space
            while self.actual[0] > self.minRadius:
                ideal = self.ideal[0] - self.step
                actual = numpy.polyval(poly, ideal)
                numpy.insert(self.ideal, 0, ideal)
                numpy.insert(self.actual, 0, actual)
            while self.actual[-1] < self.maxRadius:
                ideal = self.ideal[-1] + self.step
                actual = numpy.polyval(poly, ideal)
                numpy.insert(self.ideal, -1, ideal)
                numpy.insert(self.actual, -1, actual)
        return

    def _distortPosition(self, x, y, rFrom=None, rTo=None):
        """Distort/undistort a position.

        @param x X coordinate to distort
        @param y Y coordinate to distort
        @param rFrom Vector of lookup table providing the source radii
        @param rTo Vector of lookup table providing the target radii
        @returns Copy of input source with distorted/undistorted coordinates
        """
        assert rFrom is not None, "Source radii not provided"
        assert rTo is not None, "Target radii not provided"
        x += self.x0
        y += self.y0
        theta = math.atan2(y, x)
        radius = math.hypot(x, y)
        if radius < rFrom[0] or radius > rFrom[-1]:
            raise RuntimeError("Radius (%f from %f,%f) is outside lookup table bounds (%f,%f)" %
                               (radius, x - self.x0, y - self.y0, rFrom[0], rFrom[-1]))
        r = numpy.interp(radius, rFrom, rTo)
        if r < rTo[0] or r > rTo[-1]:
            raise RuntimeError("Radius (%f-->%f from %f,%f) is outside lookup table bounds (%f,%f)" %
                               (radius, r, x - self.x0, y - self.y0, rTo[0], rTo[-1]))
        return r * math.cos(theta) - self.x0, r * math.sin(theta) - self.y0

    def actualToIdeal(self, sources, copy=True):
        return self._distortSources(sources, rFrom=self.actual, rTo=self.ideal, copy=copy)

    def idealToActual(self, sources, copy=True):
        return self._distortSources(sources, rFrom=self.ideal, rTo=self.actual, copy=copy)

