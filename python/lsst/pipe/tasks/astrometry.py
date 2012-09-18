# 
# LSST Data Management System
# Copyright 2008, 2009, 2010, 2011 LSST Corporation.
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
import math
import numpy

import lsst.afw.geom as afwGeom
import lsst.afw.cameraGeom as afwCG
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.meas.astrom.astrom import Astrometry
from lsst.meas.astrom.sip import CreateWcsWithSip
from .detectorUtil import getCcd

class AstrometryConfig(pexConfig.Config):
    solver = pexConfig.ConfigField(
        dtype = Astrometry.ConfigClass,
        doc = "Configuration for the astrometry solver",
    )

    forceKnownWcs = pexConfig.Field(dtype=bool, doc=(
        "Assume that the input image's WCS is correct, without comparing it to any external reality." +
        " (In contrast to using Astrometry.net).  NOTE, if you set this, you probably also want to" +
        " un-set 'solver.calculateSip'; otherwise we'll still try to find a TAN-SIP WCS starting " +
        " from the existing WCS"), default=False)

class AstrometryTask(pipeBase.Task):
    """Conversion notes:
    
    Extracted from the Calibration task, since it seemed a good self-contained task.
    
    Warning: I'm not sure I'm using the filter information correctly. There are two issue:
    - Renamed policy item filters to filterTable, for clarity (it's not just a list of filter names)
      but it still needs work to flesh it out!
    - Is the filter data being handled correctly in applyColorTerms?
    - There was code that dealt with the filter table in astrometry, but it didn't seem to be doing
      anything so I removed it.
    """
    ConfigClass = AstrometryConfig

    def __init__(self, schema, **kwds):
        pipeBase.Task.__init__(self, **kwds)
        self.centroidKey = schema.addField("centroid.distorted", type="PointD",
                                           doc="centroid distorted for astrometry solver")
        self.astrometer = None

    @pipeBase.timeMethod
    def run(self, exposure, sources):
        """AstrometryTask an exposure: PSF, astrometry and photometry

        @param exposure Exposure to calibrate
        @param sources List of measured sources
        @return a pipeBase.Struct with fields:
        - matches: Astrometric matches
        - matchMeta: Metadata for astrometric matches
        """
        llc, size = self.distort(exposure, sources)
        oldCentroidKey = sources.table.getCentroidKey()
        sources.table.defineCentroid(self.centroidKey, sources.table.getCentroidErrKey(),
                                     sources.table.getCentroidFlagKey())
        matches, matchMeta = self.astrometry(exposure, sources, llc=llc, size=size)
        sources.table.defineCentroid(oldCentroidKey, sources.table.getCentroidErrKey(),
                                     sources.table.getCentroidFlagKey())
        self.refitWcs(exposure, sources, matches)

        return pipeBase.Struct(
            matches = matches,
            matchMeta = matchMeta,
        )

    @pipeBase.timeMethod
    def distort(self, exposure, sources):
        """Distort source positions before solving astrometry

        @param[in]     exposure Exposure to process
        @param[in,out] sources  SourceCatalog; getX() and getY() will be used as inputs,
                                with distorted points in "centroid.distorted" field.
        @return Lower-left corner, size of distorted image
        """
        ccd = getCcd(exposure, allowRaise=False)
        if ccd is None:
            self.log.warn("No CCD associated with exposure; assuming null distortion")
            distorter = None
        else:
            distorter = ccd.getDistortion()

        if distorter is None or exposure.getWcs().hasDistortion():
            if distorter is None:
                self.log.info("Null distortion correction")
            for s in sources:
                s.set(self.centroidKey, s.getCentroid())
            return (0,0), (exposure.getWidth(), exposure.getHeight())

        # Distort source positions
        self.log.info("Applying distortion correction: %s" % distorter.prynt())
        for s in sources:
            s.set(self.centroidKey, distorter.undistort(s.getCentroid(), ccd))

        # Get distorted image size so that astrometry_net does not clip.
        xMin, xMax, yMin, yMax = float("INF"), float("-INF"), float("INF"), float("-INF")
        for x, y in ((0.0, 0.0), (0.0, exposure.getHeight()), (exposure.getWidth(), 0.0),
                     (exposure.getWidth(), exposure.getHeight())):
            point = distorter.undistort(afwGeom.Point2D(x, y), ccd)
            x, y = point.getX(), point.getY()
            if x < xMin: xMin = x
            if x > xMax: xMax = x
            if y < yMin: yMin = y
            if y > yMax: yMax = y
        xMin = int(xMin)
        yMin = int(yMin)
        llc = (xMin, yMin)
        size = (int(xMax - xMin + 0.5), int(yMax - yMin + 0.5))

        return llc, size


    @pipeBase.timeMethod
    def astrometry(self, exposure, sources, llc=(0,0), size=None):
        """Solve astrometry to produce WCS

        @param exposure Exposure to process
        @param sources Sources
        @param llc Lower left corner (minimum x,y)
        @param size Size of exposure
        @return Star matches, match metadata
        """
        if not self.config.forceKnownWcs:
            self.log.info("Solving astrometry")

        if size is None:
            size = (exposure.getWidth(), exposure.getHeight())

        try:
            filterName = exposure.getFilter().getName()
            self.log.info("Using filter: '%s'" % filterName)
        except:
            self.log.warn("Unable to determine filter name from exposure")
            filterName = None

        if not self.astrometer:
            self.astrometer = Astrometry(self.config.solver, log=self.log)

        if self.config.forceKnownWcs:
            self.log.info("forceKnownWcs is set: using the input exposure's WCS")
            if self.config.solver.calculateSip:
                self.log.warn("Astrometry: 'forceKnownWcs' and 'solver.calculateSip' options are both set." +
                              " Will try to compute a TAN-SIP WCS starting from the assumed-correct input WCS.")
            astrom = self.astrometer.useKnownWcs(sources, exposure=exposure)
        else:
            astrom = self.astrometer.determineWcs(sources, exposure)

        if astrom is None or astrom.getWcs() is None:
            raise RuntimeError("Unable to solve astrometry")

        wcs = astrom.getWcs()
        wcs.shiftReferencePixel(llc[0], llc[1])

        matches = astrom.getMatches()
        matchMeta = astrom.getMatchMetadata()
        if matches is None or len(matches) == 0:
            raise RuntimeError("No astrometric matches")
        self.log.info("%d astrometric matches" %  (len(matches)))

        if not self.config.forceKnownWcs:
            exposure.setWcs(wcs)

        # Apply WCS to sources
        # This should be unnecessary - we update the RA/DEC in a zillion different places.  But I'm
        # not totally sure what's going on with the distortion at this point, so I'll just try to
        # replicate the old logic.
        for source in sources:
            distorted = source.get(self.centroidKey)
            sky = wcs.pixelToSky(distorted.getX(), distorted.getY())
            source.setCoord(sky) 

        self.display('astrometry', exposure=exposure, sources=sources, matches=matches)

        return matches, matchMeta

    @pipeBase.timeMethod
    def refitWcs(self, exposure, sources, matches):
        """We have better matches after solving astrometry, so re-solve the WCS

        @param exposure Exposure of interest
        @param sources Sources on image (no distortion applied)
        @param matches Astrometric matches

        @return the resolved-Wcs object, or None if config.solver.calculateSip is False.
        """
        wcs = exposure.getWcs()

        sip = None

        self.log.info("Refitting WCS")
        # Re-fit the WCS using the current matches
        if self.config.solver.calculateSip:
            try:
                sip = CreateWcsWithSip(matches, exposure.getWcs(), self.config.solver.sipOrder)
            except Exception, e:
                self.log.warn("Fitting SIP failed: %s" % e)
                sip = None

            if sip:
                wcs = sip.getNewWcs()
                self.log.info("Astrometric scatter: %f arcsec (%s non-linear terms)" %
                              (sip.getScatterOnSky().asArcseconds(), "with" if wcs.hasDistortion() else "without"))
                exposure.setWcs(wcs)
            
                # Apply WCS to sources
                for index, source in enumerate(sources):
                    sky = wcs.pixelToSky(source.getX(), source.getY())
                    source.setCoord(sky)
        else:
            self.log.warn("Not calculating a SIP solution; matches may be suspect")
        
        self.display('astrometry', exposure=exposure, sources=sources, matches=matches)

        return sip
