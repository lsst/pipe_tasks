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

import lsst.pex.config as pexConfig
import lsst.afw.detection as afwDet
import lsst.afw.display.ds9 as ds9
import lsst.afw.geom as afwGeom
import lsst.meas.astrom as measAst
import lsst.meas.astrom.astrom as measAstrometry
import lsst.meas.astrom.sip as astromSip
import lsst.meas.astrom.verifyWcs as astromVerify
import lsst.pipe.base as pipeBase
import lsst.pipe.tasks.distortion as pipeDist
from .detectorUtil import getCcd

class AstrometryConfig(pexConfig.Config):
    solver = pexConfig.ConfigField(
        dtype=measAst.MeasAstromConfig,
        doc = "Configuration for the astrometry solver",
    )

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
        self.centroidKey = schema.addField("centroid.distorted", type="Point<F8>",
                                           doc="centroid distorted for astrometry solver")

    @pipeBase.timeMethod
    def run(self, exposure, sources):
        """AstrometryTask an exposure: PSF, astrometry and photometry

        @param exposure Exposure to calibrate
        @param sources List of measured sources
        @return a pipeBase.Struct with fields:
        - matches: Astrometric matches
        - matchMeta: Metadata for astrometric matches
        """
        assert exposure is not None, "No exposure provided"

        llc, size = self.distort(exposure, sources)
        oldCentroidKey = sources.table.getCentroidKey()
        sources.table.defineCentroid(self.centroidKey, sources.table.getCentroidErrKey(),
                                     sources.table.getCentroidFlagKey())
        matches, matchMeta = self.astrometry(exposure, sources, llc=llc, size=size)
        sources.table.defineCentroid(oldCentroidKey, sources.table.getCentroidErrKey(),
                                     sources.table.getCentroidFlagKey())
        self.undistort(exposure, sources, matches)

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
        assert exposure, "No exposure provided"
        assert sources, "No sources provided"

        distorter = pipeDist.RadialPolyDistorter(ccd=getCcd(exposure))

        # Distort source positions

        # Get distorted image size so that astrometry_net does not clip.
        xMin, xMax, yMin, yMax = float("INF"), float("-INF"), float("INF"), float("-INF")
        for x, y in ((0.0, 0.0), (0.0, exposure.getHeight()), (exposure.getWidth(), 0.0),
                     (exposure.getWidth(), exposure.getHeight())):
            x, y = distorter.toCorrected(x, y)
            if x < xMin: xMin = x
            if x > xMax: xMax = x
            if y < yMin: yMin = y
            if y > yMax: yMax = y
        xMin = int(xMin)
        yMin = int(yMin)
        llc = (xMin, yMin)
        size = (int(xMax - xMin + 0.5), int(yMax - yMin + 0.5))
        for s in sources:
            x,y = distorter.toCorrected(s.getX(), s.getY())
            s.set(self.centroidKey.getX(), x)
            s.set(self.centroidKey.getY(), y)
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
        assert exposure, "No exposure provided"
        assert sources, "No sources provided"

        self.log.log(self.log.INFO, "Solving astrometry")

        if size is None:
            size = (exposure.getWidth(), exposure.getHeight())

        try:
            filterName = exposure.getFilter().getName()
            self.log.log(self.log.INFO, "Using filter: %s" % filterName)
        except:
            self.log.log(self.log.WARN, "Unable to determine filter name from exposure")
            filterName = None

        if False:
            if distortion is not None:
                # Removed distortion, so use low order
                oldOrder = self.config.sipOrder
                self.config.sipOrder = 2
#
# meas_astrom doesn't like -ve coordinates.  The proper thing to do is to fix that; but for now
# we'll pander to its whims
#
        xMin, yMin = llc
        if xMin != 0 or yMin != 0:
            for s in sources:
                s.set(self.centroidKey.getX(), s.get(self.centroidKey.getX()) - xMin)
                s.set(self.centroidKey.getY(), s.get(self.centroidKey.getY()) - yMin)
                
        astrometer = measAstrometry.Astrometry(self.config.solver, log=self.log)
        astrom = astrometer.determineWcs(sources, exposure)

        if xMin != 0 or yMin != 0:
            for s in sources:
                s.set(self.centroidKey.getX(), s.get(self.centroidKey.getX()) + xMin)
                s.set(self.centroidKey.getY(), s.get(self.centroidKey.getY()) + yMin)

        if False:
            if distortion is not None:
                self.config.sipOrder = oldOrder

        if astrom is None or astrom.getWcs() is None:
            raise RuntimeError("Unable to solve astrometry for %s", exposure.getDetector().getId())

        wcs = astrom.getWcs()
        wcs.shiftReferencePixel(llc[0], llc[1])

        matches = astrom.getMatches()
        matchMeta = astrom.getMatchMetadata()
        if matches is None or len(matches) == 0:
            raise RuntimeError("No astrometric matches for %s", exposure.getDetector().getId())
        self.log.log(self.log.INFO, "%d astrometric matches for %s" % \
                     (len(matches), exposure.getDetector().getId()))
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
    def undistort(self, exposure, sources, matches):
        """Undistort matches after solving astrometry, resolving WCS

        @param exposure Exposure of interest
        @param sources Sources on image (no distortion applied)
        @param matches Astrometric matches
        @param distortion Distortion model
        """
        assert exposure, "No exposure provided"
        assert sources, "No sources provided"
        assert matches, "No matches provided"

        # Undo distortion in matches
        self.log.log(self.log.INFO, "Removing distortion correction.")

        # Re-fit the WCS with the distortion undone
        if self.config.solver.calculateSip:
            self.log.log(self.log.INFO, "Refitting WCS with distortion removed")
            sip = astromSip.CreateWcsWithSip(matches, exposure.getWcs(), self.config.solver.sipOrder)
            wcs = sip.getNewWcs()
            self.log.log(self.log.INFO, "Astrometric scatter: %f arcsec (%s non-linear terms)" %
                         (sip.getScatterOnSky().asArcseconds(), "with" if wcs.hasDistortion() else "without"))
            exposure.setWcs(wcs)
            
            # Apply WCS to sources
            for index, source in enumerate(sources):
                sky = wcs.pixelToSky(source.getX(), source.getY())
                source.setCoord(sky)
        else:
            self.log.log(self.log.WARN, "Not calculating a SIP solution; matches may be suspect")
        
        self.display('astrometry', exposure=exposure, sources=sources, matches=matches)

