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

import lsst.afw.detection as afwDet
import lsst.afw.geom as afwGeom
import lsst.meas.astrom as measAst
import lsst.meas.astrom.sip as astromSip
import lsst.meas.astrom.verifyWcs as astromVerify
import lsst.pipe.base as pipeBase
from .distoration import createDistortion
from .detectorUtil import getCcd

class AstrometryTask(pipeBase.Task):
    """Conversion notes:
    
    Extracted from the Calibration task, since it seemed a good self-contained task.
    
    Warning: I'm not sure I'm using the filter information correctly. There are two issue:
    - Is the filter data being handled correctly in applyColorTerms?
    - There was code that dealt with the filter table in astrometry, but it didn't seem to be doing
      anything so I removed it.
    """
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

        if self.config.doDistortion:
            ccd = getCcd(exposure)
            distortion = createDistortion(ccd, self.config.distortion)
        else:
            distortion = None

        distSources, llc, size = self.distort(exposure, sources, distortion=dist)
        matches, matchMeta = self.astrometry(exposure, sources, distSources,
                                             distortion=dist, llc=llc, size=size)
        self.undistort(exposure, sources, matches, distortion=dist)
        self.verifyAstrometry(exposure, matches)

        if matches is not None and self.config.doColorTerms:
            self.applyColorTerms(exposure, matches, matchMeta)

        return pipeBase.Struct(
            matches = matches,
            matchMeta = matchMeta,
        )

    @pipeBase.timeMethod
    def distort(self, exposure, sources, distortion=None):
        """Distort source positions before solving astrometry

        @param exposure Exposure to process
        @param sources Sources with undistorted (actual) positions
        @param distortion Distortion to apply
        @return Distorted sources, lower-left corner, size of distorted image
        """
        assert exposure, "No exposure provided"
        assert sources, "No sources provided"

        if distortion is not None:
            self.log.log(self.log.INFO, "Applying distortion correction.")
            distSources = distortion.actualToIdeal(sources)
            
            # Get distorted image size so that astrometry_net does not clip.
            xMin, xMax, yMin, yMax = float("INF"), float("-INF"), float("INF"), float("-INF")
            for x, y in ((0.0, 0.0), (0.0, exposure.getHeight()), (exposure.getWidth(), 0.0),
                         (exposure.getWidth(), exposure.getHeight())):
                point = afwGeom.Point2D(x, y)
                point = distortion.actualToIdeal(point)
                x, y = point.getX(), point.getY()
                if x < xMin: xMin = x
                if x > xMax: xMax = x
                if y < yMin: yMin = y
                if y > yMax: yMax = y
            xMin = int(xMin)
            yMin = int(yMin)
            llc = (xMin, yMin)
            size = (int(xMax - xMin + 0.5), int(yMax - yMin + 0.5))
            for s in distSources:
                s.setXAstrom(s.getXAstrom() - xMin)
                s.setYAstrom(s.getYAstrom() - yMin)
        else:
            distSources = sources
            size = (exposure.getWidth(), exposure.getHeight())
            llc = (0, 0)

        self.display('distortion', exposure=exposure, sources=distSources, pause=True)
        return distSources, llc, size


    @pipeBase.timeMethod
    def astrometry(self, exposure, sources, distSources, distortion=None, llc=(0,0), size=None):
        """Solve astrometry to produce WCS

        @param exposure Exposure to process
        @param sources Sources as measured (actual) positions
        @param distSources Sources with undistorted (ideal) positions
        @param distortion Distortion model
        @param llc Lower left corner (minimum x,y)
        @param size Size of exposure
        @return Star matches, match metadata
        """
        assert exposure, "No exposure provided"
        assert distSources, "No sources provided"

        self.log.log(self.log.INFO, "Solving astrometry")

        if size is None:
            size = (exposure.getWidth(), exposure.getHeight())

        try:
            filterName = exposure.getFilter().getName()
            self.log.log(self.log.INFO, "Using filter: %s" % filterName)
        except:
            self.log.log(self.log.WARN, "Unable to determine filter name from exposure")
            filterName = None

        if distortion is not None:
            # Removed distortion, so use low order
            oldOrder = self.config.sipOrder
            self.config.sipOrder = 2

        astrom = measAst.determineWcs(self.config, exposure, distSources,
                                      log=self.log, forceImageSize=size, filterName=filterName)

        if distortion is not None:
            self.config.sipOrder = oldOrder

        if astrom is None:
            raise RuntimeError("Unable to solve astrometry for %s", exposure.getDetector().getId())

        wcs = astrom.getWcs()
        matches = astrom.getMatches()
        matchMeta = astrom.getMatchMetadata()
        if matches is None or len(matches) == 0:
            raise RuntimeError("No astrometric matches for %s", exposure.getDetector().getId())
        self.log.log(self.log.INFO, "%d astrometric matches for %s" % \
                     (len(matches), exposure.getDetector().getId()))
        exposure.setWcs(wcs)

        # Apply WCS to sources
        for index, source in enumerate(sources):
            distSource = distSources[index]
            sky = wcs.pixelToSky(distSource.getXAstrom() - llc[0], distSource.getYAstrom() - llc[1])
            source.setRaDec(sky)

            #point = afwGeom.Point2D(distSource.getXAstrom() - llc[0], distSource.getYAstrom() - llc[1])
            # in square degrees
            #areas.append(wcs.pixArea(point))

        self.display('astrometry', exposure=exposure, sources=sources, matches=matches)

        return matches, matchMeta

    @pipeBase.timeMethod
    def undistort(self, exposure, sources, matches, distortion=None):
        """Undistort matches after solving astrometry, resolving WCS

        @param exposure Exposure of interest
        @param sources Sources on image (no distortion applied)
        @param matches Astrometric matches
        @param distortion Distortion model
        """
        assert exposure, "No exposure provided"
        assert sources, "No sources provided"
        assert matches, "No matches provided"

        if distortion is None:
            # No need to do anything
            return

        # Undo distortion in matches
        self.log.log(self.log.INFO, "Removing distortion correction.")
        # Undistort directly, assuming:
        # * astrometry matching propagates the source identifier (to get original x,y)
        # * distortion is linear on very very small scales (to get x,y of catalogue)
        for m in matches:
            dx = m.first.getXAstrom() - m.second.getXAstrom()
            dy = m.first.getYAstrom() - m.second.getYAstrom()
            orig = sources[m.second.getId()]
            m.second.setXAstrom(orig.getXAstrom())
            m.second.setYAstrom(orig.getYAstrom())
            m.first.setXAstrom(m.second.getXAstrom() + dx)
            m.first.setYAstrom(m.second.getYAstrom() + dy)

        # Re-fit the WCS with the distortion undone
        if self.config.calculateSip:
            self.log.log(self.log.INFO, "Refitting WCS with distortion removed")
            sip = astromSip.CreateWcsWithSip(matches, exposure.getWcs(), self.config.sipOrder)
            wcs = sip.getNewWcs()
            self.log.log(self.log.INFO, "Astrometric scatter: %f arcsec (%s non-linear terms)" %
                         (sip.getScatterOnSky().asArcseconds(), "with" if wcs.hasDistortion() else "without"))
            exposure.setWcs(wcs)
            
            # Apply WCS to sources
            for index, source in enumerate(sources):
                sky = wcs.pixelToSky(source.getXAstrom(), source.getYAstrom())
                source.setRaDec(sky)
        else:
            self.log.log(self.log.WARN, "Not calculating a SIP solution; matches may be suspect")
        
        self.display('astrometry', exposure=exposure, sources=sources, matches=matches)

    def verifyAstrometry(self, exposure, matches):
        """Verify astrometry solution

        @param exposure Exposure of interest
        @param matches Astrometric matches
        """
        verify = dict()                    # Verification parameters
        verify.update(astromSip.sourceMatchStatistics(matches))
        verify.update(astromVerify.checkMatches(matches, exposure, log=self.log))
        for k, v in verify.items():
            exposure.getMetadata().set(k, v)

    def applyColorTerms(self, exposure, matches, matchMeta):
        """Correct astrometry for color terms
        """
        natural = exposure.getFilter().getName() # Natural band
        if natural is None:
            self.log.log(self.log.WARN, "Cannot apply color terms: filter name unknown")
            # No data to do anything
            return
        filterTable = self.config.filterTable
        filterData = filterTable.get(natural)
        if natural is None:
            self.log.log(self.log.WARN, "Cannot apply color terms: no data for filter %s" % (filterName,))
            return
        primary = filterData['primary'] # Primary band for correction
        secondary = filterData['secondary'] # Secondary band for correction

        polyData = filterData.getConfig().getDoubleArray('polynomial') # Polynomial correction
        polyData.reverse()            # Numpy wants decreasing powers
        polynomial = numpy.poly1d(polyData)

        # We already have the 'primary' magnitudes in the matches
        secondaries = measAst.readReferenceSourcesFromMetadata(
            matchMeta,
            log = self.log,
            config = self.config.astrometry,
            filterName = secondary,
        )
        secondariesDict = dict()
        for s in secondaries:
            secondariesDict[s.getId()] = (s.getPsfFlux(), s.getPsfFluxErr())
        del secondaries

        polyString = ["%f (%s-%s)^%d" % (polynomial[order+1], primary, secondary, order+1) for
                      order in range(polynomial.order)]
        self.log.log(self.log.INFO, "Adjusting reference magnitudes: %f + %s" % (polynomial[0],
                                                                                 " + ".join(polyString)))

        for m in matches:
            index = m.first.getId()
            primary = -2.5 * math.log10(m.first.getPsfFlux())
            primaryErr = m.first.getPsfFluxErr()
            
            secondary = -2.5 * math.log10(secondariesDict[index][0])
            secondaryErr = secondariesDict[index][1]

            diff = polynomial(primary - secondary)
            m.first.setPsfFlux(math.pow(10.0, -0.4*(primary + diff)))
            # XXX Ignoring the error for now
