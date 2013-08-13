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
from contextlib import contextmanager

from lsst.pex.exceptions import LsstCppException, LengthErrorException
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.cameraGeom as afwCG
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.meas.astrom.astrom import Astrometry
from lsst.meas.astrom.sip import makeCreateWcsWithSip
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
    rejectThresh = pexConfig.RangeField(dtype=float, default=3.0, doc="Rejection threshold for Wcs fitting",
                                        min=0.0, inclusiveMin=False)
    rejectIter = pexConfig.RangeField(dtype=int, default=3, doc="Rejection iterations for Wcs fitting",
                                      min=0)

class AstrometryTask(pipeBase.Task):
    """Match input sources with a reference catalog and solve for the Wcs

    The actual matching and solving is done by the 'solver'; this Task
    serves as a wrapper for taking into account the known optical distortion.
    """
    ConfigClass = AstrometryConfig

    def __init__(self, schema, **kwds):
        pipeBase.Task.__init__(self, **kwds)
        self.centroidKey = schema.addField("centroid.distorted", type="PointD",
                                           doc="centroid distorted for astrometry solver")
        self.astrometer = None

    @pipeBase.timeMethod
    def run(self, exposure, sources):
        """Match with reference sources and calculate an astrometric solution

        The reference catalog actually used is up to the implementation
        of the solver; it will be manifested in the returned matches as
        a list of ReferenceMatch objects.

        The input sources have the centroid slot moved to a new column
        which has the positions corrected for any known optical distortion;
        the 'solver' (which is instantiated in the 'astrometry' member)
        should therefore simply use the centroids provided by calling
        "getCentroid()" on the individual source records.

        @param exposure Exposure to calibrate
        @param sources SourceCatalog of measured sources
        @return a pipeBase.Struct with fields:
        - matches: Astrometric matches
        - matchMeta: Metadata for astrometric matches
        """
        with self.distortionContext(exposure, sources) as bbox:
            results = self.astrometry(exposure, sources, bbox=bbox)

        if results.matches:
            self.refitWcs(exposure, sources, results.matches)

        return results

    @pipeBase.timeMethod
    def distort(self, exposure, sources):
        """Calculate distorted source positions

        CCD images are often affected by optical distortion that makes
        the astrometric solution higher order than linear.  Unfortunately,
        most (all?) matching algorithms require that the distortion be
        small or zero, and so it must be removed.  We do this by calculating
        (un-)distorted positions, based on a known optical distortion model
        in the Ccd.

        The distortion correction moves sources, so we return the distorted
        bounding box.

        @param[in]     exposure Exposure to process
        @param[in,out] sources  SourceCatalog; getX() and getY() will be used as inputs,
                                with distorted points in "centroid.distorted" field.
        @return bounding box of distorted exposure
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
            return exposure.getBBox(afwImage.PARENT)

        # Distort source positions
        self.log.info("Applying distortion correction: %s" % distorter.prynt())
        for s in sources:
            s.set(self.centroidKey, distorter.undistort(s.getCentroid(), ccd))

        # Get distorted image size so that astrometry_net does not clip.
        corners = numpy.array([distorter.undistort(afwGeom.Point2D(cnr), ccd) for
                               cnr in exposure.getBBox().getCorners()])
        xMin, xMax = int(corners[:,0].min()), int(corners[:,0].max() + 0.5)
        yMin, yMax = int(corners[:,1].min()), int(corners[:,1].max() + 0.5)

        return afwGeom.Box2I(afwGeom.Point2I(xMin, yMin), afwGeom.Extent2I(xMax - xMin, yMax - yMin))

    @contextmanager
    def distortionContext(self, exposure, sources):
        """Context manager that applies and removes distortion

        We move the "centroid" definition in the catalog table to
        point to the distorted positions.  This is undone on exit
        from the context.

        The input Wcs is taken to refer to the coordinate system
        with the distortion correction applied, and hence no shift
        is required when the sources are distorted.  However, after
        Wcs fitting, the Wcs is in the distorted frame so when the
        distortion correction is removed, the Wcs needs to be
        shifted to compensate.

        @param exposure: Exposure holding Wcs
        @param sources: Sources to correct for distortion
        @return bounding box of distorted exposure
        """
        # Apply distortion
        bbox = self.distort(exposure, sources)
        oldCentroidKey = sources.table.getCentroidKey()
        sources.table.defineCentroid(self.centroidKey, sources.table.getCentroidErrKey(),
                                     sources.table.getCentroidFlagKey())
        try:
            yield bbox # Execute 'with' block, providing bbox to 'as' variable
        finally:
            # Un-apply distortion
            sources.table.defineCentroid(oldCentroidKey, sources.table.getCentroidErrKey(),
                                         sources.table.getCentroidFlagKey())
            x0, y0 = exposure.getXY0()
            exposure.getWcs().shiftReferencePixel(-bbox.getMinX() + x0, -bbox.getMinY() + y0)

    @pipeBase.timeMethod
    def astrometry(self, exposure, sources, bbox=None):
        """Solve astrometry to produce WCS

        @param exposure Exposure to process
        @param sources Sources
        @param bbox Bounding box, or None to use exposure
        @return Struct(matches: star matches, matchMeta: match metadata)
        """
        if not self.config.forceKnownWcs:
            self.log.info("Solving astrometry")

        if bbox is None:
            bbox = exposure.getBBox(afwImage.PARENT)

        if not self.astrometer:
            self.astrometer = Astrometry(self.config.solver, log=self.log)

        kwargs = dict(x0=bbox.getMinX(), y0=bbox.getMinY(), imageSize=bbox.getDimensions())
        if self.config.forceKnownWcs:
            self.log.info("Forcing the input exposure's WCS")
            if self.config.solver.calculateSip:
                self.log.warn("'forceKnownWcs' and 'solver.calculateSip' options are both set." +
                              " Will try to compute a TAN-SIP WCS starting from the input WCS.")
            astrom = self.astrometer.useKnownWcs(sources, exposure=exposure, **kwargs)
        else:
            astrom = self.astrometer.determineWcs(sources, exposure, **kwargs)

        if astrom is None or astrom.getWcs() is None:
            raise RuntimeError("Unable to solve astrometry")

        matches = astrom.getMatches()
        matchMeta = astrom.getMatchMetadata()
        if matches is None or len(matches) == 0:
            raise RuntimeError("No astrometric matches")
        self.log.info("%d astrometric matches" %  (len(matches)))

        if not self.config.forceKnownWcs:
            # Note that this is the Wcs for the provided positions, which may be distorted
            exposure.setWcs(astrom.getWcs())

        self.display('astrometry', exposure=exposure, sources=sources, matches=matches)

        return pipeBase.Struct(matches=matches, matchMeta=matchMeta)

    @pipeBase.timeMethod
    def refitWcs(self, exposure, sources, matches):
        """A final Wcs solution after matching and removing distortion

        Specifically, fitting the non-linear part, since the linear
        part has been provided by the matching engine.

        @param exposure Exposure of interest
        @param sources Sources on image (no distortion applied)
        @param matches Astrometric matches

        @return the resolved-Wcs object, or None if config.solver.calculateSip is False.
        """
        sip = None
        if self.config.solver.calculateSip:
            self.log.info("Refitting WCS")
            numRejected = 0
            try:
                for i in range(self.config.rejectIter):
                    sip = makeCreateWcsWithSip(matches, exposure.getWcs(), self.config.solver.sipOrder)

                    wcs = exposure.getWcs()
                    ref = numpy.array([wcs.skyToPixel(m.first.getCoord()) for m in matches])
                    src = numpy.array([m.second.getCentroid() for m in matches])
                    diff = ref - src
                    rms = diff.std()
                    trimmed = []
                    for d, m in zip(diff, matches):
                        if numpy.all(numpy.abs(d) < self.config.rejectThresh*rms):
                            trimmed.append(m)
                        else:
                            numRejected += 1
                    if len(matches) == len(trimmed):
                        break
                    matches = trimmed

                # Final fit after rejection iterations
                sip = makeCreateWcsWithSip(matches, exposure.getWcs(), self.config.solver.sipOrder)
            except LsstCppException as e:
                if not isinstance(e.message, LengthErrorException):
                    raise
                self.log.warn("Unable to fit SIP: %s" % e)

            if sip:
                wcs = sip.getNewWcs()
                self.log.info("Astrometric scatter: %f arcsec (%s non-linear terms, %d matches, %d rejected)" %
                              (sip.getScatterOnSky().asArcseconds(),
                               "with" if wcs.hasDistortion() else "without",
                               len(matches), numRejected))
                exposure.setWcs(wcs)

            # Apply WCS to sources
            for index, source in enumerate(sources):
                sky = wcs.pixelToSky(source.getX(), source.getY())
                source.setCoord(sky)
        else:
            self.log.warn("Not calculating a SIP solution; matches may be suspect")
        
        self.display('astrometry', exposure=exposure, sources=sources, matches=matches)

        return sip
