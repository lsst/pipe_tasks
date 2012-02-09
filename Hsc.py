#!/usr/bin/env python

import lsst.pex.logging as pexLog
import lsst.meas.astrom as measAstrom

import lsst.pipette.calibrate as pipCalibrate
import lsst.pipette.config as pipConfig

from lsst.pipette.timer import timecall

# Use hsc.meas.astrom, failing over to lsst.meas.astrom
class CalibrateHsc(pipCalibrate.Calibrate):
    @timecall
    def astrometry(self, exposure, sources, distSources, distortion=None, llc=(0,0), size=None):
        """Solve astrometry to produce WCS

        @param exposure Exposure to process
        @param sources Sources as measured (actual) positions
        @param distSources Sources with undistorted (actual) positions
        @param distortion Distortion model
        @param llc Lower left corner (minimum x,y)
        @param size Size of exposure
        @return Star matches, match metadata
        """
        assert exposure, "No exposure provided"
        assert distSources, "No sources provided"

        self.log.log(self.log.INFO, "Solving astrometry")

        try:
            import hsc.meas.astrom as hscAst
        except ImportError:
            hscAst = None

        wcs = exposure.getWcs()
        if wcs is None or hscAst is None:
            self.log.log(self.log.WARN, "Unable to use hsc.meas.astrom; reverting to lsst.meas.astrom")
            return super(CalibrateHsc, self).astrometry(exposure, sources, distSources,
                                                        distortion=distortion, llc=llc, size=size)

        if size is None:
            size = (exposure.getWidth(), exposure.getHeight())

        try:
            menu = self.config['filters']
            filterName = menu[exposure.getFilter().getName()]
            if isinstance(filterName, pipConfig.Config):
                filterName = filterName['primary']
            self.log.log(self.log.INFO, "Using catalog filter: %s" % filterName)
        except:
            self.log.log(self.log.WARN, "Unable to determine catalog filter from lookup table using %s" %
                         exposure.getFilter().getName())
            filterName = None

        if distortion is not None:
            # Removed distortion, so use low order
            oldOrder = self.config['astrometry']['sipOrder']
            self.config['astrometry']['sipOrder'] = 2

        log = pexLog.Log(self.log, "astrometry")
        wcs.shiftReferencePixel(-llc[0], -llc[1])

        try:
            astrom = hscAst.determineWcs(self.config['astrometry'].getPolicy(), exposure, distSources,
                                         log=log, forceImageSize=size, filterName=filterName)
            wcs.shiftReferencePixel(llc[0], llc[1])
            if astrom is None:
                raise RuntimeError("hsc.meas.astrom failed to determine the WCS")
        except Exception, e:
            self.log.log(self.log.WARN, "hsc.meas.astrom failed (%s); trying lsst.meas.astrom" % e)
            astrom = measAstrom.determineWcs(self.config['astrometry'].getPolicy(), exposure, distSources,
                                             log=log, forceImageSize=size, filterName=filterName)
        
        if distortion is not None:
            self.config['astrometry']['sipOrder'] = oldOrder

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
            sky = wcs.pixelToSky(distSource.getXAstrom(), distSource.getYAstrom())
            source.setRaDec(sky)

        self.display('astrometry', exposure=exposure, sources=sources, matches=matches)

        return matches, matchMeta
