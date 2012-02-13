#!/usr/bin/env python

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.meas.astrom as measAstrom
import lsst.pipe.tasks.astrometry as ptAstrometry
import hsc.meas.astrom.astrom as hscAstrom

class HscAstrometryConfig(ptAstrometry.AstrometryConfig):
    solver = pexConfig.ConfigField(
        dtype=hscAstrom.TaburAstrometryConfig,
        doc = "Configuration for the Tabur astrometry solver"
        )


# Use hsc.meas.astrom, failing over to lsst.meas.astrom
class HscAstrometryTask(ptAstrometry.AstrometryTask):
    ConfigClass = hscAstrom.TaburAstrometryConfig
    @pipeBase.timeMethod
    def astrometry(self, exposure, sources, distSources, llc=(0,0), size=None):
        """Solve astrometry to produce WCS

        @param exposure Exposure to process
        @param sources Sources as measured (actual) positions
        @param distSources Sources with undistorted (ideal) positions
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
            return super(HscAstrometryTask, self).astrometry(exposure, sources, distSources,
                                                             llc=llc, size=size)

        if size is None:
            size = (exposure.getWidth(), exposure.getHeight())

        wcs.shiftReferencePixel(-llc[0], -llc[1])

        try:
            astrometer = hscAstrom.TaburAstrometry(self.config.solver, log=self.log)
            astrom = astrometer.determineWcs(distSources, exposure)
            wcs.shiftReferencePixel(llc[0], llc[1])
            if astrom is None:
                raise RuntimeError("hsc.meas.astrom failed to determine the WCS")
        except Exception, e:
            self.log.log(self.log.WARN, "hsc.meas.astrom failed (%s); trying lsst.meas.astrom" % e)
            astrometer = measAstrometry.Astrometry(self.config.solver, log=self.log)
            astrom = astrometer.determineWcs(distSources, exposure)
        
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
