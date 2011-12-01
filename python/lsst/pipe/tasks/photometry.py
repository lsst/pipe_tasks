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
import numpy

import lsst.meas.utils.sourceDetection as muDetection
import lsst.meas.utils.sourceMeasurement as muMeasurement
import lsst.pipe.base as pipeBase

class PhotometryTask(pipeBase.Task):
    """Conversion notes:
    
    Warnings:
    - display is disabled until we figure out how to turn it on and off
    - thresholdMultiplier has been moved to policy
    """
    @pipeBase.timeMethod
    def run(self, exposure, psf, apcorr=None, wcs=None):
        """Run photometry

        @param exposure Exposure to process
        @param psf PSF for photometry
        @param apcorr Aperture correction to apply
        @param wcs WCS to apply
        @return a Struct with fields:
        - sources: Measured sources
        - footprintSet: Set of footprints
        """
        assert exposure, "No exposure provided"
        assert psf, "No psf provided"
        
        # why is this needed? Why can't photometry or the user import the modules?
        self._importModules()
        
        footprintSet = self.detect(exposure, psf)

        if self.policy.doBackground:
            with self.timer("backgroundDuration"):
                bg, exposure = muDetection.estimateBackground(
                    exposure, policy=self.policy.background, subtract=True)
                del bg
        
        sources = self.measure(exposure, footprintSet, psf, apcorr=apcorr, wcs=wcs)

        self.display('phot', exposure=exposure, sources=sources, pause=True)
        return pipeBase.Struct(
            sources = sources,
            footprintSet = footprintSet,
        )

    @pipeBase.timeMethod
    def detect(self, exposure, psf):
        """Detect positive sources

        @param exposure Exposure to process
        @param psf PSF for detection
        @return Positive source footprints
        """
        assert exposure, "No exposure provided"
        assert psf, "No psf provided"
        posSources, negSources = muDetection.detectSources(
            exposure, psf, self.policy.detect, extraThreshold=self.policy.thresholdMultiplier)
        numPos = len(posSources.getFootprints()) if posSources is not None else 0
        numNeg = len(negSources.getFootprints()) if negSources is not None else 0
        if numNeg > 0:
            self.log.log(self.log.WARN, "%d negative sources found and ignored" % numNeg)
        self.log.log(self.log.INFO, "Detected %d sources to %g sigma." % (numPos, policy['thresholdValue']))
        return posSources

    @pipeBase.timeMethod
    def measure(self, exposure, footprintSet, psf, apcorr=None, wcs=None):
        """Measure sources

        @param exposure Exposure to process
        @param footprintSet Set of footprints to measure
        @param psf PSF for measurement
        @param apcorr Aperture correction to apply
        @param wcs WCS to apply
        @return Source list
        """
        assert exposure, "No exposure provided"
        assert footprintSet, "No footprintSet provided"
        assert psf, "No psf provided"
        footprints = [] # Footprints to measure
        num = len(footprintSet.getFootprints())
        self.log.log(self.log.INFO, "Measuring %d positive sources" % num)
        footprints.append([footprintSet.getFootprints(), False])
        sources = muMeasurement.sourceMeasurement(exposure, psf, footprints, self.policy.measure)

        if wcs is not None:
            muMeasurement.computeSkyCoords(wcs, sources)

        if not self.policy.applyApcorr: # actually apply the aperture correction?
            apcorr = None

        if apcorr is not None:
            self.log.log(self.log.INFO, "Applying aperture correction to %d sources" % len(sources))
            for source in sources:
                x, y = source.getXAstrom(), source.getYAstrom()

                for getter, getterErr, setter, setterErr in (
                    ('getPsfFlux', 'getPsfFluxErr', 'setPsfFlux', 'setPsfFluxErr'),
                    ('getInstFlux', 'getInstFluxErr', 'setInstFlux', 'setInstFluxErr'),
                    ('getModelFlux', 'getModelFluxErr', 'setModelFlux', 'setModelFluxErr')):
                    flux = getattr(source, getter)()
                    fluxErr = getattr(source, getterErr)()
                    if (numpy.isfinite(flux) and numpy.isfinite(fluxErr)):
                        corr, corrErr = apcorr.computeAt(x, y)
                        getattr(source, setter)(flux * corr)
                        getattr(source, setterErr)(numpy.sqrt(corr**2 * fluxErr**2 + corrErr**2 * flux**2))

#         if self._display and self._display.has_key('psfinst') and self._display['psfinst']:
#             import matplotlib.pyplot as plt
#             psfMag = -2.5 * numpy.log10(numpy.array([s.getPsfFlux() for s in sources]))
#             instMag = -2.5 * numpy.log10(numpy.array([s.getInstFlux() for s in sources]))
#             fig = plt.figure(1)
#             fig.clf()
#             ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
#             ax.set_autoscale_on(False)
#             ax.set_ybound(lower=-1.0, upper=1.0)
#             ax.set_xbound(lower=-17, upper=-7)
#             ax.plot(psfMag, psfMag-instMag, 'ro')
#             ax.axhline(0.0)
#             ax.set_title('psf - inst')
#             plt.show()

        return sources

    def _importModules(self):
        """Import modules (so they can register themselves)"""
        if self.policy.imports:
            for modName in self.policy.imports:
                try:
                    module = self.policy.imports[modName]
                    self.log.log(self.log.INFO, "Importing %s (%s)" % (modName, module))
                    exec("import " + module)
                except ImportError, err:
                    self.log.log(self.log.WARN, "Failed to import %s (%s): %s" % (modName, module, err))


class RephotometryTask(PhotometryTask):
    def run(self, exposure, footprintSet, psf, apcorr=None, wcs=None):
        """Photometer footprints that have already been detected

        @param exposure Exposure to process
        @param footprintSet Set of footprints to rephotometer
        @param psf PSF for photometry
        @param apcorr Aperture correction to apply
        @param wcs WCS to apply
        @return pipeBase.Sources with fields:
        - sources: Measured sources
        """
        assert exposure, "No exposure provided"
        assert psf, "No psf provided"
        sources = self.measure(exposure, footprintSet, psf, apcorr=apcorr, wcs=wcs)
        return pipeBase.Struct(
            sources = sources,
        )

    def detect(self, exposure, psf):
        raise NotImplementedError("This method is deliberately not implemented: it should never be run!")


class PhotometryDiffTask(PhotometryTask):
    """Variant of PhotometryTask that detects and measures both negative and positive sources
    """
    def detect(self, exposure, psf):
        """Detect positive and negative sources (e.g. in a difference image)

        @param exposure Exposure to process
        @param psf PSF for detection
        @return Source footprints (positive and negative)
        """
        assert exposure, "No exposure provided"
        assert psf, "No psf provided"
        posSources, negSources = muDetection.detectSources(exposure, psf, self.policy.detect)
        numPos = len(posSources.getFootprints()) if posSources is not None else 0
        numNeg = len(negSources.getFootprints()) if negSources is not None else 0
        self.log.log(self.log.INFO, "Detected %d positive and %d negative sources to %g sigma." % 
                     (numPos, numNeg, policy['thresholdValue']))

        for f in negSources.getFootprints():
            posSources.getFootprints().push_back(f)

        return posSources
