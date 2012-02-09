#!/usr/bin/env python

import numpy
import lsst.daf.base as dafBase
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.meas.algorithms as measAlg
import lsst.afw.display.ds9 as ds9
import lsst.meas.algorithms as measAlg

from lsst.pipette.specific.Hsc import CalibrateHsc

# HSC-DC2 has very round galaxies that look much like stars, so we need to change the order of operations a
# bit to use the astrometry catalogue to select stars for the PSF determination.
class CalibrateHscDc2(CalibrateHsc):
    def run(self, exposure, defects=None, background=None):
        """Calibrate an exposure: PSF, astrometry and photometry

        @param exposure Exposure to calibrate
        @param defects List of defects on exposure
        @param background Background model
        @return Psf, Aperture correction, Sources, Matches
        """
        assert exposure is not None, "No exposure provided"

        do = self.config['do']['calibrate']

        psf, wcs = self.fakePsf(exposure)

        self.repair(exposure, psf, defects=defects, preserve=True)

        if do['psf'] or do['astrometry'] or do['zeropoint']:
            sources, footprints = self.phot(exposure, psf)
        else:
            sources, footprints = None, None

        if do['distortion']:
            dist = self.distortion(exposure)
        else:
            dist = None

        if do['astrometry'] or do['zeropoint'] or do['psf']:
            # Solving the astrometry prevents us from re-solving for astrometry again later, so save the wcs...
            wcs0 = exposure.getWcs().clone()

            distSources, llc, size = self.distort(exposure, sources, distortion=dist)
            matches, matchMeta = self.astrometry(exposure, sources, distSources,
                                                 distortion=dist, llc=llc, size=size)

            exposure.setWcs(wcs0)                                            # ... and restore it
        else:
            matches = None

        if do['psf']:
            psf, cellSet = self.psf(exposure, sources, matches)
        else:
            psf, cellSet = None, None

        if do['psf'] and do['apcorr']:
            apcorr = self.apCorr(exposure, cellSet) # calculate the aperture correction; we may use it later
        else:
            apcorr = None

        # Wash, rinse, repeat with proper PSF

        if do['psf']:
            self.repair(exposure, psf, defects=defects, preserve=False)

        if do['background']:
            self.background(exposure, footprints=footprints, background=background)

        if do['psf'] and (do['astrometry'] or do['zeropoint']):
            sources = self.rephot(exposure, footprints, psf, apcorr=apcorr)

        if do['astrometry'] or do['zeropoint']:
            distSources, llc, size = self.distort(exposure, sources, distortion=dist)
            matches, matchMeta = self.astrometry(exposure, sources, distSources,
                                                 distortion=dist, llc=llc, size=size)
            self.undistort(exposure, sources, matches, distortion=dist)
            self.verifyAstrometry(exposure, matches)
        else:
            matches, matchMeta = None, None

        if do['zeropoint']:
            self.zeropoint(exposure, matches)

        self.display('calibrate', exposure=exposure, sources=sources, matches=matches)
        return psf, apcorr, sources, matches, matchMeta

    def psf(self, exposure, sources, matches):
        """Measure the PSF

        @param exposure Exposure to process
        @param sources Measured sources on exposure
        @param matches (optional) A matchlist as returned by self.astrometry
        """
        assert exposure, "No exposure provided"
        assert sources, "No sources provided"
        psfPolicy = self.config['psf']

        algName   = psfPolicy['algorithmName']
        algPolicy = psfPolicy['algorithm']
        metadata = dafBase.PropertyList()
        self.log.log(self.log.INFO, "Measuring PSF")

        if matches:
            if True:
                #
                # The matchList copies of the sources are not identical to the input sources,
                # so replace them with our pristine originals
                #
                matchesIn = matches
                matches = []
                for ref, source, distance in matchesIn:
                    mySources = [s for s in sources if s.getId() == source.getId()]
                    if len(mySources) != 1:
                        raise RuntimeError("Failed to find matchList source ID == %d in input source list" %
                                           source.getId())
                    mySource = mySources[0]
                    
                    matches.append((ref, mySource, distance))

                    if False:
                        print mySource.getXAstrom(), source.getXAstrom() - mySource.getXAstrom(), \
                              mySource.getYAstrom(), source.getYAstrom() - mySource.getYAstrom()
            

        psfCandidateList = self.select(exposure, matches, algPolicy)

        psfDeterminer = measAlg.makePsfDeterminer(algName, algPolicy.getPolicy())
        psf, cellSet = psfDeterminer.determinePsf(exposure, psfCandidateList, metadata)
        exposure.setPsf(psf)
        return psf, cellSet


    def select(self, exposure, matches, psfPolicy):
        """Get a list of suitable stars to construct a PSF."""

        import lsstDebug
        display = lsstDebug.Info(__name__).display
        displayExposure = lsstDebug.Info(__name__).displayExposure     # display the Exposure + spatialCells
        #
        # Unpack policy
        #
        kernelSize   = psfPolicy["kernelSize"]
        borderWidth  = psfPolicy["borderWidth"]
        #
        mi = exposure.getMaskedImage()

        if display and displayExposure:
            frame = 0
            ds9.mtv(mi, frame=frame, title="PSF candidates")

        psfCandidates = []

        for val in matches:
            ref, source = val[0:2]
            if not (ref.getFlagForDetection() & measAlg.Flags.STAR) or \
                   (source.getFlagForDetection() & measAlg.Flags.BAD):
                continue
            if source.getPsfFlux() <= 0.0:
                continue
            
            try:
                cand = measAlg.makePsfCandidate(source, mi)
                #
                # The setXXX methods are class static, but it's convenient to call them on
                # an instance as we don't know Exposure's pixel type (and hence cand's exact type)
                if cand.getWidth() == 0:
                    cand.setBorderWidth(borderWidth)
                    cand.setWidth(kernelSize + 2*borderWidth)
                    cand.setHeight(kernelSize + 2*borderWidth)

                im = cand.getImage().getImage()
                max = afwMath.makeStatistics(im, afwMath.MAX).getValue()
                if not numpy.isfinite(max):
                    continue

                if display and displayExposure:
                    ds9.dot("+", source.getXAstrom() - mi.getX0(), source.getYAstrom() - mi.getY0(),
                            size=4, frame=frame, ctype=ds9.CYAN)
                    ds9.dot("o", source.getXAstrom() - mi.getX0(), source.getYAstrom() - mi.getY0(),
                            size=4, frame=frame, ctype=ds9.CYAN)
            except Exception, e:
                continue

            source.setFlagForDetection(source.getFlagForDetection() | measAlg.Flags.STAR)
            psfCandidates.append(cand)

        self.log.log(self.log.INFO, "Selected %d stars for PSF" % len(psfCandidates))
        return psfCandidates
