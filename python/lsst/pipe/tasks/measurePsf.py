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
import random
import lsst.meas.algorithms as measAlg
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.table as afwTable

class MeasurePsfConfig(pexConfig.Config):
    starSelector = measAlg.starSelectorRegistry.makeField("Star selection algorithm", default="secondMoment")
    psfDeterminer = measAlg.psfDeterminerRegistry.makeField("PSF Determination algorithm", default="pca")
    reserveFraction = pexConfig.Field(
        dtype = float,
        doc = "Fraction PSF candidates to reserve from fitting",
        default = -1.0,
    )
    reserveSeed = pexConfig.Field(
        dtype = int,
        doc = "This number will be multplied by the exposure ID to set the random seed for reserving candidates",
        default = 1,
    )

class MeasurePsfTask(pipeBase.Task):
    """Conversion notes:
    
    Split out of Calibrate since it seemed a good self-contained task
    
    @warning
    - I'm not sure I'm using metadata correctly (to replace old sdqa code)
    - The star selector and psf determiner registries will have to be modified to return a class,
      which has a ConfigClass attribute and can be instantiated with a config. Until then, there's no
      obvious way to get a registry algorithm's Config from another Config.
    """
    ConfigClass = MeasurePsfConfig

    def __init__(self, schema=None, **kwargs):
        pipeBase.Task.__init__(self, **kwargs)
        if schema is not None:
            self.candidateKey = schema.addField(
                "calib.psf.candidate", type="Flag",
                doc=("Flag set if the source was a candidate for PSF determination, "
                     "as determined by the '%s' star selector.") % self.config.starSelector.name
                )
            self.usedKey = schema.addField(
                "calib.psf.used", type="Flag",
                doc=("Flag set if the source was actually used for PSF determination, "
                     "as determined by the '%s' PSF determiner.") % self.config.psfDeterminer.name
                )

            if self.config.reserveFraction > 0:
                self.reservedKey = schema.addField(
                    "calib.psf.reserved", type="Flag",
                    doc=("Flag set if the source was selected as a PSF candidate, but was "
                         "reserved from the PSF fitting."))
                
        else:
            self.candidateKey = None
            self.usedKey = None
        self.starSelector = self.config.starSelector.apply()
        self.psfDeterminer = self.config.psfDeterminer.apply()
        
    @pipeBase.timeMethod
    def run(self, exposure, sources, expId=0, matches=None):
        """Measure the PSF

        @param[in,out]   exposure      Exposure to process; measured PSF will be installed here as well.
        @param[in,out]   sources       Measured sources on exposure; flag fields will be set marking
                                       stars chosen by the star selector and PSF determiner.
        @param[in]       expId         Exposure id used for generating random seed.
        @param[in]       matches       ReferenceMatchVector, as returned by the AstrometryTask, used
                                       by star selectors that refer to an external catalog.
        """
        self.log.info("Measuring PSF")

        psfCandidateList = self.starSelector.selectStars(exposure, sources, matches=matches)
        reserveList = []
        
        if self.config.reserveFraction > 0 :
            random.seed(self.config.reserveSeed*expId)
            reserveList = random.sample(psfCandidateList, 
                                        int((self.config.reserveFraction)*len(psfCandidateList)))

            for cand in reserveList:
                psfCandidateList.remove(cand)

            if reserveList and self.reservedKey is not None:
                for cand in reserveList:
                    source = cand.getSource()
                    source.set(self.reservedKey,True)
            
        if psfCandidateList and self.candidateKey is not None:
            for cand in psfCandidateList:
                source = cand.getSource()
                source.set(self.candidateKey, True)

        self.log.info("PSF star selector found %d candidates" % len(psfCandidateList))
        if self.config.reserveFraction > 0 :
            self.log.info("Reserved %d candidates from the fitting" % len(reserveList))

        psf, cellSet = self.psfDeterminer.determinePsf(exposure, psfCandidateList, self.metadata,
                                                       flagKey=self.usedKey)
        self.log.info("PSF determination using %d/%d stars." %
                     (self.metadata.get("numGoodStars"), self.metadata.get("numAvailStars")))

        exposure.setPsf(psf)
        return pipeBase.Struct(
            psf = psf,
            cellSet = cellSet,
        )
