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
import lsst.afw.detection as afwDet
import lsst.meas.algorithms as measAlg
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

class MeasurePsfConfig(pexConfig.Config):
    starSelector = measAlg.starSelectorRegistry.makeField("Star selection algorithm", default="secondMoment")
    psfDeterminer = measAlg.psfDeterminerRegistry.makeField("PSF Determination algorithm", default="pca")

    def __init__(self):
        pexConfig.Config.__init__(self)
        self.starSelector["secondMoment"].clumpNSigma = 2.0
        self.psfDeterminer["pca"].nEigenComponents = 4
        self.psfDeterminer["pca"].kernelSize = 7.0
        self.psfDeterminer["pca"].spatialOrder = 2
        self.psfDeterminer["pca"].kernelSizeMin = 25

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
        self.starSelector = self.config.starSelector.apply(schema=schema)
        self.psfDeterminer = self.config.psfDeterminer.apply(schema=schema)
        
    @pipeBase.timeMethod
    def run(self, exposure, sources):
        """Measure the PSF

        @param[in,out]   exposure      Exposure to process; measured PSF will be installed here as well.
        @param[in,out]   sources       Measured sources on exposure; flag fields will be set marking
                                       stars chosen by the star selector and PSF determiner.
        """
        assert exposure, "No exposure provided"
        assert sources, "No sources provided"
        self.log.log(self.log.INFO, "Measuring PSF")

        psfCandidateList = self.starSelector.selectStars(exposure, sources)

        psf, cellSet = self.psfDeterminer.determinePsf(exposure, psfCandidateList, self.metadata)
        self.log.log(self.log.INFO, "PSF determination using %d/%d stars." % 
                     (self.metadata.get("numGoodStars"), self.metadata.get("numAvailStars")))

        exposure.setPsf(psf)
        return pipeBase.Struct(
            psf = psf,
            cellSet = cellSet,
        )
