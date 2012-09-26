# 
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
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
# MERCHANTABILIY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the LSST License Statement and 
# the GNU General Public License along with this program.  If not, 
# see <http://www.lsstcorp.org/LegalNotices/>.
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

class MatchBackgroundsConfig(pexConfig.Config):

    warpingKernelName = pexConfig.Field(
        dtype = str,
        doc = """Type of kernel for remapping""",
        default = "lanczos3"
    )
    backgroundOrder = pexConfig.Field(
        dtype = int,
        doc = """Order of background Chebyshev""",
        default = 4
    )
    backgroundBinsize = pexConfig.Field(
        dtype = int,
        doc = """Bin size for background matching""",
        default = 128 #256
    )
    writeFits = pexConfig.Field(
        dtype = bool,
        doc = """Write output fits files""",
        default = False
    )
    outputPath = pexConfig.Field(
        dtype = str,
        doc = """Location of output files""",
        default = "/astro/net/pogo3/yusra/fits/testTimesBkgd50/"
    )
    
    psfMatch = pexConfig.Field(
        dtype = bool,
        doc = """Psf match all images to the model Psf""",
        default = True
    )
    refPsfSize = pexConfig.Field(
        dtype = int,
        doc = """Size of reference Psf matrix; must be same size as SDSS Psfs""",
        default = 31
    )
    refPsfSigma = pexConfig.Field(
        dtype = float,
        doc = """Gaussian sigma for Psf FWHM (pixels)""",
        default = 3.0
    )
    useNN2 = pexConfig.Field(
        dtype = bool,
        doc = """Use NN2 to estimate difference image backgrounds.""",
        default = False
    )
    
    commonMask = pexConfig.Field(
        dtype = bool,
        doc = """True -  uses sum(all masks) for a common mask for all images in background estimate
                 False - uses only sum(2 mask) appropriate for each pair of images being matched""",
        default = False
    )
    
    useMean = pexConfig.Field(
        dtype = bool,
        doc = """True -  estimates difference image background as MEAN of unmasked pixels per bin
                 False - uses MEDIAN""",
        default = False
    )    
    useDetectionBackground = pexConfig.Field(
        dtype = bool,
        doc = """True -  True uses lsst.meas.detection.getBackground()
                False - masks, grids and fits Chebyshev polynomials to the backgrounds """,
        default = False
    )
    detectionBinSize = pexConfig.Field(
        dtype = int,
        doc = """sets the binsize for detection.getbackground, if useDetectionBackground = True """,
        default = 512
    )    
   
    # With linear background model, this should fail
    # /astro/net/pogo1/stripe82/imaging/6447/40/corr/1/fpC-006447-r1-0718.fit.gz
    maxBgRms = pexConfig.Field(
        dtype = float,
        doc = """Maximum RMS of matched background differences, in counts""",
        default = 5.0
    )

    # Clouds
    # /astro/net/pogo1/stripe82/imaging/7071/40/corr/1/fpC-007071-r1-0190.fit.gz
    minFluxMag0 = pexConfig.Field(
        dtype = float,
        doc = """Minimum flux for star of mag 0""",
        default = 1.0e+10
    )

    datasetType = pexConfig.Field(
        dtype = str,
        doc = """Name of data product to fetch (calexp, etc)""",
        default = "coaddTempExp"
    )

class MatchBackgroundsTask(pipeBase.CmdLineTask):
    ConfigClass = MatchBackgroundsConfig
    _DefaultName = "matchBackgrounds"
    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)

    @pipeBase.timeMethod
    def run(self, refDataRef, toMatchDataRef):
        self.log.log(self.log.INFO, "Matching background of %s to %s" % (toMatchDataRef.dataId, refDataRef.dataId))
        
        if not refDataRef.datasetExists(self.config.datasetType):
            raise pipeBase.TaskError("Data id %s does not exist" % (refDataRef.dataId))
        refExposure = refDataRef.get(self.config.datasetType)

        if not toMatchDataRef.datasetExists(self.config.datasetType):
            raise pipeBase.TaskError("Data id %s does not exist" % (toMatchDataRef.dataId))
        sciExposure = toMatchDataRef.get(self.config.datasetType)

        matchBackgroundModel, matchedExposure = self.matchBackgrounds(refExposure, sciExposure)

        return pipeBase.Struct(
            matchBackgroundModel = matchBackgroundModel,
            matchedExposure = matchedExposure
        )

    @pipeBase.timeMethod
    def run(self, refExposure, sciExposure):
        # Do your matching here
        return matchBackgroundModel, matchedExposure

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser
        """
        return CoaddArgumentParser(name=cls._DefaultName, datasetType=cls.config.datasetType)
