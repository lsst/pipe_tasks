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
import argparse
import re
import sys
import itertools
import traceback
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base.argumentParser import IdValueAction

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

        print "SUCCESS!!!", type(refExposure), type(refExposure)
        return pipeBase.Struct(
            matchBackgroundModel = matchBackgroundModel,
            matchedExposure = matchedExposure
        )

    @pipeBase.timeMethod
    def matchBackgrounds(self, refExposure, sciExposure):
        # Do your matching here
        matchBackgroundModel = None
        matchedExposure      = None
        return matchBackgroundModel, matchedExposure

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser
        """
        return MatchBackgroundsParser(name=cls._DefaultName, datasetType=cls.ConfigClass().datasetType)

    def runParsedCmd(self, parsedCmd):
        """Run the task, given the results of parsing a command line."""
        self.runDataRefList(parsedCmd.refDataRefList[0], parsedCmd.dataRefList, doRaise=parsedCmd.doraise)

    def runDataRefList(self, refDataRef, dataRefList, doRaise=False):
        """Execute the parsed command on a sequence of dataRefs,
        including writing the config and metadata.
        """
        name = self._DefaultName
        result = []
        for dataRef in dataRefList:
            try:
                configName = self._getConfigName()
                if configName is not None:
                    dataRef.put(self.config, configName)
            except Exception, e:
                self.log.log(self.log.WARN, "Could not persist config for dataId=%s: %s" % \
                    (dataRef.dataId, e,))
            if doRaise:
                result.append(self.run(refDataRef, dataRef))
            else:
                try:
                    result.append(self.run(refDataRef, dataRef))
                except Exception, e:
                    self.log.log(self.log.FATAL, "Failed on dataId=%s: %s" % (dataRef.dataId, e))
                    if not isinstance(e, pipeBase.TaskError):
                        traceback.print_exc(file=sys.stderr)
            try:
                metadataName = self._getMetadataName()
                if metadataName is not None:
                    dataRef.put(self.getFullMetadata(), metadataName)
            except Exception, e:
                self.log.log(self.log.WARN, "Could not persist metadata for dataId=%s: %s" % \
                    (dataRef.dataId, e,))
        return result

class MatchBackgroundsParser(pipeBase.ArgumentParser):
    """A version of lsst.pipe.base.ArgumentParser specialized for background matching
    """
    def __init__(self, *args, **kwargs):
        pipeBase.ArgumentParser.__init__(self, *args, **kwargs)
        self.add_argument("--refid", nargs="*", action=RefIdValueAction,
                          help="unique, full reference background data ID, e.g. --refid visit=865833781 raft=2,2 sensor=1,0 filter=3 patch=3 tract=77,69", metavar="KEY=VALUE") 

    def _makeDataRefList(self, namespace):
        """Make namespace.dataRefList from namespace.dataIdList *AND* namespace.refDataRefList from namespace.refDataIdList
        """
        datasetType = namespace.config.datasetType
        validKeys = namespace.butler.getKeys(datasetType=datasetType, level=self._dataRefLevel)

        # NOTE: namespace.dataIdList has had its types fixed upstream
        # in parse_args, so we have to do namespace.refDataIdList here
        for dataDict in namespace.refDataIdList:
            for key, strVal in dataDict.iteritems():
                try:
                    keyType = validKeys[key]
                except KeyError:
                    self.error("Unrecognized ID key %r; valid keys are: %s" % (key, sorted(validKeys.keys())))
                if keyType != str:
                    try:
                        castVal = keyType(strVal)
                    except Exception:
                        self.error("Cannot cast value %r to %s for ID key %r" % (strVal, keyType, key,))
                    dataDict[key] = castVal


        for refPair in (("dataRefList", "dataIdList"), ("refDataRefList", "refDataIdList")):
            refList, idList = refPair
            
            exec("namespace.%s = []" % (refList))
            for dataId in eval("namespace.%s" % (idList)):
                # tract and patch are required
                for key in validKeys:
                    if key not in dataId:
                        self.error("--id must include " + key)

                dataRef = namespace.butler.dataRef(
                    datasetType = datasetType,
                    dataId = dataId,
                )
                exec("namespace.%s.append(dataRef)" % (refList))

        # CHECK : is this sufficient?
        if len(namespace.refDataRefList) != 1:
            raise RuntimeError("Ref data to match to does not evaluate to a unique dataId")

class RefIdValueAction(argparse.Action):
    """argparse action callback to add one data ID dict to namespace.dataIdList
    """
    def __call__(self, parser, namespace, values, option_string):
        """Parse --refid data and store results in namespace.refDataIdList
        
        The data format is:
        key1=value1_1[^value1_2[^value1_3...] key2=value2_1[^value2_2[^value2_3...]...

        The values (e.g. value1_1) may either be a string, or of the form "int...int" (e.g. "1..3")
        which is interpreted as "1^2^3" (inclusive, unlike a python range). So "0^2..4^7..9" is
        equivalent to "0^2^3^4^7^8^9"
        
        """
        if namespace.config is None:
            return
        idDict = dict()
        for nameValue in values:
            name, sep, valueStr = nameValue.partition("=")
            idDict[name] = []
            for v in valueStr.split("^"):
                mat = re.search(r"^(\d+)\.\.(\d+)$", v)
                if mat:
                    v1 = int(mat.group(1))
                    v2 = int(mat.group(2))
                    for v in range(v1, v2 + 1):
                        idDict[name].append(str(v))
                else:
                    idDict[name].append(v)

        keyList = idDict.keys()
        iterList = [idDict[key] for key in keyList]
        idDictList = [dict(zip(keyList, valList)) for valList in itertools.product(*iterList)]

        namespace.refDataIdList = idDictList
