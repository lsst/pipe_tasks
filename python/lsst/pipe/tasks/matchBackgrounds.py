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
import numpy as np
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.geom as afwGeom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base.argumentParser import IdValueAction


class MatchBackgroundsConfig(pexConfig.Config):

    #Not sure if we want this one
    useDetectionBackground = pexConfig.Field(
        dtype = bool,
        doc = """True -  True uses lsst.meas.detection.getBackground()
                 False - masks, grids and fits Chebyshev polynomials to the backgrounds """,
        default = False
    )

    #Chebyshev polynomial fitting options
    backgroundOrder = pexConfig.Field(
        dtype = int,
        doc = """Order of background Chebyshev""",
        default = 4
    )

    backgroundBinsize = pexConfig.Field(
        dtype = int,
        doc = """Bin size for background matching""",
        default = 128
    )

    gridStat = pexConfig.ChoiceField(
        dtype = str,
        doc = """Type of statistic to use for the grid points""",
        default = "MEAN",
        allowed = {
            "MEAN": "mean",
            "MEDIAN": "median"
            }
    )

    #Misc options
    datasetType = pexConfig.Field(
        dtype = str,
        doc = """Name of data product to fetch (calexp, etc)""",
        default = "goodSeeingCoadd_tempExp" #needs tickets/2343
        #default = "coaddTempExp" #needs tickets/2317
    )


class MatchBackgroundsTask(pipeBase.CmdLineTask):
    ConfigClass = MatchBackgroundsConfig
    _DefaultName = "matchBackgrounds"
    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)

    @pipeBase.timeMethod
    def run(self, refDataRef, toMatchDataRef):
        """
        Match the background of the science exposure to a reference exposure

        @param refDataRef: data reference for the reference exposure
        @param toMatchDataRef: data reference for the science exposure
        @return: a pipBase.Struct with fields:
        - matchBackgroundModel: an afw.Math.Function object
        - matchedExopsure: an exposure with the background level equalized to the reference exp level
        """
        self.log.info("Matching background of %s to %s" % (toMatchDataRef.dataId, refDataRef.dataId))

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
        """
        Matches a science exposure's background level to that of a reference image.
        sciExposure's image is overwritten in memory, mask preserved

        Potential TO DOs:
            check to make sure they aren't the same image?
        """

        #Check that exps are the same shape. Return if they aren't
        if (sciExposure.getDimensions() != refExposure.getDimensions()):
            wSci, hSci = sciExposure.getDimensions()
            wRef, hRef = refExposure.getDimensions()
            self.log.log(self.log.WARN,
                         "Cannot Match. Exposures different shapes. sci:(%i, %i) vs. ref:(%i, %i)" %
                         (wSci,hSci,wRef,hRef))
            self.log.log(self.log.WARN, "Returning None")
            #Could return None, sciExposure
            #but it would have to be caught by the coadder
            return None, None

        mask  = np.copy(refExposure.getMaskedImage().getMask().getArray())
        mask += sciExposure.getMaskedImage().getMask().getArray()
        #get indicies of masked pixels
        #  Currently gets all non-zero-mask pixels,
        #  but this can be tweaked to get whatver types you want.
        ix,iy = np.where((mask) > 0)

        #make difference image array
        diffArr  = np.copy(refExposure.getMaskedImage().getImage().getArray())
        diffArr -= sciExposure.getMaskedImage().getImage().getArray()

        #set image array pixels to nan if masked
        diffArr[ix, iy] = np.nan

        #bin
        width, height  = refExposure.getDimensions()
        x0, y0 = refExposure.getXY0()
        xedges = np.arange(0, width, self.config.backgroundBinsize)
        yedges = np.arange(0, height, self.config.backgroundBinsize)
        xedges = np.hstack(( xedges, width ))  #add final edge
        yedges = np.hstack(( yedges, height )) #add final edge
        #Initialize lists to hold grid.
        #Use lists/append to protect against the case where
        #a bin has no valid pixels and should not be included in the fit
        bgX = []
        bgY = []
        bgZ = []
        bgdZ = []

        for ymin, ymax in zip(yedges[0:-1],yedges[1:]):
            for xmin, xmax in zip(xedges[0:-1],xedges[1:]):
                #print ymin, ymax, xmin,xmax
                area = diffArr[ymin:ymax][:,xmin:xmax]
                # if there are less than 2 pixels with non-nan,non-masked values:
                #TODO: np.where is expensive and perhaps
                #      we can do it once above the loop and just compare indices here
                idxNotNan = np.where(~np.isnan(area))
                if len(idxNotNan[0]) >= 2:
                    bgX.append(0.5 * (x0 + xmin + x0 + xmax))
                    bgY.append(0.5 * (y0 + ymin + y0 + ymax))
                    bgdZ.append( np.std(area[idxNotNan])
                                                /np.sqrt(np.size(area[idxNotNan])))
                    if self.config.gridStat == "MEDIAN":
                       bgZ.append(np.median(area[idxNotNan]))
                    elif self.config.gridStat == 'MEAN':
                       bgZ.append(np.mean(area[idxNotNan]))
                    else:
                        print "Unspecified grid statistic. Using mean"
                        bgZ.append(np.mean(area[idxNotNan]))

        #Check that there are enough points to fit
        if len(bgZ) == 0:
            self.log.warn("No overlap with reference. Cannot match")
            return None, None
        elif len(bgZ) == 1:
            #
            #TODO:make the offset constant = bgZ
            #
            self.log.warn("Only one  point. Const offset to be applied. Not yet implemented")
            return None, None
        else:
            #Fit grid with polynomial
            bbox  = afwGeom.Box2D(refExposure.getMaskedImage().getBBox(afwImage.PARENT))
            matchBackgroundModel = self.getChebFitPoly(bbox, self.config.backgroundOrder,
            										   bgX,bgY,bgZ,bgdZ)
            #refExposure.writeFits('refExposure.fits')
            #sciExposure.writeFits('sciExposure.fits')
            im  = sciExposure.getMaskedImage().getImage()
            #matches sciExposure in place in memory
            im +=  matchBackgroundModel
            #Variance of difference Image: var(A - B) = var(A) +var(B)
            # Should add some variance to sciExp after the background matching
            # Could use result of the polynomial fitting,
            # or just the diffImage variance: prob too high but for now until testing
            var  = sciExposure.getMaskedImage().getVariance()
            var += refExposure.getMaskedImage().getVariance()
            #sciExposure.writeFits('sciExposureMatched.fits')
            #
            #import pdb; pdb.set_trace()
            #To Do: Perform RMS check here to make sure new sciExposure is matched well enough?
            #

            print "Diff Image mean Var: ", (np.mean(var.getArray()[np.where(np.isfinite(var.getArray()))]))
            diffArr  = np.copy(refExposure.getMaskedImage().getImage().getArray())
            diffArr -= sciExposure.getMaskedImage().getImage().getArray()
            diffArr[ix, iy] = np.nan
            print "ref - matched Var : ", np.std(diffArr[np.where(~np.isnan(diffArr))])**2
            #returns the background Model, and the matched science exposure
        return matchBackgroundModel, sciExposure

    def getChebFitPoly(self, bbox, degree, X, Y, Z, dZ):
        """ Temporary function to be eventually replaced in afwMath and meas_alg
    	Fits a grid of points and returns a afw.math.Chebyshev1Function2D
    
        @param bbox lsst.afw.geom.Box2D (provides the allowed x,y range)
        @param degree order of polynomial (0 for constant)
        @param X list or array of x positions of grid points
        @param Y list or array of y positions of grid points
        @param Z list or array of the values to be fit
        @param dZ list or array of the error on values to be fit.
        @return an afw.math.Chebyshev1Function2D that fits the grid supplied
        """
        poly  = afwMath.Chebyshev1Function2D(int(degree), bbox)
        terms = list(poly.getParameters())
        Ncell = np.sum(np.isfinite(Z)) #number of bins to fit: usually nbinx*nbiny
        Nterm = len(terms)
        m  = np.zeros((Ncell, Nterm))
        b  = np.zeros((Ncell))
        iv = np.zeros((Ncell))
        nc = 0
        #Would be nice if the afwMath.ChebyshevFunction2D could make the matrix for fitting:
        #so that looping here wasn't necessary
        for na in range(Ncell):
            for nt in range(Nterm):
                terms[nt] = 1.0
                poly.setParameters(terms)
                m[nc, nt] = poly(X[na], Y[na])
                terms[nt] = 0.0
            b[nc]  = Z[na]
            iv[nc] = 1/(dZ[na]*dZ[na])
            nc += 1
        M    = np.dot(np.dot(m.T, np.diag(iv)), m)
        B    = np.dot(np.dot(m.T, np.diag(iv)), b)
        try:
            Soln = np.linalg.solve(M,B)
        except:
            self.log.warn("Polynomial fit FAILED. Returning all parameters = 0")
            return afwMath.Chebyshev1Function2D(int(degree), bbox)
        poly.setParameters(Soln)
        return poly

    @pipeBase.timeMethod
    def matchBackgroundsDetection(self, refExposure, sciExposure):
        """
        Placeholder for optional background matching method. To be called when
        self.config.useDetectionBackground
        Match backgrounds using meas.algorithms.detection.getBackground()
        """
        pass

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser
        """
        return MatchBackgroundsParser(name=cls._DefaultName,
                                      datasetType=cls.ConfigClass().datasetType)

    @classmethod
    def getRunInfo(cls, parsedCmd):
        log = parsedCmd.log if not pipeBase.cmdLineTask.useMultiProcessing(parsedCmd) else None# XXX pexLogging is not yet picklable
        refDataRef = parsedCmd.refDataRefList[0]
        inputs = [pipeBase.Struct(cls=cls, config=parsedCmd.config, log=log,
                                  doraise=parsedCmd.doraise, dataRef=dataRef,
                                  refDataRef=refDataRef)
                  for dataRef in parsedCmd.dataRefList]
        return pipeBase.Struct(func=runTask, inputs=inputs)

    def runDataRef(self, refDataRef, dataRef, doraise=False):
        """Execute the parsed command on a sequence of dataRefs,
        including writing the config and metadata.
        """
        try:
            configName = self._getConfigName()
            if configName is not None:
                dataRef.put(self.config, configName)
        except Exception, e:
            self.log.warn("Could not persist config for dataId=%s: %s" % \
                (dataRef.dataId, e,))
        if doraise:
            self.run(refDataRef, dataRef)
        else:
            try:
                self.run(refDataRef, dataRef)
            except Exception, e:
                self.log.fatal("Failed on dataId=%s: %s" % (dataRef.dataId, e))
                if not isinstance(e, pipeBase.TaskError):
                    traceback.print_exc(file=sys.stderr)
        try:
            metadataName = self._getMetadataName()
            if metadataName is not None:
                dataRef.put(self.getFullMetadata(), metadataName)
        except Exception, e:
            self.log.warn("Could not persist metadata for dataId=%s: %s" % \
                (dataRef.dataId, e,))

class MatchBackgroundsParser(pipeBase.ArgumentParser):
    """A version of lsst.pipe.base.ArgumentParser specialized for background matching
    """
    def __init__(self, *args, **kwargs):
        pipeBase.ArgumentParser.__init__(self, *args, **kwargs)
        self.add_argument("--refid", nargs="*", action=RefIdValueAction,
                          help="unique, full reference background data ID, e.g. --refid visit=865833781 filter=3 patch=3 tract=77,69", metavar="KEY=VALUE")

    def _makeDataRefList(self, namespace):
        """Make namespace.dataRefList from namespace.dataIdList *AND*
           namespace.refDataRefList from namespace.refDataIdList
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

def runTask(args):
    task = args.cls(name = args.cls._DefaultName, config=args.config, log=args.log)
    task.runDataRef(args.refDataRef, args.dataRef, doraise=args.doraise)
