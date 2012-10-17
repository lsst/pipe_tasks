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
import lsstDebug
from lsst.pipe.base.argumentParser import IdValueAction

class MatchBackgroundsConfig(pexConfig.Config):

    datasetType = pexConfig.Field(
        dtype = str,
        doc = """Name of data product to fetch (goodSeeingCoadd_tempExp, deepCoadd_tempExp etc)""",
        default = "goodSeeingCoadd_tempExp"
        #default = "coaddTempExp" #needs tickets/2317
        )
    
    useDetectionBackground = pexConfig.Field(
        dtype = bool,
        doc = """True -  True uses lsst.meas.algorithms.detection.getBackground()
                 False - masks, grids and fits Chebyshev polynomials to the backgrounds """,
        default = False
        )

    #Common Options
    ignoredPixelMask = pexConfig.ListField(
        doc="Names of mask planes to ignore while estimating the background",
        dtype=str, default = ["EDGE", "DETECTED", "DETECTED_NEGATIVE","SAT","BAD","INTRP","CR"],
        itemCheck = lambda x: x in afwImage.MaskU().getMaskPlaneDict().keys(),
        )
        
    gridStatistic = pexConfig.ChoiceField(
        dtype = str,
        doc = """Type of statistic to use for the grid points""",
        default = "MEAN",
        allowed = {
            "MEAN": "mean",
            "MEDIAN": "median",
            "CLIPMEAN": "clipped mean"
            }
        )

    undersampleStyle = pexConfig.ChoiceField(
        doc="behaviour if there are too few points in grid for requested interpolation style when matching",
        dtype=str, default="INCREASE_NXNYSAMPLE",
        allowed={
            "THROW_EXCEPTION": "throw an exception if there are too few points",
            "REDUCE_INTERP_ORDER": "use an interpolation style with a lower order.",
            "INCREASE_NXNYSAMPLE": "Increase the number of samples used to make the interpolation grid.",
            }
        )
     #Spline Options   
    splineBinSize = pexConfig.RangeField(
        doc="how large a region of the sky should be used for each background point when matching",
        dtype=int, default=256,
        min=10
        )
    
    algorithm = pexConfig.ChoiceField(
        dtype = str,
        doc = "how to interpolate the background values. This maps to an enum; see afw::math::Background",
        default = "AKIMA_SPLINE",
        allowed={
             "CONSTANT" : "Use a single constant value",
             "LINEAR" : "Use linear interpolation",
             "NATURAL_SPLINE" : "cubic spline with zero second derivative at endpoints",
             "AKIMA_SPLINE": "higher-level nonlinear spline that is more robust to outliers",
             "NONE": "No background estimation is to be attempted",
             }
        )
    
    #Polynomial fitting options
    backgroundOrder = pexConfig.Field(
        dtype = int,
        doc = """Order of background Chebyshev""",
        default = 4
        )

    chebBinSize = pexConfig.Field(
        dtype = int,
        doc = """Bin size for background matching. Can be arbitrarily small becase backgroundOrder sets the order""",
        default = 128
        )

    

class MatchBackgroundsTask(pipeBase.Task):
    ConfigClass = MatchBackgroundsConfig
    _DefaultName = "matchBackgrounds"
    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)

    @pipeBase.timeMethod
    def run(self, refDataRef, toMatchDataRefList):
        """
        Match the background of the science exposure to a reference exposure

        @param refDataRef: data reference for the reference exposure
        @param toMatchDataRef: data reference for the science exposure
        @return: a pipBase.Struct with fields:
        - matchBackgroundModel: an afw.Math.Function or a afw.Math.background
        - matchedExopsure: an exposure with the background level equalized to the reference exp level

        TO DO:
        This should be a subtask called by assembleCoadd.py
        It will take in a reference dataRef and a list of science dataRefs
        and return a list of background objects. 
        """
        self.log.info("Matching background of %s to %s" % (toMatchDataRef.dataId, refDataRef.dataId))

        if not refDataRef.datasetExists(self.config.datasetType):
            raise pipeBase.TaskError("Data id %s does not exist" % (refDataRef.dataId))
        refExposure = refDataRef.get(self.config.datasetType)


        if not toMatchDataRef.datasetExists(self.config.datasetType):
            raise pipeBase.TaskError("Data id %s does not exist" % (toMatchDataRef.dataId))
        sciExposure = toMatchDataRef.get(self.config.datasetType)

        if lsstDebug.Info(__name__).savefig:
            refExposure.writeFits(lsstDebug.Info(__name__).figpath + 'refExposure.fits')
            sciExposure.writeFits(lsstDebug.Info(__name__).figpath + 'sciExposure.fits')
            
        if self.config.useDetectionBackground:
            matchBackgroundModel, matchedExposure, matchedMSE, diffImVar = self.matchBackgroundsDetection(refExposure, sciExposure)
        else:
            matchBackgroundModel, matchedExposure, matchedMSE, diffImVar = self.matchBackgrounds(refExposure, sciExposure)

        if lsstDebug.Info(__name__).savefig:
            refExposure.writeFits(lsstDebug.Info(__name__).figpath + 'sciMatchedExposure.fits')
    
        sciExposure = toMatchDataRef.get(self.config.datasetType)
        return pipeBase.Struct(
            matchBackgroundModel = matchBackgroundModel,
            matchedExposure = matchedExposure
        )

    @pipeBase.timeMethod
    def matchBackgrounds(self, refExposure, sciExposure):
        """
        Matches a science exposure's background level to that of a reference image.
        sciExposure's image is overwritten in memory, mask preserved
        @param refExposure
        @param sciExposure
        @returns an afw::math::Chebyshev1Function2 modeling the difference (refExposure - sciExposure)

        Potential TO DOs:
            check to make sure they aren't the same image?
        """

        #Check Configs
        x,y = sciExposure.getDimensions()
        shortSide = min(x,y)
        #Check that short side >= binsize
        if shortSide < self.config.chebBinSize:
            raise ValueError("%d = config.chebBinSize > shorter dimension = %d"%(self.config.chebBinSize, shortSide))
        #Check that Order + 1 <= npoints
        npoints = shortSide//self.config.chebBinSize
        if shortSide % self.config.chebBinSize != 0:
            npoints += 1
        if self.config.backgroundOrder > npoints -1:
            raise ValueError("%d = config.backgroundOrder > npoints - 1 = %d"%(self.config.backgroundOrder, npoints - 1))

        #Check that exps are the same shape. Throw Exception if they aren't
        if (sciExposure.getDimensions() != refExposure.getDimensions()):
            wSci, hSci = sciExposure.getDimensions()
            wRef, hRef = refExposure.getDimensions()
            raise RuntimeError(
                "Exposures different dimensions. sci:(%i, %i) vs. ref:(%i, %i)" %
                (wSci,hSci,wRef,hRef))

        #deep copy
        diffIm = afwImage.MaskedImageF(refExposure.getMaskedImage(),True)
        diffIm -= sciExposure.getMaskedImage()

        sctrl = afwMath.StatisticsControl()
        sctrl.setAndMask(afwImage.MaskU.getPlaneBitMask(self.config.ignoredPixelMask))
        sctrl.setNanSafe(True)

        if self.config.gridStatistic == "MEDIAN":
            statsFlag = afwMath.MEDIAN
        elif self.config.gridStatistic == "CLIPMEAN":
            statsFlag = afwMath.CLIPMEAN
        elif self.config.gridStatistic == "MEAN":
            statsFlag = afwMath.MEAN

        #local copy of order incase it changes below:
        order = self.config.backgroundOrder     
        bbox  = afwGeom.Box2D(refExposure.getMaskedImage().getBBox(afwImage.PARENT))    
        #bgZ is a list of the diffImage background estimates per bin, bgdZ contains stdev/sqrt(N)    
        bgX, bgY, bgZ, bgdZ = self.gridImage(diffIm, self.config.chebBinSize, sctrl, statsFlag)
        minNumberGridPoints = min(len(set(bgX)),len(set(bgY)))
        
        #Check that there are enough points to fit
        if len(bgZ) == 0:
            raise ValueError("No overlap with reference. Cannot match")
        elif minNumberGridPoints <= self.config.backgroundOrder:
            #must either lower order or raise number of bins or throw exception
            if self.config.undersampleStyle == "THROW_EXCEPTION":
                raise ValueError("Image does not cover enough of ref image for order and binsize")
            elif self.config.undersampleStyle == "REDUCE_INTERP_ORDER":
                self.log.warn("Reducing order to %d"%(minNumberGridPoints - 1))
                order = minNumberGridPoints - 1
            elif self.config.undersampleStyle == "INCREASE_NXNYSAMPLE":
                newBinSize = (minNumberGridPoints*self.config.chebBinSize)// (self.config.backgroundOrder +1)
                bgX, bgY, bgZ, bgdZ = self.gridImage(diffIm, newBinSize, sctrl, statsFlag)
                self.log.warn("Decreasing binsize to %d"%(newBinSize))
                
    
        #Fit grid with polynomial
        matchBackgroundModel, resids, rms = self.getChebFitPoly(bbox, order,
                                                   bgX,bgY,bgZ,bgdZ)

        im  = sciExposure.getMaskedImage().getImage()
        #matches sciExposure in place in memory
        im +=  matchBackgroundModel
        
        #Add RMS from fit sciExp's variance after the background matching
        if lsstDebug.Info(__name__).savefig:
            sciExposure.writeFits(lsstDebug.Info(__name__).figpath + 'sciExposureMatched.fits')
        var = sciExposure.getMaskedImage().getVariance()
        var += rms**2

        #Check fit! Compare MSE with mean variance of the difference Image
        stats = afwMath.makeStatistics(diffIm.getVariance(),diffIm.getMask(),afwMath.MEAN, sctrl)
        meanVar, _ = stats.getResult(afwMath.MEAN)
        #print "Diff Image mean Var: ", meanVar
        dim = diffIm.getImage()
        dim -= matchBackgroundModel
        stats = afwMath.makeStatistics(diffIm, afwMath.MEANSQUARE | afwMath.VARIANCE, sctrl)
        MSE, _ =  stats.getResult(afwMath.MEANSQUARE)
        #print "ref - matched Var : ", MSE
        return matchBackgroundModel, sciExposure, MSE, meanVar

    def gridImage(self, maskedImage, binsize, sctrl, statsFlag):
        #bin
        width, height  = maskedImage.getDimensions()
        x0, y0 = maskedImage.getXY0()
        xedges = np.arange(0, width, binsize)
        yedges = np.arange(0, height, binsize)
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
                subBBox = afwGeom.Box2I(afwGeom.PointI(int(x0 + xmin),int(y0 + ymin)),afwGeom.PointI(int(x0 + xmax-1),int(y0 + ymax-1)))
                subIm = afwImage.MaskedImageF(maskedImage, subBBox, afwImage.PARENT, False)
                stats = afwMath.makeStatistics(subIm,afwMath.MEAN|afwMath.NPOINT|afwMath.STDEV|afwMath.MEDIAN,sctrl)
                # if there are less than 2 pixels with non-nan,non-masked values:
                npoints, _ = stats.getResult(afwMath.NPOINT)
                if npoints >= 2:
                    stdev, _ = stats.getResult(afwMath.STDEV)
                    if stdev < 1e-8:
                        #Zero variance. Unrealistic. #Set to some low but reasonable value
                        self.log.warn("changing stdev of bin from %.03e to non-zero 1e-8" % (stdev))
                        stdev = 1e-8
                    bgX.append(0.5 * (x0 + xmin + x0 + xmax))
                    bgY.append(0.5 * (y0 + ymin + y0 + ymax))
                    bgdZ.append(stdev/np.sqrt(npoints))
                    est, _ = stats.getResult(statsFlag)
                    bgZ.append(est)

        return bgX, bgY, bgZ, bgdZ
        

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
        except Exception, e:
            raise RuntimeError("Failed to fit background: %s" % e)
        poly.setParameters(Soln)
        vPoly = np.vectorize(poly)
        model = vPoly(X,Y)
        resids = Z - model
        RMS = np.sqrt(np.mean(resids**2))
        if lsstDebug.Info(__name__).display:
            self.debugPlot(X,Y,Z,dZ,poly, bbox, model, resids)
        return poly, resids, RMS

    def debugPlot(self,X,Y,Z,dZ,poly, bbox, model,resids):
        """ Will generate a plot showing the background fit and residuals.
        It is called when lsstDebug.Info(__name__).display = True
        Saves the fig to lsstDebug.Info(__name__).figpath if
        lsstDebug.Info(__name__).savefig = True
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors
        from mpl_toolkits.axes_grid1 import ImageGrid
        Zeroim = afwImage.MaskedImageF(afwGeom.Box2I(bbox))
        Zeroim += poly
        x0, y0 = Zeroim.getXY0()
        dx, dy = Zeroim.getDimensions()
        max, min  = np.max(np.array(Z)), np.min(np.array(Z))
        norm = matplotlib.colors.normalize(vmax=max, vmin= min)
        maxdiff = np.max(np.abs(resids))
        diffnorm = matplotlib.colors.normalize(vmax=maxdiff, vmin= -maxdiff)
        rms = np.sqrt(np.mean(resids**2))
        from mpl_toolkits.axes_grid1 import ImageGrid
        fig = plt.figure(1, (8, 6))
        dz = np.array(dZ)
        grid = ImageGrid(fig, 111, nrows_ncols = (1, 2), axes_pad=0.1,
                         share_all=True, label_mode = "L", cbar_mode = "each",
                         cbar_size = "7%", cbar_pad="2%", cbar_location = "top")
        im = grid[0].imshow(Zeroim.getImage().getArray(),
                            extent=(x0, x0+dx, y0+dy, y0), norm = norm,
                            cmap='Spectral')
        im = grid[0].scatter(X, Y, c=Z, s = 1/dz/10, edgecolor='none', norm=norm,
                             marker='o',cmap='Spectral')
        im2 = grid[1].scatter(X,Y,  c=resids, edgecolor='none', norm=diffnorm,
                              marker='s', cmap='seismic')
        grid.cbar_axes[0].colorbar(im)
        grid.cbar_axes[1].colorbar(im2)
        grid[0].axis([x0,x0+dx,y0+dy,y0])
        grid[1].axis([x0,x0+dx,y0+dy,y0])
        grid[0].set_xlabel("model and grid")
        grid[1].set_xlabel("residuals. rms = %0.3f"%(rms))
        if lsstDebug.Info(__name__).savefig:
            fig.savefig(lsstDebug.Info(__name__).figpath + 'debug.png')
        plt.show()
        #plt.clf()
       
    @pipeBase.timeMethod
    def matchBackgroundsDetection(self, refExposure, sciExposure):
        """
        Match sciExposure's background level to that of refExposure
             using meas.algorithms.detection.getBackground()
        @param refExposure
        @param sciExposure
        @returns an lsst::afw::math::Background object containing
        the model of the background of the difference image (refExposure - sciExposures)
        """
        import lsst.meas.algorithms.detection as detection
        config = detection.BackgroundConfig()
        config.isNanSafe = True
        config.binSize = self.config.splineBinSize
        config.ignoredPixelMask = self.config.ignoredPixelMask
        config.algorithm = self.config.algorithm
        config.undersampleStyle = self.config.undersampleStyle

        im  = refExposure.getMaskedImage()
        diff = im.Factory(im,True) 
        diff -= sciExposure.getMaskedImage()
        try:
            bkgd = detection.getBackground(diff, config)
        except Exception, e:
            raise RuntimeError("Failed to fit background: %s" % (e))

        #Add offset to sciExposure
        sci  = sciExposure.getMaskedImage()
        sci += bkgd.getImageF()
        
        if self.config.gridStatistic == "MEDIAN":
            statsFlag = afwMath.MEDIAN
        elif self.config.gridStatistic == "MEAN":
            statsFlag = afwMath.MEAN
        elif self.config.gridStatistic == "CLIPMEAN":    
            statsFlag = afwMath.CLIPMEAN
            
        #Check fit! Compare MSE with mean variance of the difference Image
        sctrl = afwMath.StatisticsControl()
        sctrl.setAndMask(afwImage.MaskU.getPlaneBitMask(self.config.ignoredPixelMask))
        sctrl.setNanSafe(True)
            
        if lsstDebug.Info(__name__).display:
            bbox  = afwGeom.Box2D(refExposure.getMaskedImage().getBBox(afwImage.PARENT))
            X, Y, Z, dZ = self.gridImage(diff, self.config.splineBinSize, sctrl, statsFlag)
            x0, y0 = diff.getXY0()
            Xshift = [int(x - x0) for x in X]
            Yshift = [int(y - y0) for y in Y]
            model = np.empty(len(Z))
            for i in range(len(X)):
                model[i] = bkgd.getPixel(int(Xshift[i]),int(Yshift[i]))
            resids = Z - model             
            #vGetModel = np.vectorize(bkgd.getPixel)
            #model = vGetModel(X,Y) # TypeError: in method 'Background_getPixel', argument 2 of type 'int'
            self.debugPlot(X,Y,Z,dZ, bkgd.getImageF(), bbox, model, resids)
            
        stats = afwMath.makeStatistics(diff.getVariance(),diff.getMask(),afwMath.MEAN, sctrl) 
        meanVar, _ = stats.getResult(afwMath.MEAN)
        #print "Diff Image mean Var: ", meanVar
        dim = diff.getImage()
        dim -= bkgd.getImageF()
        stats = afwMath.makeStatistics(diff, afwMath.MEANSQUARE |afwMath.VARIANCE, sctrl)
        MSE, _ =  stats.getResult(afwMath.MEANSQUARE)
        #print "ref - matched Var : ", MSE
        return bkgd, sciExposure, MSE, meanVar


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
