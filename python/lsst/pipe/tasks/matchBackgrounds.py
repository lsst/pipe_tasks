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

import numpy
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.geom as afwGeom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsstDebug

class MatchBackgroundsConfig(pexConfig.Config):

    usePolynomial = pexConfig.Field(
        dtype = bool,
        doc = "Fit background difference with Chebychev polynomial interpolation " \
        "(using afw.math.Approximate)? If False, fit with spline interpolation using afw.math.Background",
        default = False
    )
    order = pexConfig.Field(
        dtype = int,
        doc = "Order of Chebyshev polynomial background model. Ignored if usePolynomial False",
        default = 8
    )
    badMaskPlanes = pexConfig.ListField(
        doc = "Names of mask planes to ignore while estimating the background",
        dtype = str, default = ["EDGE", "DETECTED", "DETECTED_NEGATIVE","SAT","BAD","INTRP","CR"],
        itemCheck = lambda x: x in afwImage.MaskU().getMaskPlaneDict().keys(),
    )
    gridStatistic = pexConfig.ChoiceField(
        dtype = str,
        doc = "Type of statistic to estimate pixel value for the grid points",
        default = "MEAN",
        allowed = {
            "MEAN": "mean",
            "MEDIAN": "median",
            "MEANCLIP": "clipped mean"
            }
    )
    undersampleStyle = pexConfig.ChoiceField(
        doc = "Behaviour if there are too few points in grid for requested interpolation style. " \
        "Note: INCREASE_NXNYSAMPLE only allowed for usePolynomial=True.",
        dtype = str,
        default = "REDUCE_INTERP_ORDER",
        allowed = {
            "THROW_EXCEPTION": "throw an exception if there are too few points",
            "REDUCE_INTERP_ORDER": "use an interpolation style with a lower order.",
            "INCREASE_NXNYSAMPLE": "Increase the number of samples used to make the interpolation grid.",
            }
    )
    binSize = pexConfig.Field(
        doc = "Bin size for gridding the difference image and fitting a spatial model",
        dtype=int,
        default=256
    )
    interpStyle = pexConfig.ChoiceField(
        dtype = str,
        doc = "Algorithm to interpolate the background values; ignored if usePolynomial is True" \
              "Maps to an enum; see afw.math.Background",
        default = "AKIMA_SPLINE",
        allowed={
             "CONSTANT" : "Use a single constant value",
             "LINEAR" : "Use linear interpolation",
             "NATURAL_SPLINE" : "cubic spline with zero second derivative at endpoints",
             "AKIMA_SPLINE": "higher-level nonlinear spline that is more robust to outliers",
             "NONE": "No background estimation is to be attempted",
             }
    )
    numSigmaClip = pexConfig.Field(
        dtype = int,
        doc = "Sigma for outlier rejection; ignored if gridStatistic != 'MEANCLIP'.",
        default = 3
    )
    numIter = pexConfig.Field(
        dtype = int,
        doc = "Number of iterations of outlier rejection; ignored if gridStatistic != 'MEANCLIP'.",
        default = 2
    )
    bestRefWeightCoverage = pexConfig.RangeField(
        dtype = float,
        doc = "Weight given to coverage (number of pixels that overlap with patch), " \
        "when calculating best reference exposure. Higher weight prefers exposures with high coverage." \
        "Ignored when reference visit is supplied",
        default = 0.4,
        min = 0., max = 1.
    )
    bestRefWeightVariance = pexConfig.RangeField(
        dtype = float,
        doc = "Weight given to image variance when calculating best reference exposure. " \
        "Higher weight prefers exposures with low image variance. Ignored when reference visit is supplied",
        default = 0.4,
        min = 0., max = 1.
    )
    bestRefWeightLevel = pexConfig.RangeField(
        dtype = float,
        doc = "Weight given to mean background level when calculating best reference exposure. " \
        "Higher weight prefers exposures with low mean background level. " \
        "Ignored when reference visit is supplied.",
        default = 0.2,
        min = 0., max = 1.
    )


class MatchBackgroundsTask(pipeBase.Task):
    ConfigClass = MatchBackgroundsConfig
    _DefaultName = "matchBackgrounds"
    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)

        self.sctrl = afwMath.StatisticsControl()
        self.sctrl.setAndMask(afwImage.MaskU.getPlaneBitMask(self.config.badMaskPlanes))
        self.sctrl.setNanSafe(True)

    @pipeBase.timeMethod
    def run(self, toMatchRefList, refVisitRef=None, tempExpName = None):
        """Match the backgrounds of a list of coadd temp exposures to a reference coadd temp exposure.

        Choose a refVisitRef automatically if none supplied.

        @param toMatchRefList: List containing data references to science exposures to be matched
        @param refVisitRef: Data reference for the reference exposure. Default = None.
        @param tempExpName: Name of the coadd temp exposures.  For example: 'goodSeeingCoadd_tempExp'
        or 'deepCoadd_tempExp'. Default = None.
        @return: a pipBase.Struct with fields:
        - backgroundModelList:  List of afw.Math.Background or afw.Math.Approximate objects containing the
                                model offset to be added to the original science exposures.
        - fitRMSList: List of RMS values of the fit. Describes how well the model fit the difference image.
                      It can be used to properly increase the variance plane of the image.
                      Currently, all values = 0.
        - isReference: List of bools -- all elements are False except for the one corresponding to the
                       reference visit.
        """

        numExp = len(toMatchRefList)
        if numExp < 1:
            raise pipeBase.TaskError("No exposures to match")

        if tempExpName is None:
            tempExpName = self.config.datasetType

        if refVisitRef is None:
            refVisitRef = self.getBestRefExposure(toMatchRefList, tempExpName)

        backgroundMatchResultList = []

        if not refVisitRef.datasetExists(tempExpName):
            raise pipeBase.TaskError("Reference data id %s does not exist" % (refVisitRef.dataId))
        refExposure = refVisitRef.get(tempExpName)

        keySet = set(toMatchRefList[0].dataId)
        visitKeySet = set(toMatchRefList[0].dataId) - set(['tract','patch'])

        self.log.info("Matching %d Exposures" % (numExp))

        for toMatchRef in toMatchRefList:
            self.log.info("Matching background of %s to %s" % (toMatchRef.dataId, refVisitRef.dataId))
            try:
                #store a string specifying the visit to label debug plot
                toMatchExposure = toMatchRef.get(tempExpName)
                self.debugVisitString = ''.join([str(toMatchRef.dataId[vk]) for vk in visitKeySet])
    	        backgroundInfoStruct = self.matchBackgrounds(refExposure, toMatchExposure)
            except Exception, e:
                #if match fails (e.g. low coverage), insert Nones as placeholders in output lists
                self.log.warn("Failed to fit background %s: %s" % (toMatchRef.dataId, e))
                backgroundInfoStruct = pipeBase.Struct(
                    matchBackgroundModel = None,
                    matchedExposure = None,
                    fitRMS = None,
                    matchedMSE = None,
                    diffImVar = None)

            if tuple(toMatchRef.dataId[k] for k in keySet) == tuple(refVisitRef.dataId[k] for k in keySet):
                isReference = True
            else:
                isReference = False

            #make new struct to return without the exposure
            toReturnBackgroundStruct = pipeBase.Struct(
                    backgroundModel = backgroundInfoStruct.matchBackgroundModel,
                    fitRMS = backgroundInfoStruct.fitRMS,
                    matchedMSE = backgroundInfoStruct.matchedMSE,
                    diffImVar = backgroundInfoStruct.diffImVar,
                    isReference = isReference)

            backgroundMatchResultList.append(toReturnBackgroundStruct)
            
        return pipeBase.Struct(
            backgroundModelStructList = backgroundMatchResultList)

    @pipeBase.timeMethod
    def getBestRefExposure(self, refList, tempExpName):
        """Return the dataRef of the best exposure to use as the reference exposure

        Calculate an appropriate reference exposure by minimizing a cost function that penalizes
        high variance,  high background level, and low coverage. Use the following config parameters:
        bestRefWeightCoverage: Float between 0 and 1. Weighting of coverage in the cost function.
        bestRefWeightVariance: Float between 0 and 1. Weight of the variance in the cost function.
        bestRefWeightLevel: Float between 0 and 1. Weight of background level in the cost function

        @param refList: List containing data references to exposures
        @param tempExpName: Name of the dataType of the coadd temp exposures: e.g. 'goodSeeingCoadd_tempExp'

        @return: a data reference pointing to the best reference exposure
        """
        self.log.info("Calculating best reference visit")
        varList =  []
        meanBkgdLevelList = []
        coverageList = []

        for ref in refList:
            if not ref.datasetExists(tempExpName):
                self.log.warn("Data id %s does not exist. Skipping for ref visit calculation" % (ref.dataId))
                continue
            tempExp = ref.get(tempExpName)
            maskedImage = tempExp.getMaskedImage()
            statObjIm = afwMath.makeStatistics(maskedImage.getImage(), maskedImage.getMask(),
                afwMath.MEAN | afwMath.NPOINT | afwMath.VARIANCE, self.sctrl)
            meanVar, meanVarErr = statObjIm.getResult(afwMath.VARIANCE)
            meanBkgdLevel, meanBkgdLevelErr = statObjIm.getResult(afwMath.MEAN)
            npoints, npointsErr = statObjIm.getResult(afwMath.NPOINT)
            varList.append(meanVar)
            meanBkgdLevelList.append(meanBkgdLevel)
            coverageList.append(npoints)

        if not coverageList:
             raise pipeBase.TaskError('No temp exposures of type %s found calculate best reference exposure'%
                               (tempExpName))

        # Normalize metrics to range from  0 to 1
        varArr = numpy.array(varList)/numpy.max(varList)
        meanBkgdLevelArr = numpy.array(meanBkgdLevelList)/numpy.max(meanBkgdLevelList)
        coverageArr = numpy.min(coverageList)/numpy.array(coverageList)

        costFunctionArr = self.config.bestRefWeightVariance  * varArr
        costFunctionArr += self.config.bestRefWeightLevel * meanBkgdLevelArr
        costFunctionArr += self.config.bestRefWeightCoverage * coverageArr
        return refList[numpy.argmin(costFunctionArr)]


    @pipeBase.timeMethod
    def matchBackgrounds(self, refExposure, sciExposure):
        """
        Match science exposure's background level to that of reference exposure.

        Process creates a difference image of the reference exposure minus the science exposure. It then
        generates an afw.math.Background object, and assumes that the mask plane already has detections set.
        The 'background' of is fit by spline interpolation or by polynomial interpolation by the Approximate
        class. This fit of difference image is added to the science exposure in memory.
        Fit diagnostics are also calculated and returned.

        @param refExposure: reference exposure (unaltered)
        @param sciExposure: science exposure (background level matched to that of reference exposure)
        @returns a pipBase.Struct with fields:
            -matchBackgroundModel: an afw.math.Approximate or an afw.math.Background
            -matchedExposure: new pointer to matched science exposure.
            -fitRMS: rms of the fit. This is the sqrt(mean(residuals**2)).
            -matchedMSE: the MSE of the reference and matched images: mean((refImage - matchedSciImage)**2)
             should be comparable to difference image's mean variance,
            -diffImVar: the mean variance of the difference image.
        """

        if lsstDebug.Info(__name__).savefits:
            refExposure.writeFits(lsstDebug.Info(__name__).figpath + 'refExposure.fits')
            sciExposure.writeFits(lsstDebug.Info(__name__).figpath + 'sciExposure.fits')

        # Check Configs for polynomials:
        if self.config.usePolynomial:
            x, y = sciExposure.getDimensions()
            shortSideLength = min(x, y)
            if shortSideLength < self.config.binSize:
                raise ValueError("%d = config.binSize > shorter dimension = %d" % (self.config.binSize,
                                                                                   shortSideLength))
            npoints = shortSideLength // self.config.binSize
            if shortSideLength % self.config.binSize != 0:
                npoints += 1

            if self.config.order > npoints - 1:
                raise ValueError("%d = config.order > npoints - 1 = %d" % (self.config.order, npoints - 1))

        # Check that exposures are same shape
        if (sciExposure.getDimensions() != refExposure.getDimensions()):
            wSci, hSci = sciExposure.getDimensions()
            wRef, hRef = refExposure.getDimensions()
            raise RuntimeError(
                "Exposures are different dimensions. sci:(%i, %i) vs. ref:(%i, %i)" %
                (wSci,hSci,wRef,hRef))

        statsFlag = getattr(afwMath, self.config.gridStatistic)
        self.sctrl.setNumSigmaClip(self.config.numSigmaClip)
        self.sctrl.setNumIter(self.config.numIter)

        im  = refExposure.getMaskedImage()
        diffMI = im.Factory(im,True)
        diffMI -= sciExposure.getMaskedImage()

        width = diffMI.getWidth()
        height = diffMI.getHeight()
        nx = width // self.config.binSize
        if width % self.config.binSize != 0:
            nx += 1
        ny = height // self.config.binSize
        if height % self.config.binSize != 0:
            ny += 1

        bctrl = afwMath.BackgroundControl(nx, ny, self.sctrl, statsFlag)
        bctrl.setUndersampleStyle(self.config.undersampleStyle)
        bctrl.setInterpStyle(self.config.interpStyle)

        bkgd = afwMath.makeBackground(diffMI, bctrl)

        # Will remove this block if Approximate can check these 3 things:
        # 1) Check that order/bin size make sense:
        # 2) Change binsize or order if underconstrained.
        # 3) Add some tiny Gaussian noise if the image is completely uniform
        if self.config.usePolynomial:
            order = self.config.order
            bgX, bgY, bgZ, bgdZ = self._gridImage(diffMI, self.config.binSize, statsFlag)
            minNumberGridPoints = min(len(set(bgX)),len(set(bgY)))
            if len(bgZ) == 0:
                raise ValueError("No overlap with reference. Cannot match")
            elif minNumberGridPoints <= self.config.order:
                #must either lower order or raise number of bins or throw exception
                if self.config.undersampleStyle == "THROW_EXCEPTION":
                    raise ValueError("Image does not cover enough of ref image for order and binsize")
                elif self.config.undersampleStyle == "REDUCE_INTERP_ORDER":
                    self.log.warn("Reducing order to %d"%(minNumberGridPoints - 1))
                    order = minNumberGridPoints - 1
                elif self.config.undersampleStyle == "INCREASE_NXNYSAMPLE":
                    newBinSize = (minNumberGridPoints*self.config.binSize)// (self.config.order +1)
                    bctrl.setNxSample(newBinSize)
                    bctrl.setNySample(newBinSize)
                    bkgd = afwMath.makeBackground(diffMI, bctrl) #do over
                    self.log.warn("Decreasing binsize to %d"%(newBinSize))

            if not any(dZ > 1e-8 for dZ in bgdZ) and not any(bgZ): #uniform image
                gaussianNoiseIm = afwImage.ImageF(diffMI.getImage(), True)
                afwMath.randomGaussianImage(gaussianNoiseIm, afwMath.Random(1))
                gaussianNoiseIm *= 1e-8
                diffMI += gaussianNoiseIm
                bkgd = afwMath.makeBackground(diffMI, bctrl)

        #Add offset to sciExposure
        try:
            if self.config.usePolynomial:
                actrl = afwMath.ApproximateControl(afwMath.ApproximateControl.CHEBYSHEV,
                                                   order,
                                                   order)
                undersampleStyle = getattr(afwMath, self.config.undersampleStyle)
                approx = bkgd.getApproximate(actrl,undersampleStyle)
                bkgdImage = approx.getImage()
            else:
                bkgdImage = bkgd.getImageF()
        except Exception, e:
            raise RuntimeError("Background/Approximation failed to interp image %s: %s" % (
                self.debugVisitString, e))

       	sciMIPointer  = sciExposure.getMaskedImage()
        sciMIPointer += bkgdImage

        if lsstDebug.Info(__name__).savefits:
            sciExposure.writeFits(lsstDebug.Info(__name__).figpath + 'sciMatchedExposure.fits')

        if lsstDebug.Info(__name__).savefig:
            bbox  = afwGeom.Box2D(refExposure.getMaskedImage().getBBox(afwImage.PARENT))
            X, Y, Z, dZ = self._gridImage(diffMI, self.config.binSize, statsFlag)
            x0, y0 = diffMI.getXY0()
            Xshift = [int(x - x0) for x in X] # get positions in local image coords
            Yshift = [int(y - y0) for y in Y]
            modelValueArr = numpy.empty(len(Z))
            for i in range(len(X)):
                modelValueArr[i] = bkgdImage.get(int(Xshift[i]),int(Yshift[i]))
            resids = Z - modelValueArr
            try:
            	self._debugPlot(X, Y, Z, dZ, bkgdImage, bbox, modelValueArr, resids)
            except Exception, e:
                self.log.warn('Debug plot not generated: %s'%(e))

        stats = afwMath.makeStatistics(diffMI.getVariance(),diffMI.getMask(),afwMath.MEAN, self.sctrl)
        meanVar, _ = stats.getResult(afwMath.MEAN)

        diffImPointer  = diffMI.getImage()
        diffImPointer -= bkgdImage #This changes the image inside diffMI: should now have a mean ~ 0

        stats = afwMath.makeStatistics(diffMI, afwMath.MEANSQUARE | afwMath.VARIANCE, self.sctrl)
        mse, _ =  stats.getResult(afwMath.MEANSQUARE)

        outBkgd =  approx if self.config.usePolynomial else bkgd

        rms = 0.0  #place holder for an error on the fit to add to the matchedImage
        return pipeBase.Struct(
             matchBackgroundModel = outBkgd,
             matchedExposure = sciExposure,
             fitRMS = rms,
             matchedMSE = mse,
             diffImVar = meanVar)

    def _debugPlot(self, X, Y, Z, dZ, modelImage, bbox, model, resids):
        """Generate a plot showing the background fit and residuals.

        It is called when lsstDebug.Info(__name__).savefig = True
        Saves the fig to lsstDebug.Info(__name__).figpath
        Displays on screen if lsstDebug.Info(__name__).display = True

        @param X: list or array of x positions
        @param Y: list or array of y positions
        @param Z: list or array of the grid values that were interpolated
        @param dZ: list or array of the error on the grid values
        @param modelImage: image ofthe model of the fit
        @param model: list of len(Z) containing the grid values predicted by the model
        @param resids: Z - model
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors
        from mpl_toolkits.axes_grid1 import ImageGrid
        zeroIm = afwImage.MaskedImageF(afwGeom.Box2I(bbox))
        zeroIm += modelImage
        x0, y0 = zeroIm.getXY0()
        dx, dy = zeroIm.getDimensions()
        if len(Z) == 0:
            self.log.warn("No grid. Skipping plot generation.")
        else:
            max, min  = numpy.max(numpy.array(Z)), numpy.min(numpy.array(Z))
            norm = matplotlib.colors.normalize(vmax=max, vmin= min)
            maxdiff = numpy.max(numpy.abs(resids))
            diffnorm = matplotlib.colors.normalize(vmax=maxdiff, vmin= -maxdiff)
            rms = numpy.sqrt(numpy.mean(resids**2))
            from mpl_toolkits.axes_grid1 import ImageGrid
            fig = plt.figure(1, (8, 6))
            dz = numpy.array(dZ)
            meanDz = numpy.mean(dZ)
            grid = ImageGrid(fig, 111, nrows_ncols = (1, 2), axes_pad=0.1,
                             share_all=True, label_mode = "L", cbar_mode = "each",
                             cbar_size = "7%", cbar_pad="2%", cbar_location = "top")
            im = grid[0].imshow(zeroIm.getImage().getArray(),
                                extent=(x0, x0+dx, y0+dy, y0), norm = norm,
                                cmap='Spectral')
            im = grid[0].scatter(X, Y, c=Z, s = 15.*meanDz/dz, edgecolor='none', norm=norm,
                                 marker='o',cmap='Spectral')
            im2 = grid[1].scatter(X,Y,  c=resids, edgecolor='none', norm=diffnorm,
                                  marker='s', cmap='seismic')
            grid.cbar_axes[0].colorbar(im)
            grid.cbar_axes[1].colorbar(im2)
            grid[0].axis([x0, x0+dx, y0+dy, y0])
            grid[1].axis([x0, x0+dx, y0+dy, y0])
            grid[0].set_xlabel("model and grid")
            grid[1].set_xlabel("residuals. rms = %0.3f"%(rms))
            if lsstDebug.Info(__name__).savefig:
                fig.savefig(lsstDebug.Info(__name__).figpath + self.debugVisitString + '.png')
            if lsstDebug.Info(__name__).display:
                plt.show()
            plt.clf()

    def _gridImage(self, maskedImage, binsize, statsFlag):
        """Private method to grid an image for debugging"""
        width, height  = maskedImage.getDimensions()
        x0, y0 = maskedImage.getXY0()
        xedges = numpy.arange(0, width, binsize)
        yedges = numpy.arange(0, height, binsize)
        xedges = numpy.hstack(( xedges, width ))  #add final edge
        yedges = numpy.hstack(( yedges, height )) #add final edge

        # Use lists/append to protect against the case where
        # a bin has no valid pixels and should not be included in the fit
        bgX = []
        bgY = []
        bgZ = []
        bgdZ = []

        for ymin, ymax in zip(yedges[0:-1],yedges[1:]):
            for xmin, xmax in zip(xedges[0:-1],xedges[1:]):
                subBBox = afwGeom.Box2I(afwGeom.PointI(int(x0 + xmin),int(y0 + ymin)),
                                        afwGeom.PointI(int(x0 + xmax-1),int(y0 + ymax-1)))
                subIm = afwImage.MaskedImageF(maskedImage, subBBox, afwImage.PARENT, False)
                stats = afwMath.makeStatistics(subIm,
                                               afwMath.MEAN|afwMath.MEANCLIP|afwMath.MEDIAN| \
                                               afwMath.NPOINT|afwMath.STDEV,
                                               self.sctrl)
                npoints, _ = stats.getResult(afwMath.NPOINT)
                if npoints >= 2:
                    stdev, _ = stats.getResult(afwMath.STDEV)
                    if stdev < 1e-8:
                        #Zero variance. Set to some low but reasonable value
                        stdev = 1e-8
                    bgX.append(0.5 * (x0 + xmin + x0 + xmax))
                    bgY.append(0.5 * (y0 + ymin + y0 + ymax))
                    bgdZ.append(stdev/numpy.sqrt(npoints))
                    est, _ = stats.getResult(statsFlag)
                    bgZ.append(est)

        return bgX, bgY, bgZ, bgdZ
