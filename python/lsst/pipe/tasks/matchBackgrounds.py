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
import lsst.coadd.utils as coaddUtils
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
    def run(self, expRefList, imageScalerList=None, refExpDataRef=None, expDatasetType = None, skyInfo=None):
        """Match the backgrounds of a list of coadd temp exposures to a reference coadd temp exposure.

        Choose a refExpDataRef automatically if none supplied.
        
        @warning: all exposures must exist on disk

        @param expRefList: list of data references to exposures to be background-matched;
            all exposures must exist.
        @param imageScalerList: list of image scalers (coaddUtils.ImageScaler);
            if None then the images are not scaled
        @param refExpDataRef: data reference for the reference exposure.
            If None, then this task selects the best exposures from expRefList.
            if not None then must be one of the exposures in expRefList.
        @param expDatasetType: dataset type of exposures, e.g. 'goodSeeingCoadd_tempExp'
        or 'deepCoadd_tempExp'. Default = None.
        @return: a pipBase.Struct containing these fields:
        - backgroundModelStructList: a list of pipeBase.Struct, one per exposure in expRefList,
            each of which contains these fields:
            - isReference: this is the reference exposure (only one returned Struct will
                contain True for this value, unless the ref exposure is listed multiple times)
            - backgroundModel: differential background model (afw.Math.Background or afw.Math.Approximate).
                Add this to the science exposure to match the reference exposure.
            - fitRMS: list of RMS values of the fit. Describes how well the model fit the difference image.
                It can be used to properly increase the variance plane of the image. Currently always 0.
            - matchedMSE: ???
            - diffImVar: ???
            All fields except isReference will be None if isReference True or the fit failed.
        """
        numExp = len(expRefList)
        if numExp < 1:
            raise pipeBase.TaskError("No exposures to match")

        if expDatasetType is None:
            expDatasetType = self.config.datasetType

        if imageScalerList is None:
            self.log.info("imageScalerList is None; no photometric zero point scaling will be performed")
            imageScalerList = [coaddUtils.ImageScaler(1.0)] * numExp
        
        if len(expRefList) != len(imageScalerList):
            raise RuntimeError("len(expRefList) = %s != %s = len(imageScalerList)" % \
                (len(expRefList), len(imageScalerList)))

        # compute:
        # - refInd: index of reference image in expDataList
        # - refIndList: index of all exposures in expDataList that match the reference
        # The latter is used to avoid matching an exposure to itself in the unlikely case
        # that some exposures are duplicated in expDataList.
        expKeyList = refDataRef.butlerSubset.butler.getKeys(expDatasetType)
        idMatcher = IdMatcher(refDataRef.butlerSubset.butler.getKeys(expDatasetType))
        if refExpDataRef is None:
            # select the best reference exposure from expRefList
            refInd = self.selectRefExposure(
                expRefList = expRefList,
                imageScalerList = imageScalerList,
                expDatasetType = expDatasetType,
            )
            refIndList = idMatcher.matchList(id0 = expIdList[refInd].dataId, idList = expIdList)
            if refInd not in refIndList:
                raise RuntimeError("Internal error: selected reference %s not found in expRefList")
        else:
            # user specified the reference exposure; make sure it exists in expRefList
            expIdList = [expRef.dataId for expRef in expRefList]
            refIndList = idMatcher.matchList(id0 = refDataRef.dataId, idList = expIdList)
            if not refIndList:
                raise pipeBase.TaskError("Reference exposure %s is not one of the exposures in expRefList" % \
                    (refExpDataRef.dataId,))
            refInd = refIndList[0]

        refExpDataRef = expRefList[refInd]
        refExposure = refExpDataRef.get(expDatasetType, immediate=True)
        refMI = refExposure.getMaskedImage()
        refImageScaler = imageScalerList[refInd]
        refImageScaler.scaleMaskedImage(refMI)

        debugIdKeyList = tuple(set(expKeyList) - set(['tract','patch']))

        self.log.info("Matching %d Exposures" % (numExp))

        backgroundMatchResultList = []
        for ind, toMatchRef in enumerate(expRefList):
            if ind in refIndList:
                toReturnBackgroundStruct = pipeBase.Struct(
                    backgroundModel = None,
                    fitRMS = None,
                    matchedMSE = None,
                    diffImVar = None,
                    isReference = True,
                )
            else:
                imageScaler = imageScalerList[ind]
                self.log.info("Matching background of %s to %s" % (toMatchRef.dataId, refExpDataRef.dataId))
                try:
                    toMatchExposure = toMatchRef.get(expDatasetType, immediate=True)
                    toMatchMI = toMatchExposure.getMaskedImage()
                    imageScaler.scaleMaskedImage(toMatchMI)
                    #store a string specifying the visit to label debug plot
                    self.debugDataIdString = ''.join([str(toMatchRef.dataId[vk]) for vk in debugIdKeyList])
                    backgroundInfoStruct = self.matchBackgrounds(
                        refExposure = refExposure,
                        sciExposure = toMatchExposure,
                    )
                except Exception, e:
                    #if match fails (e.g. low coverage), insert Nones as placeholders in output lists
                    self.log.warn("Failed to fit background %s: %s" % (toMatchRef.dataId, e))
                    backgroundInfoStruct = pipeBase.Struct(
                        matchBackgroundModel = None,
                        matchedExposure = None,
                        fitRMS = None,
                        matchedMSE = None,
                        diffImVar = None,
                        isReference = False,
                    )

                #make new struct to return without the exposure
                toReturnBackgroundStruct = pipeBase.Struct(
                    isReference = False,
                    backgroundModel = backgroundInfoStruct.matchBackgroundModel,
                    fitRMS = backgroundInfoStruct.fitRMS,
                    matchedMSE = backgroundInfoStruct.matchedMSE,
                    diffImVar = backgroundInfoStruct.diffImVar,
                )
    
                backgroundMatchResultList.append(toReturnBackgroundStruct)
            
        return pipeBase.Struct(
            backgroundModelStructList = backgroundMatchResultList)
        
    @pipeBase.timeMethod
    def selectRefExposure(self, expRefList, imageScalerList, expDatasetType):
        """Find best exposure to use as the reference exposure

        Calculate an appropriate reference exposure by minimizing a cost function that penalizes
        high variance,  high background level, and low coverage. Use the following config parameters:
        - bestRefWeightCoverage
        - bestRefWeightVariance
        - bestRefWeightLevel

        @param[in] expRefList: list of data references to exposures.
            Retrieves dataset type specified by expDatasetType.
            If an exposure is not found, it is skipped with a warning.
        @param[in] imageScalerList: list of image scalers (coaddUtils.ImageScaler);
            must be the same length as expRefList
        @param[in] expDatasetType: dataset type of exposure: e.g. 'goodSeeingCoadd_tempExp'

        @return: index of best exposure
        
        @raise pipeBase.TaskError if none of the exposures in expRefList are found.
        """
        self.log.info("Calculating best reference visit")
        varList =  []
        meanBkgdLevelList = []
        coverageList = []
        
        if len(expRefList) != len(imageScalerList):
            raise RuntimeError("len(expRefList) = %s != %s = len(imageScalerList)" % \
                (len(expRefList), len(imageScalerList)))

        for expRef, imageScaler  in zip(expRefList, imageScalerList):
            exposure = expRef.get(expDatasetType, immediate=True)
            maskedImage = exposure.getMaskedImage()
            imageScaler.scaleMaskedImage(maskedImage)
            statObjIm = afwMath.makeStatistics(maskedImage.getImage(), maskedImage.getMask(),
                afwMath.MEAN | afwMath.NPOINT | afwMath.VARIANCE, self.sctrl)
            meanVar, meanVarErr = statObjIm.getResult(afwMath.VARIANCE)
            meanBkgdLevel, meanBkgdLevelErr = statObjIm.getResult(afwMath.MEAN)
            npoints, npointsErr = statObjIm.getResult(afwMath.NPOINT)
            varList.append(meanVar)
            meanBkgdLevelList.append(meanBkgdLevel)
            coverageList.append(npoints)

        if not coverageList:
             raise pipeBase.TaskError(
                "None of the candidate %s exist; cannot select best reference exposure" % (expDatasetType,))

        # Normalize metrics to range from  0 to 1
        varArr = numpy.array(varList)/numpy.max(varList)
        meanBkgdLevelArr = numpy.array(meanBkgdLevelList)/numpy.max(meanBkgdLevelList)
        coverageArr = numpy.min(coverageList)/numpy.array(coverageList)

        costFunctionArr = self.config.bestRefWeightVariance  * varArr
        costFunctionArr += self.config.bestRefWeightLevel * meanBkgdLevelArr
        costFunctionArr += self.config.bestRefWeightCoverage * coverageArr
        return numpy.argmin(costFunctionArr)

    @pipeBase.timeMethod
    def matchBackgrounds(self, refExposure, sciExposure):
        """
        Match science exposure's background level to that of reference exposure.

        Process creates a difference image of the reference exposure minus the science exposure, and then
        generates an afw.math.Background object. It assumes (but does not require/check) that the mask plane
        already has detections set. If detections have not been set/masked, sources will bias the
        background estimation. 
        The 'background' of the difference image is smoothed by spline interpolation (by the Background class)
        or by polynomial interpolation by the Approximate class. This model of difference image is added to the
        science exposure in memory.
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

        # Some config and input checks if config.usePolynomial:
        # 1) Check that order/bin size make sense:
        # 2) Change binsize or order if underconstrained.
        # 3) Add some tiny Gaussian noise if the image is completely uniform
        #        (change after ticket 2411)
        if self.config.usePolynomial:
            order = self.config.order
            bgX, bgY, bgZ, bgdZ = self._gridImage(diffMI, self.config.binSize, statsFlag)
            minNumberGridPoints = min(len(set(bgX)),len(set(bgY)))
            if len(bgZ) == 0:
                raise ValueError("No overlap with reference. Nothing to match")
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
                self.debugDataIdString, e))

        sciMI  = sciExposure.getMaskedImage()
        sciMI += bkgdImage
        del sciMI

        if lsstDebug.Info(__name__).savefits:
            sciExposure.writeFits(lsstDebug.Info(__name__).figpath + 'sciMatchedExposure.fits')

        if lsstDebug.Info(__name__).savefig:
            bbox  = afwGeom.Box2D(refExposure.getMaskedImage().getBBox(afwImage.PARENT))
            X, Y, Z, dZ = self._gridImage(diffMI, self.config.binSize, statsFlag)
            x0, y0 = diffMI.getXY0()
            Xshift = X - x0 # get positions in local image coords
            Yshift = Y - y0
            modelValueArr = numpy.empty(len(Z))
            for i in range(len(X)):
                modelValueArr[i] = bkgdImage.get(int(Xshift[i]),int(Yshift[i]))
            resids = Z - modelValueArr
            try:
                self._debugPlot(X, Y, Z, dZ, bkgdImage, bbox, modelValueArr, resids)
            except Exception, e:
                self.log.warn('Debug plot not generated: %s'%(e))

        meanVar = afwMath.makeStatistics(diffMI.getVariance(),diffMI.getMask(),
                                         afwMath.MEAN, self.sctrl).getValue()

        diffIm  = diffMI.getImage()
        diffIm -= bkgdImage #diffMI should now have a mean ~ 0
        del diffIm
        mse = afwMath.makeStatistics(diffMI, afwMath.MEANSQUARE, self.sctrl).getValue()

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

        @param X: array of x positions
        @param Y: array of y positions
        @param Z: array of the grid values that were interpolated
        @param dZ: array of the error on the grid values
        @param modelImage: image ofthe model of the fit
        @param model: array of len(Z) containing the grid values predicted by the model
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
            max, min  = numpy.max(Z), numpy.min(Z)
            norm = matplotlib.colors.normalize(vmax=max, vmin= min)
            maxdiff = numpy.max(numpy.abs(resids))
            diffnorm = matplotlib.colors.normalize(vmax=maxdiff, vmin= -maxdiff)
            rms = numpy.sqrt(numpy.mean(resids**2))
            from mpl_toolkits.axes_grid1 import ImageGrid
            fig = plt.figure(1, (8, 6))
            meanDz = numpy.mean(dZ)
            grid = ImageGrid(fig, 111, nrows_ncols = (1, 2), axes_pad=0.1,
                             share_all=True, label_mode = "L", cbar_mode = "each",
                             cbar_size = "7%", cbar_pad="2%", cbar_location = "top")
            im = grid[0].imshow(zeroIm.getImage().getArray(),
                                extent=(x0, x0+dx, y0+dy, y0), norm = norm,
                                cmap='Spectral')
            im = grid[0].scatter(X, Y, c=Z, s = 15.*meanDz/dZ, edgecolor='none', norm=norm,
                                 marker='o',cmap='Spectral')
            im2 = grid[1].scatter(X, Y,  c=resids, edgecolor='none', norm=diffnorm,
                                  marker='s', cmap='seismic')
            grid.cbar_axes[0].colorbar(im)
            grid.cbar_axes[1].colorbar(im2)
            grid[0].axis([x0, x0+dx, y0+dy, y0])
            grid[1].axis([x0, x0+dx, y0+dy, y0])
            grid[0].set_xlabel("model and grid")
            grid[1].set_xlabel("residuals. rms = %0.3f"%(rms))
            if lsstDebug.Info(__name__).savefig:
                fig.savefig(lsstDebug.Info(__name__).figpath + self.debugDataIdString + '.png')
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

        return numpy.array(bgX), numpy.array(bgY), numpy.array(bgZ), numpy.array(bgdZ)

class IdMatcher(object):
    def __init__(self, keyNames):
        """Construct an IdMatcher
        
        @param[in] keyNames: list of key names
        """
        self._keyNames = tuple(keyNames)
    
    def _makeKey(self, id):
        """Return a tuple of values for the specified keyNames
        
        @param[in] id: ID dict

        @raise KeyError if id is missing a key in keyNames
        """
        return tuple(dataId[key] for key in self._keyNames)

    def isMatch(self, id0, id1):
        """Return True if id0 == id1 for the keys in keyNames
        
        @param[in] id0: ID dict 0
        @param[in] id1: ID dict 1
        
        @raise KeyError if either ID is missing a key in keyNames
        """
        return self._makeKey(id0) == self._makeKey(id1)
    
    def matchList(self, id0, idList):
        """Return a list of indices of matches
        
        @return tuple of indices of matches
        
        @raise KeyError if any ID is missing a key in keyNames
        """
        key0 = self._makeKey(id0)
        return tuple(ind for ind, id in idList if self_makeKey(id) == key0)
