#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2008, 2009, 2010, 2011, 2012 LSST Corporation.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
import math, os
import lsst.pex.config as pexConfig
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.meas.algorithms.detection as detection
import lsst.coadd.utils as coaddUtils
import lsst.pipe.base as pipeBase
from lsst.pipe.tasks.processCcdSdss import ProcessCcdSdssTask
from lsst.pipe.tasks.outlierRejectedCoadd import OutlierRejectedCoaddConfig, OutlierRejectedCoaddTask, \
    ExposureMetadata, _subBBoxIter, ExposureMetadata
from .coadd import CoaddTask

import lsst.afw.display.ds9 as ds9      # useful for debugging. Can go in production

class SkyMatchedOutlierRejectedCoaddConfig(OutlierRejectedCoaddConfig):
    nRunMax = pexConfig.Field(
        doc = "If > 0, process at most this many runs (debugging only)",
        dtype = int,
        default = 0,
    )
    readMergedExposure = pexConfig.Field(
        doc = "Read per-run merged exposures from disk (if available) rather than recomputing them",
        dtype = bool,
        default = True,
    )
    saveMergedImage = pexConfig.Field(
        doc = "Save the per-run merged exposures to disk as coaddTempExp, field == 0",
        dtype = bool,
        default = True,
    )    
    saveMatchedImage = pexConfig.Field(
        doc = "Save the per-run matched exposures to disk as coaddTempExp, field == 1",
        dtype = bool,
        default = False,
    )    
    matchBackground = pexConfig.Field(
        doc = "Match the background levels between runs.  N.b. If this is False the backgrounds must have been pre-subtracted",
        dtype = bool,
        default = True,
    )    
    canonicalRun = pexConfig.Field(
        doc = "Reference all runs to this one",
        dtype = int,
        default = -1,
    )
    usePerRunCalib = pexConfig.Field(
        doc = "Use a single zero point for a patch for all the fields of a run",
        dtype = bool,
        default = True,
        )

    statisticsProperty = pexConfig.ChoiceField(
        doc="type of statistic to use for grid points when matching",
        dtype=str, default="MEDIAN",
        allowed={
            "MEANCLIP": "clipped mean",
            "MEAN": "unclipped mean",
            "MEDIAN": "median",
            }
        )
    undersampleStyle = pexConfig.ChoiceField(
        doc="behaviour if there are too few points in grid for requested interpolation style when matching",
        dtype=str, default="REDUCE_INTERP_ORDER",
        allowed={
            "THROW_EXCEPTION": "throw an exception if there are too few points",
            "REDUCE_INTERP_ORDER": "use an interpolation style with a lower order.",
            "INCREASE_NXNYSAMPLE": "Increase the number of samples used to make the interpolation grid.",
            }
        )
    binSize = pexConfig.RangeField(
        doc="how large a region of the sky should be used for each background point when matching",
        dtype=int, default=1024,
        min=10
        )
    
class PerRunZeroPointScaler(coaddUtils.ZeroPointScaler):
    def __init__(self, *args, **kwargs):
        coaddUtils.ZeroPointScaler.__init__(self, *args, **kwargs)
        self.perRunCalib = None
        
    def scaleExposure(self, exposure):
        """Scale exposure to the desired photometric zeroPoint, in place
        
        @param[in,out] exposure: exposure to scale; it must have a valid Calib;
            the pixel values and Calib zeroPoint are scaled
        @return scaleFac: scale factor, where new image values = original image values * scaleFac
        """
        exposureCalib = self.perRunCalib if self.perRunCalib else exposure.getCalib()

        exposureFluxMag0Sigma = exposureCalib.getFluxMag0()[1]
        fluxAtZeroPoint = exposureCalib.getFlux(self._zeroPoint)
        scaleFac = 1.0 / fluxAtZeroPoint
        maskedImage = exposure.getMaskedImage()
        maskedImage *= scaleFac
        exposureFluxMag0Sigma *= scaleFac
        exposure.getCalib().setFluxMag0(self.getCalib().getFluxMag0()[0], exposureFluxMag0Sigma)
        
        return scaleFac

class SkyMatchedOutlierRejectedCoaddTask(OutlierRejectedCoaddTask):
    """Construct a sky-matched outlier-rejected (robust mean) coadd
    """
    ConfigClass = SkyMatchedOutlierRejectedCoaddConfig
    _DefaultName = "outlierRejectedCoadd"

    def __init__(self, *args, **kwargs):
        OutlierRejectedCoaddTask.__init__(self, *args, **kwargs)
        self.zeroPointScaler = PerRunZeroPointScaler(self.config.coaddZeroPoint)
    
    @pipeBase.timeMethod
    def run(self, patchRef):
        """PSF-match, warp and coadd images, using outlier rejection
        
        PSF matching is to a double gaussian model with core FWHM = desiredFwhm
        and wings of amplitude 1/10 of core and FWHM = 2.5 * core.
        The size of the PSF matching kernel is the same as the size of the kernel
        found in the first calibrated science exposure, since there is no benefit
        to making it any other size.
        
        PSF-matching is performed before warping so the code can use the PSF models
        associated with the calibrated science exposures (without having to warp those models).
        
        @param patchRef: data reference for sky map. Must include keys "tract", "patch",
            plus the camera-specific filter key (e.g. "filter" or "band")

        @return: a pipeBase.Struct with fields:
        - coaddExposure: coadd exposure
        """
        skyInfo = self.getSkyInfo(patchRef)
        datasetType = self.config.coaddName + "Coadd"
        
        wcs = skyInfo.wcs
        bbox = skyInfo.bbox
        
        imageRefList = self.selectExposures(patchRef=patchRef, wcs=wcs, bbox=bbox)

        numExp = len(imageRefList)
        if numExp < 1:
            raise pipeBase.TaskError("No exposures to coadd")
        self.log.log(self.log.INFO, "Coadd %s calexp" % (numExp,))
    
        doPsfMatch = self.config.desiredFwhm > 0
        if not doPsfMatch:
            self.log.log(self.log.INFO, "No PSF matching will be done (desiredFwhm <= 0)")
        #
        # Setup to match backgrounds
        #
        config = detection.BackgroundConfig()
        config.isNanSafe = True
        config.statisticsProperty = self.config.statisticsProperty
        config.undersampleStyle = self.config.undersampleStyle
        config.binSize = self.config.binSize
        #
        # Reorganise the imageRefList to put the canonical run first
        #
        #imageRefList.sort(lambda a, b: cmp(a.dataId["run"], b.dataId["run"])) # sorted by run number

        runs = sorted(list(set([x.dataId["run"] for x in imageRefList]))) # available runs

        canonicalRun = self.config.canonicalRun
        if canonicalRun <= 0:
            canonicalRun = runs[0]
        else:
            if not canonicalRun in runs:
                self.log.log(self.log.ERR, "Canonical run %d is not included in the list of available runs: %s"
                             % (canonicalRun, ", ".join([str(r) for r in runs])))
                

            runs = [canonicalRun] + [r for r in runs if r != canonicalRun]

        if self.config.nRunMax > 0 and self.config.nRunMax < len(runs):
            runs = runs[0:self.config.nRunMax]
            numExp = 0
            for run in runs:
                numExp += len([x for x in imageRefList if x.dataId["run"] == run])

            print "Trimming to %d runs, %d fields" % (len(runs), numExp)

        exposureMetadataList = []
        ind = 0                         # counter for exposures we're processing
        for run in runs:
            thisRunImageRefList = [x for x in imageRefList if x.dataId["run"] == run]

            dataRef = thisRunImageRefList[0]

            tempDataId = dataRef.dataId.copy()
            tempDataId.update(patchRef.dataId)
            tempDataId["field"] = 0     # per-run patch image

            tempDataRef = dataRef.butlerSubset.butler.dataRef(
                datasetType = "coaddTempExp",
                dataId = tempDataId,
            )

            if self.config.readMergedExposure and tempDataRef.datasetExists():
                nExp = len(thisRunImageRefList)
                self.log.log(self.log.INFO, "Reading %d..%d of %d: merge of %s for %s" % \
                                 (ind + 1, ind + nExp, numExp,
                                  "run %(run)s %(filter)s%(camcol)s" % dataRef.dataId,
                                  "tract %(tract)d patch %(patch)s" % patchRef.dataId))
                exposure = tempDataRef.get()
                ind += nExp
            else:
                # Get the read-proxies for all the exposures belonging to this run
                exposures = []
                for dataRef in thisRunImageRefList:
                    if dataRef.datasetExists("fpC"):
                        exposures.append(self.getCalExp(dataRef, getPsf=doPsfMatch))
                    else:
                        exposures.append(None)

                if self.config.usePerRunCalib:
                    fluxMag0, fluxMagErr0, nexp = 0.0, 0.0, 0
                    for exposure in exposures:
                        if exposure:
                            fm0, fm0Err = exposure.getCalib().getFluxMag0()

                            nexp += 1
                            fluxMag0 += fm0
                            fluxMagErr0 += fm0Err**2

                    fluxMag0 /= nexp
                    fluxMagErr0 /= nexp
                    self.zeroPointScaler.perRunCalib = afwImage.Calib()
                    self.zeroPointScaler.perRunCalib.setFluxMag0(fluxMag0, math.sqrt(fluxMagErr0))

                runPatchExposure = None                         # where we'll mosaic all the fields in the run
                maskedImageList = afwImage.vectorMaskedImageF() # [] is rejected by afwMath.statisticsStack
                for dataRef, exposure in zip(thisRunImageRefList, exposures):
                    ind += 1
                    if not exposure:
                        self.log.log(self.log.WARN, "Could not find fpC %s; skipping it" % (dataRef.dataId,))
                        continue

                    self.log.log(self.log.INFO, "Processing exposure %d of %d: %s for %s" % \
                                     (ind, numExp,
                                      "run %(run)s %(filter)s%(camcol)s %(field)s" % dataRef.dataId,
                                      "tract %(tract)d patch %(patch)s" % patchRef.dataId))

                    try:
                        exposure = self.preprocessExposure(exposure, wcs=wcs, destBBox=bbox)
                    except Exception, e:
                        self.log.log(self.log.WARN, "Error preprocessing exposure %s; skipping it: %s" % \
                            (dataRef.dataId, e))
                        continue

                    expMeta = ExposureMetadata(None, exposure, self.getBadPixelMask())

                    maskedImageList.append(exposure.getMaskedImage())

                    if not runPatchExposure:
                        runPatchExposure = exposure

                if maskedImageList:
                    try:
                        self.log.log(self.log.INFO, "Assembling %d for %s" % (run, patchRef.dataId))
                    except Exception, e:
                        print "RHL", e
                exposure = runPatchExposure
                exposure.setMaskedImage(mergeMaskedImagesFromRun(maskedImageList))

                if self.config.saveMergedImage:
                    tempDataRef.put(exposure)

            if self.config.matchBackground:
                #
                # if we aren't the canonical run, match to it
                #
                if run == canonicalRun:
                    canonicalExposure = exposure
                else:
                    self.log.log(self.log.INFO, "Matching run %d to run %d" % (run, canonicalRun))
                    mi = exposure.getMaskedImage()
                    diff = mi.Factory(mi, True)
                    diff -= canonicalExposure.getMaskedImage()

                    try:
                        bkgd = detection.getBackground(diff, config)
                    except Exception, e:
                        print >> sys.stderr, "Failed to fit background for %d: %s" % (run, e)
                        continue

                    mi -= bkgd.getImageF()

                expMetadata = ExposureMetadata(
                    dataRef = tempDataRef,
                    exposure = exposure,
                    badPixelMask = self.getBadPixelMask(),
                    )
                exposureMetadataList.append(expMetadata)

                tempDataRef.dataId["field"] = 1 # matched image
                tempDataRef.put(exposure)
            
            del exposure
        #
        # We have all the runs warped to the patch's footprint, so we can stack them together
        #
        if not exposureMetadataList:
            raise pipeBase.TaskError("No images to coadd")

        edgeMask = afwImage.MaskU.getPlaneBitMask("EDGE")
        
        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setNumSigmaClip(self.config.sigmaClip)
        statsCtrl.setNumIter(self.config.clipIter)
        statsCtrl.setAndMask(self.getBadPixelMask())
        statsCtrl.setNanSafe(True)
        statsCtrl.setCalcErrorFromInputVariance(True)

        if self.config.doSigmaClip:
            statsFlags = afwMath.MEANCLIP
        else:
            statsFlags = afwMath.MEAN
    
        coaddExposure = afwImage.ExposureF(bbox, wcs)
        coaddExposure.setCalib(self.zeroPointScaler.getCalib())
    
        filterDict = {} # dict of name: Filter
        for expMeta in exposureMetadataList:
            filterDict.setdefault(expMeta.filter.getName(), expMeta.filter)
        if len(filterDict) == 1:
            coaddExposure.setFilter(filterDict.values()[0])
        self.log.log(self.log.INFO, "Filter=%s" % (coaddExposure.getFilter().getName(),))
    
        coaddMaskedImage = coaddExposure.getMaskedImage()
        subregionSizeArr = self.config.subregionSize
        subregionSize = afwGeom.Extent2I(subregionSizeArr[0], subregionSizeArr[1])
        for subBBox in _subBBoxIter(bbox, subregionSize):
            self.log.log(self.log.INFO, "Computing coadd %s" % (subBBox,))
            coaddView = afwImage.MaskedImageF(coaddMaskedImage, subBBox, afwImage.PARENT, False)
            maskedImageList = afwImage.vectorMaskedImageF() # [] is rejected by afwMath.statisticsStack
            weightList = []
            for expMeta in exposureMetadataList:
                if not subBBox.overlaps(expMeta.bbox):
                    # there is no overlap between this temporary exposure and this coadd subregion
                    self.log.log(self.log.INFO, "Skipping %s; no overlap" % (expMeta.path,))
                    continue
                
                if expMeta.bbox.contains(subBBox):
                    # this temporary image fully overlaps this coadd subregion
                    exposure = expMeta.dataRef.get("coaddTempExp_sub", bbox=subBBox, imageOrigin="PARENT")
                    maskedImage = exposure.getMaskedImage()
                else:
                    # this temporary image partially overlaps this coadd subregion;
                    # make a new image of EDGE pixels using the coadd subregion
                    # and set the overlapping pixels from the temporary exposure
                    overlapBBox = afwGeom.Box2I(expMeta.bbox)
                    overlapBBox.clip(subBBox)
                    self.log.log(self.log.INFO,
                        "Processing %s; grow from %s to %s" % (expMeta.path, overlapBBox, subBBox))
                    maskedImage = afwImage.MaskedImageF(subBBox)
                    maskedImage.getMask().set(edgeMask)
                    tempExposure = expMeta.dataRef.get("coaddTempExp_sub",
                        bbox=overlapBBox, imageOrigin="PARENT")
                    tempMaskedImage = tempExposure.getMaskedImage()
                    maskedImageView = afwImage.MaskedImageF(maskedImage, overlapBBox, afwImage.PARENT, False)
                    maskedImageView <<= tempMaskedImage
                maskedImageList.append(maskedImage)
                weightList.append(expMeta.weight)

            if len(maskedImageList) > 0:
                try:
                    coaddSubregion = afwMath.statisticsStack(
                        maskedImageList, statsFlags, statsCtrl, weightList)
        
                    coaddView <<= coaddSubregion
                except Exception, e:
                    self.log.log(self.log.ERR, "Cannot compute this subregion: %s" % (e,))
            else:
                self.log.log(self.log.WARN, "No images to coadd in this subregion")
    
        coaddUtils.setCoaddEdgeBits(coaddMaskedImage.getMask(), coaddMaskedImage.getVariance())
        self.postprocessCoadd(coaddExposure)
        #
        # Add metadata about this coadd
        #
        md = coaddExposure.getMetadata()
        md.setInt("CANONRUN", canonicalRun)
        for i, run in enumerate(runs):
            thisRunImageRefList = [x for x in imageRefList if x.dataId["run"] == run]
            for dataRef in thisRunImageRefList:
                md.addString("INPUT_FPC", "%(run)s %(filter)s%(camcol)s %(field)s" % dataRef.dataId)
        del md

        self.persistCoadd(patchRef, coaddExposure)
        #
        # We had to save the files to read back for the coadd, but now we can delete them
        # N.b. It's pretty quick to regenerate the matched files if we kept the per-run
        # warped images
        #
        if not self.config.saveMatchedImage:
            for expMeta in exposureMetadataList:
                try:
                    os.remove(expMeta.dataRef.get("coaddTempExp_filename")[0])
                except Exception, e:
                    self.log.log(self.log.WARN, e)
    
        return pipeBase.Struct(
            coaddExposure = coaddExposure,
        )

    def getCalExp(self, dataRef, getPsf=True):
        """Return one "calexp" calibrated exposure, perhaps with psf
        
        @param dataRef: a sensor-level data reference
        @param getPsf: include the PSF?
        @return calibrated exposure with psf
        """
        task = ProcessCcdSdssTask()
        task.config.removeOverlap = True
        task.config.overlapSize = 128 - 19     # leave enough for the Lanczos kernel

        exposure = task.makeExp(dataRef)
        if getPsf:
            psf = dataRef.get('psField')
            exposure.setPsf(psf)

        return exposure

def mergeMaskedImagesFromRun(maskedImageList):
    """Merge a set of maskedImages that are from the same run
    """
    if len(maskedImageList) == 0:
        raise RuntimeError("Failed to find any exposures")

    statsCtrl = afwMath.StatisticsControl()
    statsCtrl.setCalcErrorFromInputVariance(True)

    weights = [1.0]*len(maskedImageList)
    mi = afwMath.statisticsStack(maskedImageList, afwMath.MEAN, statsCtrl, weights)
    mi.setXY0(maskedImageList[0].getXY0())

    return mi
    
