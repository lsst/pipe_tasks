from __future__ import absolute_import, division, print_function


import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.coadd.utils as coaddUtils
import lsst.pex.config as pexConfig
from lsst.pipe.tasks.assembleCoadd import _subBBoxIter

from lsst.pipe.tasks.assembleCoadd import AssembleCoaddTask, AssembleCoaddConfig
#
# LSST Data Management System
# Copyright 2008-2016 AURA/LSST.
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
# see <https://www.lsstcorp.org/LegalNotices/>.
#


class DcrAssembleCoaddConfig(AssembleCoaddConfig):
    dcrBufferSize = pexConfig.Field(
        dtype=int,
        doc="Number of pixels to grow the subregion bounding box by.",
        default=5,
    )


class DcrAssembleCoaddTask(AssembleCoaddTask):

    ConfigClass = DcrAssembleCoaddConfig
    _DefaultName = "dcrAssembleCoadd"

    def assemble(self, skyInfo, tempExpRefList, imageScalerList, weightList, bgInfoList=None,
                 altMaskList=None, mask=None):
        """!
        \anchor AssembleCoaddTask.assemble_

        \brief Assemble a coadd from input warps

        Assemble the coadd using the provided list of coaddTempExps. Since the full coadd covers a patch (a
        large area), the assembly is performed over small areas on the image at a time in order to
        conserve memory usage. Iterate over subregions within the outer bbox of the patch using
        \ref assembleSubregion to stack the corresponding subregions from the coaddTempExps with the
        statistic specified. Set the edge bits the coadd mask based on the weight map.

        \param[in] skyInfo: Patch geometry information, from getSkyInfo
        \param[in] tempExpRefList: List of data references to Warps (previously called CoaddTempExps)
        \param[in] imageScalerList: List of image scalers
        \param[in] weightList: List of weights
        \param[in] bgInfoList: List of background data from background matching, or None
        \param[in] altMaskList: List of alternate masks to use rather than those stored with tempExp, or None
        \param[in] mask: Mask to ignore when coadding
        \return coadded exposure
        """
        tempExpName = self.getTempExpDatasetName(self.warpType)
        self.log.info("Assembling %s %s", len(tempExpRefList), tempExpName)
        if mask is None:
            mask = self.getBadPixelMask()

        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setNumSigmaClip(self.config.sigmaClip)
        statsCtrl.setNumIter(self.config.clipIter)
        statsCtrl.setAndMask(mask)
        statsCtrl.setNanSafe(True)
        statsCtrl.setWeighted(True)
        statsCtrl.setCalcErrorFromInputVariance(True)
        for plane, threshold in self.config.maskPropagationThresholds.items():
            bit = afwImage.Mask.getMaskPlane(plane)
            statsCtrl.setMaskPropagationThreshold(bit, threshold)

        statsFlags = afwMath.stringToStatisticsProperty(self.config.statistic)

        if bgInfoList is None:
            bgInfoList = [None]*len(tempExpRefList)

        if altMaskList is None:
            altMaskList = [None]*len(tempExpRefList)

        coaddExposure = afwImage.ExposureF(skyInfo.bbox, skyInfo.wcs)
        coaddExposure.setCalib(self.scaleZeroPoint.getCalib())
        coaddExposure.getInfo().setCoaddInputs(self.inputRecorder.makeCoaddInputs())
        self.assembleMetadata(coaddExposure, tempExpRefList, weightList)
        coaddMaskedImage = coaddExposure.getMaskedImage()
        subregionSizeArr = self.config.subregionSize
        subregionSize = afwGeom.Extent2I(subregionSizeArr[0], subregionSizeArr[1])
        for subBBox in _subBBoxIter(skyInfo.bbox, subregionSize):
            try:
                self.assembleSubregion(coaddExposure, subBBox, tempExpRefList, imageScalerList,
                                       weightList, bgInfoList, altMaskList, statsFlags, statsCtrl)
            except Exception as e:
                self.log.fatal("Cannot compute coadd %s: %s", subBBox, e)

        coaddUtils.setCoaddEdgeBits(coaddMaskedImage.getMask(), coaddMaskedImage.getVariance())

        return coaddExposure

    def assembleSubregion(self, coaddExposure, bbox, tempExpRefList, imageScalerList, weightList,
                          bgInfoList, altMaskList, statsFlags, statsCtrl):
        """!
        \brief Assemble the coadd for a sub-region.

        For each coaddTempExp, check for (and swap in) an alternative mask if one is passed. If background
        matching is enabled, add the background and background variance from each coaddTempExp. Remove mask
        planes listed in config.removeMaskPlanes, Finally, stack the actual exposures using
        \ref afwMath.statisticsStack "statisticsStack" with the statistic specified
        by statsFlags. Typically, the statsFlag will be one of afwMath.MEAN for a mean-stack or
        afwMath.MEANCLIP for outlier rejection using an N-sigma clipped mean where N and iterations
        are specified by statsCtrl.  Assign the stacked subregion back to the coadd.

        \param[in] coaddExposure: The target image for the coadd
        \param[in] bbox: Sub-region to coadd
        \param[in] tempExpRefList: List of data reference to tempExp
        \param[in] imageScalerList: List of image scalers
        \param[in] weightList: List of weights
        \param[in] bgInfoList: List of background data from background matching
        \param[in] altMaskList: List of alternate masks to use rather than those stored with tempExp, or None
        \param[in] statsFlags: afwMath.Property object for statistic for coadd
        \param[in] statsCtrl: Statistics control object for coadd
        """
        self.log.debug("Computing coadd over %s", bbox)
        bbox_grow = afwGeom.Box2I(bbox)
        bbox_grow.grow(self.config.dcrBufferSize)
        bbox_grow.clip(coaddExposure.getBBox())
        tempExpName = self.getTempExpDatasetName(self.warpType)
        coaddMaskedImage = coaddExposure.getMaskedImage()
        maskedImageList = []
        for tempExpRef, imageScaler, bgInfo, altMask in zip(tempExpRefList, imageScalerList, bgInfoList,
                                                            altMaskList):
            exposure = tempExpRef.get(tempExpName + "_sub", bbox=bbox_grow)
            maskedImage = exposure.getMaskedImage()

            if altMask:
                altMaskSub = altMask.Factory(altMask, bbox_grow, afwImage.PARENT)
                maskedImage.getMask().swap(altMaskSub)
            imageScaler.scaleMaskedImage(maskedImage)

            if self.config.doMatchBackgrounds and not bgInfo.isReference:
                backgroundModel = bgInfo.backgroundModel
                backgroundImage = backgroundModel.getImage() if \
                    self.matchBackgrounds.config.usePolynomial else \
                    backgroundModel.getImageF()
                backgroundImage.setXY0(coaddMaskedImage.getXY0())
                maskedImage += backgroundImage.Factory(backgroundImage, bbox_grow, afwImage.PARENT, False)
                var = maskedImage.getVariance()
                var += (bgInfo.fitRMS)**2

            if self.config.removeMaskPlanes:
                mask = maskedImage.getMask()
                for maskPlane in self.config.removeMaskPlanes:
                    try:
                        mask &= ~mask.getPlaneBitMask(maskPlane)
                    except Exception as e:
                        self.log.warn("Unable to remove mask plane %s: %s", maskPlane, e.message)

            maskedImageCut = maskedImage.Factory(maskedImage, bbox=bbox)
            maskedImageList.append(maskedImageCut)

        with self.timer("stack"):
            coaddSubregion = afwMath.statisticsStack(
                maskedImageList, statsFlags, statsCtrl, weightList)

        coaddMaskedImage.assign(coaddSubregion, bbox)
