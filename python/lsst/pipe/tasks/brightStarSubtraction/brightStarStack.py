# This file is part of pipe_tasks.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Stack bright star postage stamp cutouts to produce an extended PSF model."""

__all__ = ["BrightStarStackConnections", "BrightStarStackConfig", "BrightStarStackTask"]

import numpy as np
from astropy.stats import sigma_clip
from lsst.afw.image import ImageF, MaskedImageF
from lsst.afw.math import StatisticsControl, statisticsStack, stringToStatisticsProperty
from lsst.geom import Box2I, Point2I
from lsst.meas.algorithms import BrightStarStamps
from lsst.pex.config import Field, ListField
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base.connectionTypes import Input, Output
from lsst.utils.timer import timeMethod

NEIGHBOR_MASK_PLANE = "NEIGHBOR"


class BrightStarStackConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "detector"),
):
    """Connections for BrightStarStackTask."""

    brightStarStamps = Input(
        name="brightStarStamps",
        storageClass="BrightStarStamps",
        doc="Set of preprocessed postage stamp cutouts, each centered on a single bright star.",
        dimensions=("visit", "detector"),
        multiple=True,
        deferLoad=True,
    )
    extendedPsf = Output(
        name="extendedPsf2",  # extendedPsfDetector ???
        storageClass="ExtendedPsf",  # MaskedImageF
        doc="Extended PSF model, built from stacking bright star cutouts.",
        dimensions=("band",),
    )


class BrightStarStackConfig(
    PipelineTaskConfig,
    pipelineConnections=BrightStarStackConnections,
):
    """Configuration parameters for BrightStarStackTask."""

    subsetStampNumber = Field[int](
        doc="Number of stamps per subset to generate stacked images for.",
        default=20,
    )
    badMaskPlanes = ListField[str](
        doc="Mask planes that identify excluded (masked) pixels.",
        default=[
            "BAD",
            "CR",
            "CROSSTALK",
            "EDGE",
            "NO_DATA",
            # "SAT",
            # "SUSPECT",
            "UNMASKEDNAN",
            NEIGHBOR_MASK_PLANE,
        ],
    )


class BrightStarStackTask(PipelineTask):
    """Stack bright star postage stamps to produce an extended PSF model."""

    ConfigClass = BrightStarStackConfig
    _DefaultName = "brightStarStack"
    config: BrightStarStackConfig

    def __init__(self, initInputs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        output = self.run(**inputs)
        butlerQC.put(output, outputRefs)

    @timeMethod
    def run(
        self,
        brightStarStamps: BrightStarStamps,
    ):
        """Identify bright stars within an exposure using a reference catalog,
        extract stamps around each, then preprocess them.

        Bright star preprocessing steps are: shifting, warping and potentially
        rotating them to the same pixel grid; computing their annular flux,
        and; normalizing them.

        Parameters
        ----------
        inputExposure : `~lsst.afw.image.ExposureF`
            The image from which bright star stamps should be extracted.
        inputBackground : `~lsst.afw.image.Background`
            The background model for the input exposure.
        refObjLoader : `~lsst.meas.algorithms.ReferenceObjectLoader`, optional
            Loader to find objects within a reference catalog.
        dataId : `dict` or `~lsst.daf.butler.DataCoordinate`
            The dataId of the exposure (including detector) that bright stars
            should be extracted from.

        Returns
        -------
        brightStarResults : `~lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``brightStarStamps``
                (`~lsst.meas.algorithms.brightStarStamps.BrightStarStamps`)
        """
        statisticsControl = StatisticsControl(
            numSigmaClip=3,
            numIter=5,
        )

        extendedPsfMI = None
        extendedImages = []
        tempImages = []
        tempVariances = []
        tempIndex = 0
        for stampsDDH in brightStarStamps:
            stamps = stampsDDH.get()
            if not stamps:
                continue
            tempIndex += 1

            for stamp in stamps:
                stampMI = stamp.maskedImage
                stampBBox = stampMI.getBBox()

                # Apply fitted components
                stampMI -= stamp.pedestal
                xGrid, yGrid = np.meshgrid(stampBBox.getX().arange(), stampBBox.getY().arange())
                xPlane = ImageF((xGrid * stamp.xGradient).astype(np.float32), xy0=stampMI.getXY0())
                yPlane = ImageF((yGrid * stamp.yGradient).astype(np.float32), xy0=stampMI.getXY0())
                stampMI -= xPlane
                stampMI -= yPlane
                stampMI *= stamp.scale

                badMaskBitMask = stampMI.mask.getPlaneBitMask(self.config.badMaskPlanes)
                breakpoint()
                stampMask = (stampMI.mask.array & badMaskBitMask).astype(bool)
                stampMI.image.array[stampMask] = 0
                stampMI.variance.array[stampMask] = 0

                tempImages.append(stampMI.image.array)
                tempVariances.append(stampMI.variance.array)

                if tempIndex == self.config.numVisitStack:
                    # for i in range(tempImage.shape[0]):
                    #     for j in range(tempImage.shape[1]):
                    #         pixel_values = tempImage[i, j, :]
                    #         stats = afwMath.makeStatistics(pixel_values, afwMath.MEANCLIP, sctrl)
                    #         mean_image[i, j] = stats.getValue(afwMath.MEANCLIP)

                    tempImages2 = np.stack(tempImages)
                    tempVariances2 = np.stack(tempVariances)

                    clippedImages = sigma_clip(tempImages2, axis=2, sigma=3)
                    clippedVariances = sigma_clip(tempVariances2, axis=2, sigma=3)

                    clippedImage = np.mean(clippedImages, axis=2)
                    extendedImages.append(clippedImage)

                    tempIndex = 0

        breakpoint()
        extendedImages2 = np.stack(extendedImages)
        clippedImages2 = sigma_clip(extendedImages2, axis=0, sigma=3)
        extendedImage = np.mean(clippedImages2, axis=0)
        extendedPsfMI = MaskedImageF(image=ImageF(extendedImage), variance=ImageF(extendedImage))

        return Struct(extendedPsf=extendedPsfMI)

        # if stamp.psfReducedChiSquared > 5:
        #     continue
        # stampWeight = 1 / stamp.psfReducedChiSquared
        # stampMI *= stampWeight
        # stampWI = ImageF(stampBBox, stampWeight)
        # stampWI.array[stampMask] = 0

        # if not extendedPsfMI:
        #     extendedPsfMI = stampMI.clone()
        #     extendedPsfWI = stampWI.clone()
        # else:
        #     extendedPsfMI += stampMI
        #     extendedPsfWI += stampWI

        # if extendedPsfMI:
        #     extendedPsfMI /= extendedPsfWI
        # breakpoint()

        # stack = []
        # chiStack = []
        # for loop over all groups:
        # load up all visits for this detector
        # drop all with GOF > thresh
        # sigma-clip mean stack the rest
        # append to stack
        # compute the scatter (MAD/sigma-clipped var, etc) of the rest
        # divide by sqrt(var plane), and append to chiStack
        # after for-loop, combine images in median stack for final result
        # also combine chi-images, save separately

        # idea: run with two different thresholds, and compare the results

        # medianStack = []
        # for loop over all groups:
        # load up all visits for this detector
        # drop all with GOF > thresh
        # median/sigma-clip stack the rest
        # append to medianStack
        # after for-loop, combine images in median stack for final result

    def _configureStacking(self, numSigmaClip, numIter, badMaskBitMask, stackingStatistic):
        """Configure stacking statistic and control from config fields."""
        statisticsControl = StatisticsControl(numSigmaClip=numSigmaClip, numIter=numIter)
        statisticsFlag = stringToStatisticsProperty(stackingStatistic)
        statisticsControl.setAndMask(badMaskBitMask)
        return statisticsControl, statisticsFlag

    def _configureStacking(self, example_stamp):
        """Configure stacking statistic and control from config fields."""
        stats_control = StatisticsControl(
            numSigmaClip=self.config.num_sigma_clip,
            numIter=self.config.num_iter,
        )
        if bad_masks := self.config.bad_mask_planes:
            and_mask = example_stamp.mask.getPlaneBitMask(bad_masks[0])
            for bm in bad_masks[1:]:
                and_mask = and_mask | example_stamp.mask.getPlaneBitMask(bm)
            stats_control.setAndMask(and_mask)
        stats_flags = stringToStatisticsProperty(self.config.stacking_statistic)
        return stats_control, stats_flags
