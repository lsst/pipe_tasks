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
from lsst.afw.image import ImageF
from lsst.afw.math import StatisticsControl, statisticsStack, stringToStatisticsProperty
from lsst.geom import Point2I
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
        storageClass="ImageF",  # MaskedImageF
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
        default=2,
    )
    globalReducedChiSquaredThreshold = Field[float](
        doc="Threshold for global reduced chi-squared for bright star stamps.",
        default=5.0,
    )
    psfReducedChiSquaredThreshold = Field[float](
        doc="Threshold for PSF reduced chi-squared for bright star stamps.",
        default=50.0,
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
    stackType = Field[str](
        default="MEANCLIP",
        doc="Statistic name to use for stacking (from `~lsst.afw.math.Property`)",
    )
    stackNumSigmaClip = Field[float](
        doc="Number of sigma to use for clipping when stacking.",
        default=3.0,
    )
    stackNumIter = Field[int](
        doc="Number of iterations to use for clipping when stacking.",
        default=5,
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

    def _applyStampFit(self, stamp):
        """Apply fitted stamp components to a single bright star stamp."""
        stampMI = stamp.maskedImage
        stampBBox = stampMI.getBBox()
        xGrid, yGrid = np.meshgrid(stampBBox.getX().arange(), stampBBox.getY().arange())
        xPlane = ImageF((xGrid * stamp.xGradient).astype(np.float32), xy0=stampMI.getXY0())
        yPlane = ImageF((yGrid * stamp.yGradient).astype(np.float32), xy0=stampMI.getXY0())
        stampMI -= stamp.pedestal
        stampMI -= xPlane
        stampMI -= yPlane
        stampMI /= stamp.scale

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
        stackTypeProperty = stringToStatisticsProperty(self.config.stackType)
        statisticsControl = StatisticsControl(
            numSigmaClip=self.config.stackNumSigmaClip,
            numIter=self.config.stackNumIter,
        )

        subsetStampMIs = []
        tempStampMIs = []
        all_stars = 0
        used_stars = 0
        for stampsDDH in brightStarStamps:
            stamps = stampsDDH.get()
            all_stars += len(stamps)
            for stamp in stamps:
                if (
                    stamp.globalReducedChiSquared > self.config.globalReducedChiSquaredThreshold
                    or stamp.psfReducedChiSquared > self.config.psfReducedChiSquaredThreshold
                ):
                    continue
                stampMI = stamp.maskedImage
                self._applyStampFit(stamp)
                tempStampMIs.append(stampMI)

                badMaskBitMask = stampMI.mask.getPlaneBitMask(self.config.badMaskPlanes)
                statisticsControl.setAndMask(badMaskBitMask)

                # Amir: In case the total number of stamps is less than 20, the following will result in an
                # empty subsetStampMIs list.
                if len(tempStampMIs) == self.config.subsetStampNumber:
                    subsetStampMIs.append(statisticsStack(tempStampMIs, stackTypeProperty, statisticsControl))
                    # TODO: what to do with remaining temp stamps?
                    tempStampMIs = []
                    used_stars += self.config.subsetStampNumber

        self.metadata["psfStarCount"] = {}
        self.metadata["psfStarCount"]["all"] = all_stars
        self.metadata["psfStarCount"]["used"] = used_stars
        # TODO: which stamp mask plane to use here?
        # TODO: Amir: there might be cases where subsetStampMIs is an empty list. What do we want to do then?
        # Currently, we get an "IndexError: list index out of range"
        badMaskBitMask = subsetStampMIs[0].mask.getPlaneBitMask(self.config.badMaskPlanes)
        statisticsControl.setAndMask(badMaskBitMask)
        extendedPsfMI = statisticsStack(subsetStampMIs, stackTypeProperty, statisticsControl)

        extendedPsfExtent = extendedPsfMI.getBBox().getDimensions()
        extendedPsfOrigin = Point2I(-1 * (extendedPsfExtent.x // 2), -1 * (extendedPsfExtent.y // 2))
        extendedPsfMI.setXY0(extendedPsfOrigin)
        # return Struct(extendedPsf=[extendedPsfMI])

        return Struct(extendedPsf=extendedPsfMI.getImage())

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
