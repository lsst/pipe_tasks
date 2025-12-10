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
    dimensions=("instrument", "band"),
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
        storageClass="ImageF",  # stamp_imF
        doc="Extended PSF model, built from stacking bright star cutouts.",
        dimensions=("band",),
    )


class BrightStarStackConfig(
    PipelineTaskConfig,
    pipelineConnections=BrightStarStackConnections,
):
    """Configuration parameters for BrightStarStackTask."""

    global_reduced_chi_squared_threshold = Field[float](
        doc="Threshold for global reduced chi-squared for bright star stamps.",
        default=5.0,
    )
    psf_reduced_chi_squared_threshold = Field[float](
        doc="Threshold for PSF reduced chi-squared for bright star stamps.",
        default=50.0,
    )
    bright_star_threshold = Field[float](
        doc="Stars brighter than this magnitude, are considered as bright stars.",
        default=12.0,
    )
    bright_global_reduced_chi_squared_threshold = Field[float](
        doc="Threshold for global reduced chi-squared for bright star stamps.",
        default=250.0,
    )
    psf_bright_reduced_chi_squared_threshold = Field[float](
        doc="Threshold for PSF reduced chi-squared for bright star stamps.",
        default=400.0,
    )

    bad_mask_planes = ListField[str](
        doc="Mask planes that identify excluded (masked) pixels.",
        default=[
            "BAD",
            "CR",
            "CROSSTALK",
            "EDGE",
            "NO_DATA",
            "SAT",
            "SUSPECT",
            "UNMASKEDNAN",
            NEIGHBOR_MASK_PLANE,
        ],
    )
    stack_type = Field[str](
        default="MEDIAN",
        doc="Statistic name to use for stacking (from `~lsst.afw.math.Property`)",
    )
    stack_num_sigma_clip = Field[float](
        doc="Number of sigma to use for clipping when stacking.",
        default=3.0,
    )
    stack_num_iter = Field[int](
        doc="Number of iterations to use for clipping when stacking.",
        default=5,
    )
    magnitude_bins = ListField[int](
        doc="Only used if stack_type == WEIGHTED_MEDIAN. Bins of magnitudes for weighting purposes.",
        default=[20, 19, 18, 17, 16, 15, 13, 10],
    )
    subset_stamp_number = ListField[int](
        doc="Only used if stack_type == WEIGHTED_MEDIAN. Number of stamps per subset to generate stacked "
        "images for. The length of this parameter must be equal to the length of magnitude_bins plus one.",
        default=[300, 200, 150, 100, 100, 100, 1],
    )
    min_focal_plane_radius = Field[float](
        doc="Minimum distance to focal plane center in mm. Stars with a focal plane radius smaller than "
        "this will be omitted.",
        default=-1.0,
    )
    max_focal_plane_radius = Field[float](
        doc="Maximum distance to focal plane center in mm. Stars with a focal plane radius greater than "
        "this will be omitted.",
        default=2000.0,
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
        stampMI = stamp.stamp_im
        stamp_bbox = stampMI.getBBox()

        x_grid, y_grid = np.meshgrid(stamp_bbox.getX().arange(), stamp_bbox.getY().arange())

        x_plane = ImageF((x_grid * stamp.gradient_x).astype(np.float32), xy0=stampMI.getXY0())
        y_plane = ImageF((y_grid * stamp.gradient_y).astype(np.float32), xy0=stampMI.getXY0())

        x_curve = ImageF((x_grid**2 * stamp.curvature_x).astype(np.float32), xy0=stampMI.getXY0())
        y_curve = ImageF((y_grid**2 * stamp.curvature_y).astype(np.float32), xy0=stampMI.getXY0())
        xy_curve = ImageF((x_grid * y_grid * stamp.curvature_xy).astype(np.float32), xy0=stampMI.getXY0())

        stampMI -= stamp.pedestal
        stampMI -= x_plane
        stampMI -= y_plane
        stampMI -= x_curve
        stampMI -= y_curve
        stampMI -= xy_curve
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
        stack_type_property = stringToStatisticsProperty(self.config.stack_type)
        statistics_control = StatisticsControl(
            numSigmaClip=self.config.stack_num_sigma_clip,
            numIter=self.config.stack_num_iter,
        )

        mag_bins_dict = {}
        subset_stampMIs = {}
        self.metadata["psf_star_count"] = {}
        self.metadata["psf_star_count"]["all"] = 0
        for i in range(len(self.config.subset_stamp_number)):
            self.metadata["psf_star_count"][str(self.config.magnitude_bins[i + 1])] = 0
        for stampsDDH in brightStarStamps:
            stamps = stampsDDH.get()
            self.metadata["psf_star_count"]["all"] += len(stamps)
            for stamp in stamps:
                if stamp.ref_mag >= self.config.bright_star_threshold:
                    global_reduced_chi_squared_threshold = self.config.global_reduced_chi_squared_threshold
                    psf_reduced_chi_squared_threshold = self.config.psf_reduced_chi_squared_threshold
                else:
                    global_reduced_chi_squared_threshold = (
                        self.config.bright_global_reduced_chi_squared_threshold
                    )
                    psf_reduced_chi_squared_threshold = self.config.psf_bright_reduced_chi_squared_threshold
                for i in range(len(self.config.subset_stamp_number)):
                    if (
                        stamp.global_reduced_chi_squared > global_reduced_chi_squared_threshold
                        or stamp.psf_reduced_chi_squared > psf_reduced_chi_squared_threshold
                        or stamp.focal_plane_radius < self.config.min_focal_plane_radius
                        or stamp.focal_plane_radius > self.config.max_focal_plane_radius
                    ):
                        continue

                    if (
                        stamp.ref_mag < self.config.magnitude_bins[i]
                        and stamp.ref_mag > self.config.magnitude_bins[i + 1]
                    ):
                        self._applyStampFit(stamp)
                        if not self.config.magnitude_bins[i + 1] in mag_bins_dict.keys():
                            mag_bins_dict[self.config.magnitude_bins[i + 1]] = []
                        stampMI = stamp.stamp_im
                        mag_bins_dict[self.config.magnitude_bins[i + 1]].append(stampMI)
                        bad_mask_bit_mask = stampMI.mask.getPlaneBitMask(self.config.bad_mask_planes)
                        statistics_control.setAndMask(bad_mask_bit_mask)
                        if (
                            len(mag_bins_dict[self.config.magnitude_bins[i + 1]])
                            == self.config.subset_stamp_number[i]
                        ):
                            if self.config.magnitude_bins[i + 1] not in subset_stampMIs.keys():
                                subset_stampMIs[self.config.magnitude_bins[i + 1]] = []
                            subset_stampMIs[self.config.magnitude_bins[i + 1]].append(
                                statisticsStack(
                                    mag_bins_dict[self.config.magnitude_bins[i + 1]],
                                    stack_type_property,
                                    statistics_control,
                                )
                            )
                            self.metadata["psf_star_count"][str(self.config.magnitude_bins[i + 1])] += len(
                                mag_bins_dict[self.config.magnitude_bins[i + 1]]
                            )
                            mag_bins_dict[self.config.magnitude_bins[i + 1]] = []

        for key in mag_bins_dict.keys():
            if key not in subset_stampMIs.keys():
                subset_stampMIs[key] = []
                subset_stampMIs[key].append(
                    statisticsStack(mag_bins_dict[key], stack_type_property, statistics_control)
                )
                self.metadata["psf_star_count"][str(key)] += len(mag_bins_dict[key])

        final_subset_stampMIs = []
        for key in subset_stampMIs.keys():
            final_subset_stampMIs.extend(subset_stampMIs[key])
        bad_mask_bit_mask = final_subset_stampMIs[0].mask.getPlaneBitMask(self.config.bad_mask_planes)
        statistics_control.setAndMask(bad_mask_bit_mask)
        extendedPsfMI = statisticsStack(final_subset_stampMIs, stack_type_property, statistics_control)

        extendedPsfExtent = extendedPsfMI.getBBox().getDimensions()
        extendedPsfOrigin = Point2I(-1 * (extendedPsfExtent.x // 2), -1 * (extendedPsfExtent.y // 2))
        extendedPsfMI.setXY0(extendedPsfOrigin)

        return Struct(extendedPsf=extendedPsfMI.getImage())
