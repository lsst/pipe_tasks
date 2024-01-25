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

"""Read preprocessed bright stars and stack to build an extended PSF model."""

__all__ = [
    "FocalPlaneRegionExtendedPsf",
    "ExtendedPsf",
    "StackBrightStarsConfig",
    "StackBrightStarsTask",
    "MeasureExtendedPsfConfig",
    "MeasureExtendedPsfTask",
    "DetectorsInRegion",
]

import numpy as np
from functools import reduce
from operator import ior
from dataclasses import dataclass

from lsst.afw.fits import Fits, readMetadata
from lsst.afw.geom import SpanSet, Stencil
from lsst.afw.image import ImageF, MaskedImageF, MaskX, ExposureF
from lsst.afw.math import StatisticsControl, statisticsStack, stringToStatisticsProperty, makeStatistics
from lsst.daf.base import PropertyList
from lsst.geom import Extent2I, Point2I, Box2I
from lsst.pex.config import ChoiceField, Config, ConfigDictField, ConfigurableField, Field, ListField
from lsst.pipe.base import PipelineTaskConfig, PipelineTaskConnections, Struct, Task
from lsst.pipe.base.connectionTypes import Input, Output
from lsst.pipe.tasks.coaddBase import subBBoxIter


def find_region_for_detector(detector_id, detectors_focal_plane_regions):
    """Find the focal plane region that contains a given detector.

    Parameters
    ----------
    detector_id : `int`
        The detector ID.

    detectors_focal_plane_regions :
            `dict` [`str`, `lsst.pipe.tasks.extended_psf.DetectorsInRegion`]
        A dictionary containing focal plane region names as keys, and the
        corresponding detector IDs encoded within the values.

    Returns
    -------
    key: `str`
        The name of the region to which the given detector belongs.

    Raises
    ------
    KeyError
        Raised if the given detector is not included in any focal plane region.
    """
    for region_id, detectors_in_region in detectors_focal_plane_regions.items():
        if detector_id in detectors_in_region.detectors:
            return region_id
    raise KeyError(
        "Detector %d is not included in any focal plane region.",
        detector_id,
    )


class DetectorsInRegion(Config):
    """Provides a list of detectors that define a region."""

    detectors = ListField[int](
        doc="A list containing the detectors IDs.",
        default=[],
    )


@dataclass
class FocalPlaneRegionExtendedPsf:
    """Single extended PSF over a focal plane region.

    The focal plane region is defined through a list of detectors.

    Parameters
    ----------
    extended_psf_image : `lsst.afw.image.MaskedImageF`
        Image of the extended PSF model.
    region_detectors : `lsst.pipe.tasks.extended_psf.DetectorsInRegion`
        List of detector IDs that define the focal plane region over which this
        extended PSF model has been built (and can be used).
    """

    extended_psf_image: MaskedImageF
    region_detectors: DetectorsInRegion


class ExtendedPsf:
    """Extended PSF model.

    Each instance may contain a default extended PSF, a set of extended PSFs
    that correspond to different focal plane regions, or both. At this time,
    focal plane regions are always defined as a subset of detectors.

    Parameters
    ----------
    default_extended_psf : `lsst.afw.image.MaskedImageF`
        Extended PSF model to be used as default (or only) extended PSF model.
    """

    def __init__(self, default_extended_psf=None):
        self.default_extended_psf = default_extended_psf
        self.focal_plane_regions = {}
        self.detectors_focal_plane_regions = {}

    def add_regional_extended_psf(self, extended_psf_image, region_name, region_detectors):
        """Add a new focal plane region, along with its extended PSF, to the
        ExtendedPsf instance.

        Parameters
        ----------
        extended_psf_image : `lsst.afw.image.MaskedImageF`
            Extended PSF model for the region.
        region_name : `str`
            Name of the focal plane region. Will be converted to all-uppercase.
        region_detectors : `lsst.pipe.tasks.extended_psf.DetectorsInRegion`
            List of detector IDs for the detectors that define a region on the
            focal plane.
        """
        region_name = region_name.upper()
        if region_name in self.focal_plane_regions:
            raise ValueError(f"Region name {region_name} is already used by this ExtendedPsf instance.")
        self.focal_plane_regions[region_name] = FocalPlaneRegionExtendedPsf(
            extended_psf_image=extended_psf_image, region_detectors=region_detectors
        )
        self.detectors_focal_plane_regions[region_name] = region_detectors

    def __call__(self, detector=None):
        """Return the appropriate extended PSF.

        If the instance contains no extended PSF defined over focal plane
        regions, the default extended PSF will be returned regardless of
        whether a detector ID was passed as argument.

        Parameters
        ----------
        detector : `int`, optional
            Detector ID. If focal plane region PSFs are defined, is used to
            determine which model to return.

        Returns
        -------
        extendedPsfImage : `lsst.afw.image.MaskedImageF`
            The extended PSF model. If this instance contains extended PSFs
            defined over focal plane regions, the extended PSF model for the
            region that contains ``detector`` is returned. If not, the default
            extended PSF is returned.
        """
        if detector is None:
            if self.default_extended_psf is None:
                raise ValueError("No default extended PSF available; please provide detector number.")
            return self.default_extended_psf
        elif not self.focal_plane_regions:
            return self.default_extended_psf
        return self.get_extended_psf(region_name=detector)

    def __len__(self):
        """Returns the number of extended PSF models present in the instance.

        Note that if the instance contains both a default model and a set of
        focal plane region models, the length of the instance will be the
        number of regional models, plus one (the default). This is true even
        in the case where the default model is one of the focal plane
        region-specific models.
        """
        n_regions = len(self.focal_plane_regions)
        if self.default_extended_psf is not None:
            n_regions += 1
        return n_regions

    def get_extended_psf(self, region_name):
        """Returns the extended PSF for a focal plane region or detector.

        This method takes either a region name or a detector ID as input. If
        the input is a `str` type, it is assumed to be the region name and if
        the input is a `int` type it is assumed to be the detector ID.

        Parameters
        ----------
        region_name : `str` or `int`
            Name of the region (str) or detector (int) for which the extended
            PSF should be retrieved.

        Returns
        -------
        extended_psf_image: `lsst.afw.image.MaskedImageF`
            The extended PSF model for the requested region or detector.

        Raises
        ------
        ValueError
            Raised if the input is not in the correct type.
        """
        if isinstance(region_name, str):
            return self.focal_plane_regions[region_name].extended_psf_image
        elif isinstance(region_name, int):
            region_name = find_region_for_detector(region_name, self.detectors_focal_plane_regions)
            return self.focal_plane_regions[region_name].extended_psf_image
        else:
            raise ValueError("A region name with `str` type or detector number with `int` must be provided")

    def write_fits(self, filename):
        """Write this object to a file.

        Parameters
        ----------
        filename : `str`
            Name of file to write.
        """
        # Create primary HDU with global metadata.
        metadata = PropertyList()
        metadata["HAS_DEFAULT"] = self.default_extended_psf is not None
        if self.focal_plane_regions:
            metadata["HAS_REGIONS"] = True
            metadata["REGION_NAMES"] = list(self.focal_plane_regions.keys())
            for region, e_psf_region in self.focal_plane_regions.items():
                metadata[region] = e_psf_region.region_detectors.detectors
        else:
            metadata["HAS_REGIONS"] = False
        fits_primary = Fits(filename, "w")
        fits_primary.createEmpty()
        fits_primary.writeMetadata(metadata)
        fits_primary.closeFile()
        # Write default extended PSF.
        if self.default_extended_psf is not None:
            default_hdu_metadata = PropertyList()
            default_hdu_metadata.update({"REGION": "DEFAULT", "EXTNAME": "IMAGE"})
            self.default_extended_psf.image.writeFits(filename, metadata=default_hdu_metadata, mode="a")
            default_hdu_metadata.update({"REGION": "DEFAULT", "EXTNAME": "MASK"})
            self.default_extended_psf.mask.writeFits(filename, metadata=default_hdu_metadata, mode="a")
        # Write extended PSF for each focal plane region.
        for j, (region, e_psf_region) in enumerate(self.focal_plane_regions.items()):
            metadata = PropertyList()
            metadata.update({"REGION": region, "EXTNAME": "IMAGE"})
            e_psf_region.extended_psf_image.image.writeFits(filename, metadata=metadata, mode="a")
            metadata.update({"REGION": region, "EXTNAME": "MASK"})
            e_psf_region.extended_psf_image.mask.writeFits(filename, metadata=metadata, mode="a")

    def writeFits(self, filename):
        """Alias for ``write_fits``; for compatibility with the Butler."""
        self.write_fits(filename)

    @classmethod
    def read_fits(cls, filename):
        """Build an instance of this class from a file.

        Parameters
        ----------
        filename : `str`
            Name of the file to read.
        """
        # Extract info from metadata.
        global_metadata = readMetadata(filename, hdu=0)
        has_default = global_metadata.getBool("HAS_DEFAULT")
        if global_metadata.getBool("HAS_REGIONS"):
            focal_plane_region_names = global_metadata.getArray("REGION_NAMES")
        else:
            focal_plane_region_names = []
        f = Fits(filename, "r")
        n_extensions = f.countHdus()
        extended_psf_parts = {}
        for j in range(1, n_extensions):
            md = readMetadata(filename, hdu=j)
            if has_default and md["REGION"] == "DEFAULT":
                if md["EXTNAME"] == "IMAGE":
                    default_image = ImageF(filename, hdu=j)
                elif md["EXTNAME"] == "MASK":
                    default_mask = MaskX(filename, hdu=j)
                continue
            if md["EXTNAME"] == "IMAGE":
                extended_psf_part = ImageF(filename, hdu=j)
            elif md["EXTNAME"] == "MASK":
                extended_psf_part = MaskX(filename, hdu=j)
            extended_psf_parts.setdefault(md["REGION"], {})[md["EXTNAME"].lower()] = extended_psf_part
        # Handle default if present.
        if has_default:
            extended_psf = cls(MaskedImageF(default_image, default_mask))
        else:
            extended_psf = cls()
        # Ensure we recovered an extended PSF for all focal plane regions.
        if len(extended_psf_parts) != len(focal_plane_region_names):
            raise ValueError(
                f"Number of per-region extended PSFs read ({len(extended_psf_parts)}) does not "
                "match with the number of regions recorded in the metadata "
                f"({len(focal_plane_region_names)})."
            )
        # Generate extended PSF regions mappings.
        for r_name in focal_plane_region_names:
            extended_psf_image = MaskedImageF(**extended_psf_parts[r_name])
            region_detectors = DetectorsInRegion()
            region_detectors.detectors = global_metadata.getArray(r_name)
            extended_psf.add_regional_extended_psf(extended_psf_image, r_name, region_detectors)
        # Instantiate ExtendedPsf.
        return extended_psf

    @classmethod
    def readFits(cls, filename):
        """Alias for ``readFits``; exists for compatibility with the Butler."""
        return cls.read_fits(filename)
        """Alias for ``readFits``; exists for compatibility with the Butler."""
        return cls.read_fits(filename)


class StackBrightStarsConfig(Config):
    """Configuration parameters for StackBrightStarsTask."""

    subregion_size = ListField[int](
        doc="Size, in pixels, of the subregions over which the stacking will be " "iteratively performed.",
        default=(100, 100),
    )
    stacking_statistic = ChoiceField[str](
        doc="Type of statistic to use for stacking.",
        default="MEANCLIP",
        allowed={
            "MEAN": "mean",
            "MEDIAN": "median",
            "MEANCLIP": "clipped mean",
        },
    )
    num_sigma_clip = Field[float](
        doc="Sigma for outlier rejection; ignored if stacking_statistic != 'MEANCLIP'.",
        default=4,
    )
    num_iter = Field[int](
        doc="Number of iterations of outlier rejection; ignored if stackingStatistic != 'MEANCLIP'.",
        default=3,
    )
    bad_mask_planes = ListField[str](
        doc="Mask planes that define pixels to be excluded from the stacking of the bright star stamps.",
        default=("BAD", "CR", "CROSSTALK", "EDGE", "NO_DATA", "SAT", "SUSPECT", "UNMASKEDNAN"),
    )
    do_mag_cut = Field[bool](
        doc="Apply magnitude cut before stacking?",
        default=False,
    )
    mag_limit = Field[float](
        doc="Magnitude limit, in Gaia G; all stars brighter than this value will be stacked",
        default=18,
    )


class StackBrightStarsTask(Task):
    """Stack bright stars together to build an extended PSF model."""

    ConfigClass = StackBrightStarsConfig
    _DefaultName = "stack_bright_stars"

    def _set_up_stacking(self, example_stamp):
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

    def run(self, bss_ref_list, region_name=None):
        """Read input bright star stamps and stack them together.

        The stacking is done iteratively over smaller areas of the final model
        image to allow for a great number of bright star stamps to be used.

        Parameters
        ----------
        bss_ref_list : `list` of
                `lsst.daf.butler._deferredDatasetHandle.DeferredDatasetHandle`
            List of available bright star stamps data references.
        region_name : `str`, optional
            Name of the focal plane region, if applicable. Only used for
            logging purposes, when running over multiple such regions
            (typically from `MeasureExtendedPsfTask`)
        """
        if region_name:
            region_message = f" for region '{region_name}'."
        else:
            region_message = "."
        self.log.info(
            "Building extended PSF from stamps extracted from %d detector images%s",
            len(bss_ref_list),
            region_message,
        )
        preparedStampsDict = self.prepareStamps(bss_ref_list)
        self.performStacking(preparedStampsDict)
        extendedPsfsDict = {}
        for binTag in preparedStampsDict.keys():
            extendedPsfsDict[binTag] = {}
            extendedPsfsDict[binTag]['stackedImage'] = preparedStampsDict[binTag]['stackedImage']

        # # read in example set of full stamps
        # example_bss = bss_ref_list[0].get()
        # example_stamp = example_bss[0].stamp_im
        # # creating groups of stamps based on their magnitude bins
        # # create model image
        # ext_psf = MaskedImageF(example_stamp.getBBox())
        # # divide model image into smaller subregions
        # subregion_size = Extent2I(*self.config.subregion_size)
        # sub_bboxes = subBBoxIter(ext_psf.getBBox(), subregion_size)
        # # compute approximate number of subregions
        # n_subregions = ((ext_psf.getDimensions()[0]) // (subregion_size[0] + 1)) * (
        #     (ext_psf.getDimensions()[1]) // (subregion_size[1] + 1)
        # )
        # self.log.info(
        #     "Stacking performed iteratively over approximately %d smaller areas of the final model image.",
        #     n_subregions,
        # )
        # set up stacking statistic
        # stats_control, stats_flags = self._set_up_stacking(example_stamp)
        # # perform stacking
        # for jbbox, bbox in enumerate(sub_bboxes):
        #     all_stars = None
        #     for i, bss_ref in enumerate(bss_ref_list):
        #         read_stars = bss_ref.get(parameters={"bbox": bbox})
        #         if self.config.do_mag_cut:
        #             read_stars = read_stars.selectByMag(magMax=self.config.mag_limit)
        #         if all_stars:
        #             all_stars.extend(read_stars)
        #         else:
        #             all_stars = read_stars
        #     # TODO: DM-27371 add weights to bright stars for stacking
        #     coadd_sub_bbox = statisticsStack(all_stars.getMaskedImages(), stats_flags, stats_control)
        #     ext_psf.assign(coadd_sub_bbox, bbox)
        # return ext_psf
        return extendedPsfsDict

    def performStacking(self, preparedStampsDict):
        for binTag in preparedStampsDict.keys():
            self.log.info(
                "Stacking performed iteratively over approximately %d smaller areas of the model image for "
                "stars in %s magnitude bin.",
                preparedStampsDict[binTag]["NSubRegions"],
                binTag,
            )
            stackedImage = statisticsStack(preparedStampsDict[binTag]['stamps'], preparedStampsDict[binTag]["statsFlags"], preparedStampsDict[binTag]["statsControl"])
            preparedStampsDict[binTag]['stackedImage'].assign(stackedImage)

    def prepareStamps(self, bss_ref_list):
        preparedStampsDict = self.groupStamps(bss_ref_list)

        # Reading the size of the subregions from the config file
        # Should we have one subregion for each group of stamps or one
        # subregion for each stamp?
        subregion_size = Extent2I(*self.config.subregion_size)
        for binTag in preparedStampsDict.keys():
            # create model image
            preparedStampsDict[binTag]['stackedImage'] = MaskedImageF(preparedStampsDict[binTag]['stamps'][0].getBBox())
            self.generateSubBboxes(preparedStampsDict[binTag], subregion_size)
            self.setUpStats(preparedStampsDict[binTag])
        return preparedStampsDict

    def groupStamps(self, bss_ref_list):
        """Group bright star stamps by magnitude bins.

        Parameters
        ----------
        bss_ref_list : `list` of
                `lsst.daf.butler._deferredDatasetHandle.DeferredDatasetHandle`
            List of available bright star stamps data references.

        Returns
        -------
        groupedStampsDict :
            `dict` [`str`,
                    `lsst.pipe.tasks.processBrightStars.BrightStarStamp`]
            Dictionary containing the magnitude bins tags as keys and list of
            bright stars stamps for each magnitude bin as the value.
        """
        groupedStampsDict = {}
        for bss_ref in bss_ref_list:
            bss = bss_ref.get()
            for stamp in bss:
                binTag = stamp.binTag
                if binTag not in groupedStampsDict.keys():
                    groupedStampsDict[binTag] = {}
                    groupedStampsDict[binTag]["stamps"] = [stamp.stamp_im]
                else:
                    groupedStampsDict[binTag]["stamps"].append(stamp.stamp_im)
        return groupedStampsDict

    def generateSubBboxes(self, magBinStampsDict, subregion_size):
        """Generate subregions for each group of stamps.

        Parameters
        ----------
        groupedStampsDict : `dict` of `list` of `lsst.pipe.tasks.processBrightStars.BrightStarStamp`
            List of lists of bright star stamps, grouped by detector.
        Returns
        -------
        subBboxesDict : `dict` of `list` of `lsst.geom.Box2I`
            List of lists of subregions for each group of stamps.
        """
        # divide model image into smaller subregions
        # subBboxes =
        magBinStampsDict["subBboxes"] = subBBoxIter(magBinStampsDict['stackedImage'].getBBox(), subregion_size)
        magBinStampsDict["NSubRegions"] = ((magBinStampsDict['stackedImage'].getDimensions()[0]) // (subregion_size[0] + 1)) * (
            (magBinStampsDict['stackedImage'].getDimensions()[1]) // (subregion_size[1] + 1)
        )

    def setUpStats(self, magBinStampsDict):
        stats_control, stats_flags = self._set_up_stacking(magBinStampsDict['stamps'][0])
        magBinStampsDict["statsControl"] = stats_control
        magBinStampsDict["statsFlags"] = stats_flags


class MeasureExtendedPsfConnections(PipelineTaskConnections, dimensions=("band", "instrument")):
    input_brightStarStamps = Input(
        doc="Input list of bright star collections to be stacked.",
        name="brightStarStamps",
        storageClass="BrightStarStamps",
        dimensions=("visit", "detector"),
        deferLoad=True,
        multiple=True,
    )
    extended_psf = Output(
        doc="Extended PSF model built by stacking bright stars.",
        name="extended_psf",
        storageClass="ExtendedPsf",
        dimensions=("band",),
    )
    # stacked_stamps = Output(
    #     doc="Extended PSF model built by stacking bright stars.",
    #     name="stacked_stamps",
    #     storageClass="StackedStamps",
    #     dimensions=("band",),
    # )


class AnnulusRadii(Config):
    """Provides a list containing inner and outer radius for a scaling annulus."""

    radii = ListField[int](
        doc="A list containing the inner and outer radii of the scaling annulus.",
        default=[],
    )


class MeasureExtendedPsfConfig(PipelineTaskConfig, pipelineConnections=MeasureExtendedPsfConnections):
    """Configuration parameters for MeasureExtendedPsfTask."""

    stack_bright_stars = ConfigurableField(
        target=StackBrightStarsTask,
        doc="Stack selected bright stars",
    )
    detectors_focal_plane_regions = ConfigDictField(
        keytype=str,
        itemtype=DetectorsInRegion,
        doc=(
            "Mapping from focal plane region names to detector IDs. "
            "If empty, a constant extended PSF model is built from all selected bright stars. "
            "It's possible for a single detector to be included in multiple regions if so desired."
        ),
        default={},
    )
    scalingAnnuli = ConfigDictField(
        keytype=str,
        itemtype=AnnulusRadii,
        doc="The inner and outer radii for the scaling annuli.",
        default={},
    )
    badMaskPlanes = ListField[str](
        doc="Mask planes that, if set, lead to associated pixels not being included in the computation of "
        "the scaling factor (`BAD` should always be included). Ignored if scalingType is `annularFlux`, "
        "as the stamps are expected to already be normalized.",
        # Note that `BAD` should always be included, as secondary detected
        # sources (i.e., detected sources other than the primary source of
        # interest) also get set to `BAD`.
        default=("BAD", "CR", "CROSSTALK", "EDGE", "NO_DATA", "SAT", "SUSPECT", "UNMASKEDNAN"),
    )


class MeasureExtendedPsfTask(Task):
    """Build and save extended PSF model.

    The model is built by stacking bright star stamps, extracted and
    preprocessed by
    `lsst.pipe.tasks.processBrightStars.ProcessBrightStarsTask`.

    If a mapping from detector IDs to focal plane regions is provided, a
    different extended PSF model will be built for each focal plane region. If
    not, a single constant extended PSF model is built with all available data.
    """

    ConfigClass = MeasureExtendedPsfConfig
    _DefaultName = "measureExtendedPsf"

    def __init__(self, initInputs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("stack_bright_stars")
        self.detectors_focal_plane_regions = self.config.detectors_focal_plane_regions
        if not self.config.scalingAnnuli:
            self.setScalingAnnularRadii()
        else:
            self.scalingAnnuli = self.config.scalingAnnuli
        self.regionless_dets = []

    def select_detector_refs(self, ref_list):
        """Split available sets of bright star stamps according to focal plane
        regions.

        Parameters
        ----------
        ref_list : `list` of
                `lsst.daf.butler._deferredDatasetHandle.DeferredDatasetHandle`
            List of available bright star stamps data references.
        """
        region_ref_list = {region: [] for region in self.detectors_focal_plane_regions.keys()}
        for dataset_handle in ref_list:
            detector_id = dataset_handle.ref.dataId["detector"]
            if detector_id in self.regionless_dets:
                continue
            try:
                region_name = find_region_for_detector(detector_id, self.detectors_focal_plane_regions)
            except KeyError:
                self.log.warning(
                    "Bright stars were available for detector %d, but it was missing from the %s config "
                    "field, so they will not be used to build any of the extended PSF models.",
                    detector_id,
                    "'detectors_focal_plane_regions'",
                )
                self.regionless_dets.append(detector_id)
                continue
            region_ref_list[region_name].append(dataset_handle)
        return region_ref_list

    def scaleStackedStamps(self, stackedStamps):
        binTags = [binTag for binTag in stackedStamps.keys()]
        jointPSF = None
        # self.scaleFactors = {}
        # annuliTags = [annulusTag for annulusTag in self.scalingAnnuli.keys()]
        for i, key in enumerate(self.scalingAnnuli.keys()):
            if i < len(binTags) -1:
                # make cutout from the larger stacked image
                innImage = stackedStamps[binTags[i]]['stackedImage'].clone()
                outImage = stackedStamps[binTags[i+1]]['stackedImage'].clone()
                BBox = innImage.getBBox()
                stampSize = [BBox.getWidth(), BBox.getHeight()]
                xy0, cutout = self._getCutout(outImage, stampSize)
                # make annulus for :stackedStamps[binTags[i+1]]
                innerImage = self.createMaskedImage(innImage, self.scalingAnnuli[key].radii, key=key, binTag=binTags[i])
                outerImage = self.createMaskedImage(cutout, self.scalingAnnuli[key].radii, key=key, binTag=binTags[i+1])
                # make annulus for :stackedStamps[binTags[i]]
                # find the scaling factor (includes _setUpStatistics)
                scaleFactor = self.findScalingFactor(outerImage, innerImage)
                # scale stacked image[i] and stich it into stacked image[i+1]
                # self.scaleFactors[key] = scaleFactor
                if jointPSF is None:
                    jointPSF = self.joinStackedStamps(
                        stackedStamps[binTags[i]]['stackedImage'],
                        stackedStamps[binTags[i+1]]['stackedImage'],
                        self.scalingAnnuli[key].radii,
                        xy0,
                        scaleFactor,
                        key)
                else:
                    jointPSF = self.joinStackedStamps(
                        jointPSF,
                        stackedStamps[binTags[i+1]]['stackedImage'],
                        self.scalingAnnuli[key].radii,
                        xy0,
                        scaleFactor,
                        key)
            else:
                # TODO: If there is no stamp for a given magnitude bin, we need to make sure the right way of joining is implemented. For example, what should happend if a bin which is not the faintest or brightest bin has no stamp?
                print("One or more magnitude bins are not represented by stamps!")

        return jointPSF
    
    #     annulus1
    #     annulus2
    #     .
    #     .
    #     .
    #     annulusNStiches

    #     annulus1 => scale the inner to the level of the outer
    #     annulus2 => scale the inner to the level of the outer

    #     three possible stitching:
    #         1. keep the inner annulus
    #         2. keep the outer annulus
    #         3. find the mean of the two annuli
    #     NStitches = len(stackedStamps.keys()) - 1
    #     for i in range(NStitches):
    #         print(NStitches)

    def findScalingFactor(self, outerImage, innerImage):
        """Find the scaling factor between two annuli.

        Parameters
        ----------
        outerImage : `lsst.afw.image.MaskedImageF`
            The masked stacked image of the brighter end of the join-bin.
        innerImage : `lsst.afw.image.MaskedImageF`
            The masked stacked image of the fainter end of the join-bin.

        Returns
        -------
        scalingFactor : `float`
            The scaling factor for the fainter stacked image.
        """
        # This needs more thought! Is there a way to include the mask planes
        # from both stacked images? or maybe we do not need to include the
        # maske planes at all?
        self._setUpStatistics(outerImage.mask)
        xy = outerImage.clone()
        xy.image.array *= innerImage.image.array
        xx = innerImage.clone()
        xx.image.array = innerImage.image.array**2
        xySum = makeStatistics(xy, self.statsFlag, self.statsControl).getValue()
        xxSum = makeStatistics(xx, self.statsFlag, self.statsControl).getValue()
        scalingFactor = xySum / xxSum if xxSum else 1
        return scalingFactor
    
    def createMaskedImage(self, image, annulusRadii, key=None, binTag=None):
        imCenter = image.image.getBBox().getCenter()
        imCenter = int(imCenter[0]), int(imCenter[1])
        outerCircle = SpanSet.fromShape(annulusRadii[1], Stencil.CIRCLE, offset=imCenter)
        innerCircle = SpanSet.fromShape(annulusRadii[0], Stencil.CIRCLE, offset=imCenter)
        annulusWidth = annulusRadii[1] - annulusRadii[0]
        if annulusWidth < 1:
            raise ValueError("The annulus width must be greater than 1 pixel.")
        annulus = outerCircle.intersectNot(innerCircle)

        maskedImageSize = image.getDimensions()
        maskPlaneDict = image.mask.getMaskPlaneDict()
        annulusImage = MaskedImageF(maskedImageSize, planeDict=maskPlaneDict)
        annulusImage.image.array[:] = np.nan

        # annulusMask = annulusImage.mask
        # annulusMask.array[:] = 2 ** maskPlaneDict["NO_DATA"]
        annulus.copyMaskedImage(image, annulusImage)
        # from astropy.io import fits
        # output_fits_file = "example_images/" + key + binTag + '.fits'
        # # Create a Primary HDU with the data
        # primary_hdu = fits.PrimaryHDU(annulusImage.image.array)
        # # Create an HDU list
        # hdul = fits.HDUList([primary_hdu])
        # # Write the HDU list to the FITS file
        # hdul.writeto(output_fits_file, overwrite=True)
        return annulusImage

    def _getCutout(self, inputImage, stampSize: list[int]):
        # TODO: Replace this method with exposure getCutout after DM-40042.
        BBox = inputImage.getBBox()
        # coordPix = np.array(refImageBBox.getCenter())
        corner = Point2I(np.array(BBox.getCenter()) - np.array(stampSize) / 2)
        dimensions = Extent2I(stampSize)
        stampBBox = Box2I(corner, dimensions)
        xy0 = [stampBBox.beginX, stampBBox.beginY]
        overlapBBox = Box2I(stampBBox)
        overlapBBox.clip(inputImage.getBBox())
        # if overlapBBox.getArea() > 0:
            # Create full-sized stamp with pixels initially flagged as NO_DATA.
        stamp = MaskedImageF(bbox=stampBBox)
        stamp.image[:] = np.nan
        stamp.mask.set(inputImage.mask.getPlaneBitMask("NO_DATA"))
        # Restore pixels which overlap the input exposure.
        inputMI = inputImage.image
        overlap = inputMI.Factory(inputMI, overlapBBox)
        stamp.image[overlapBBox] = overlap
        stamp.setXY0(0, 0)
        # Set detector and WCS.
        # stamp.setDetector(inputExposure.getDetector())
        # stamp.setWcs(inputExposure.getWcs())
        # else:
        #     stamp = None
        return xy0, stamp

    def writetoFits(self, image, filename):
        from astropy.io import fits
        # Create a Primary HDU with the data
        primary_hdu = fits.PrimaryHDU(image)
        # Create an HDU list
        hdul = fits.HDUList([primary_hdu])
        # Write the HDU list to the FITS file
        hdul.writeto(filename, overwrite=True)
    
    def _setUpStatistics(self, exampleMask):
    # def _setUpStatistics(self, exampleMask=stackedStamps['central']['stackedImage'].mask):
        """Configure statistics control and flag, for use if ``scalingType`` is
        `leastSquare`.
        """
        # if self.config.scalingType == "leastSquare":
        # Set the mask planes which will be ignored.
        andMask = reduce(
            ior,
            (exampleMask.getPlaneBitMask(bm) for bm in self.config.badMaskPlanes),
        )
        self.statsControl = StatisticsControl(
            # andMask=andMask,
        )
        self.statsFlag = stringToStatisticsProperty("SUM")
            
    def setScalingAnnularRadii(self):
        """Set default scaling annular flux radii."""
        self.scalinTags = ["200", "201", "202"]
        self.scalingAnnuli = {}
        # radii = [[12, 20], [23, 33], [90, 105]]
        radii = [[12, 22], [70, 85], [150, 200]]
        for i, key in enumerate(self.scalinTags):
            annulusRadii = AnnulusRadii()
            annulusRadii.radii = radii[i]
            self.scalingAnnuli[key] = annulusRadii

    def joinStackedStamps(self, innerImage, outerImage, raddi, xy0, scaleFactor, key):
        # Scaling the inner image using the scaling factor:
        innerImage.image.array *= scaleFactor
        innerImage.setXY0(xy0[0], xy0[1])
        imCenter = innerImage.image.getBBox().getCenter()
        imCenter = int(imCenter[0]), int(imCenter[1])
        innerCircle = SpanSet.fromShape(raddi[0], Stencil.CIRCLE, offset=imCenter)
        maskedImageSize = outerImage.getDimensions()
        maskPlaneDict = outerImage.mask.getMaskPlaneDict()
        jointImage = MaskedImageF(maskedImageSize, planeDict=maskPlaneDict)
        jointImage.image.array[:] = outerImage.image.array[:]

        # annulusMask = annulusImage.mask
        # annulusMask.array[:] = 2 ** maskPlaneDict["NO_DATA"]
        innerCircle.copyMaskedImage(innerImage, jointImage)
        filename = key + 'finaljoint.fits'
        self.writetoFits(jointImage.image.array, filename)
        return jointImage
    
    def stitchModels(self, stackedStamps, i, stitched=None):
        for binTag in stackedStamps.keys():

            if stitched is None:
                stitched = stackedStamps[i]['stackedImage']
        else:
            stitched = self.stitch(stackedStamps[i]['stackedImage'], stitched)
        return stitched
    
    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        input_data = butlerQC.get(inputRefs)
        bss_ref_list = input_data["input_brightStarStamps"]
        output_e_psf = self.run(bss_ref_list)
        output = Struct(extended_psf=output_e_psf)
        butlerQC.put(output, outputRefs)
    def run(self, bss_ref_list):
        if not self.config.detectors_focal_plane_regions:
            self.log.info(
                "No detector groups were provided to MeasureExtendedPsfTask; computing a single, "
                "constant extended PSF model over all available observations."
            )
            # output_e_psf = [ExtendedPsf(stackedImage) for stackedImage in self.stack_bright_stars.run(bss_ref_list)]
            stackedStamps = self.stack_bright_stars.run(bss_ref_list)
            scaledStackedStamps = self.scaleStackedStamps(stackedStamps)
            # TO DO: the final product should also carry the binTag or image size of the largest stamp.
            output_e_psf = ExtendedPsf(scaledStackedStamps)
        else:
            stackedStamps = ExtendedPsf()
            region_ref_list = self.select_detector_refs(bss_ref_list)
            for region_name, ref_list in region_ref_list.items():
                if not ref_list:
                    # no valid references found
                    self.log.warning(
                        "No valid brightStarStamps reference found for region '%s'; skipping it.",
                        region_name,
                    )
                    continue
                ext_psf = self.stack_bright_stars.run(ref_list, region_name)
                stackedStamps.add_regional_extended_psf(
                    ext_psf, region_name, self.detectors_focal_plane_regions[region_name]
                )
        return output_e_psf
