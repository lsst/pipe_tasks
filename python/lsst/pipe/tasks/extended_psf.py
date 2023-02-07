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

"""Read preprocessed bright stars and stack them to build an extended
PSF model.
"""

__all__ = ["FocalPlaneRegionExtendedPsf", "ExtendedPsf", "StackBrightStarsConfig",
           "StackBrightStarsTask", "MeasureExtendedPsfConfig", "MeasureExtendedPsfTask"]

from dataclasses import dataclass
from typing import List

from lsst.afw import image as afwImage
from lsst.afw import fits as afwFits
from lsst.afw import math as afwMath
from lsst.daf.base import PropertyList
from lsst.pipe import base as pipeBase
from lsst.pipe.tasks.assembleCoadd import AssembleCoaddTask
import lsst.pex.config as pexConfig
from lsst.geom import Extent2I


@dataclass
class FocalPlaneRegionExtendedPsf:
    """Single extended PSF over a focal plane region.

    The focal plane region is defined through a list
    of detectors.

    Parameters
    ----------
    extended_psf_image : `lsst.afw.image.MaskedImageF`
        Image of the extended PSF model.
    detector_list : `list` [`int`]
        List of detector IDs that define the focal plane region over which this
        extended PSF model has been built (and can be used).
    """
    extended_psf_image: afwImage.MaskedImageF
    detector_list: List[int]


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

    def add_regional_extended_psf(self, extended_psf_image, region_name, detector_list):
        """Add a new focal plane region, along wit hits extended PSF, to the
        ExtendedPsf instance.

        Parameters
        ----------
        extended_psf_image : `lsst.afw.image.MaskedImageF`
            Extended PSF model for the region.
        region_name : `str`
            Name of the focal plane region. Will be converted to all-uppercase.
        detector_list : `list` [`int`]
            List of IDs for the detectors that define the focal plane region.
        """
        region_name = region_name.upper()
        if region_name in self.focal_plane_regions:
            raise ValueError(f"Region name {region_name} is already used by this ExtendedPsf instance.")
        self.focal_plane_regions[region_name] = FocalPlaneRegionExtendedPsf(
            extended_psf_image=extended_psf_image, detector_list=detector_list)
        for det in detector_list:
            self.detectors_focal_plane_regions[det] = region_name

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
        return self.get_regional_extended_psf(detector=detector)

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

    def get_regional_extended_psf(self, region_name=None, detector=None):
        """Returns the extended PSF for a focal plane region.

        The region can be identified either by name, or through a detector ID.

        Parameters
        ----------
        region_name : `str` or `None`, optional
            Name of the region for which the extended PSF should be retrieved.
            Ignored if  ``detector`` is provided. Must be provided if
            ``detector`` is None.
        detector : `int` or `None`, optional
            If provided, returns the extended PSF for the focal plane region
            that includes this detector.

        Raises
        ------
        ValueError
            Raised if neither ``detector`` nor ``regionName`` is provided.
        """
        if detector is None:
            if region_name is None:
                raise ValueError("One of either a regionName or a detector number must be provided.")
            return self.focal_plane_regions[region_name].extended_psf_image
        return self.focal_plane_regions[self.detectors_focal_plane_regions[detector]].extended_psf_image

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
                metadata[region] = e_psf_region.detector_list
        else:
            metadata["HAS_REGIONS"] = False
        fits_primary = afwFits.Fits(filename, "w")
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
        """Alias for ``write_fits``; exists for compatibility with the Butler.
        """
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
        global_metadata = afwFits.readMetadata(filename, hdu=0)
        has_default = global_metadata.getBool("HAS_DEFAULT")
        if global_metadata.getBool("HAS_REGIONS"):
            focal_plane_region_names = global_metadata.getArray("REGION_NAMES")
        else:
            focal_plane_region_names = []
        f = afwFits.Fits(filename, "r")
        n_extensions = f.countHdus()
        extended_psf_parts = {}
        for j in range(1, n_extensions):
            md = afwFits.readMetadata(filename, hdu=j)
            if has_default and md["REGION"] == "DEFAULT":
                if md["EXTNAME"] == "IMAGE":
                    default_image = afwImage.ImageF(filename, hdu=j)
                elif md["EXTNAME"] == "MASK":
                    default_mask = afwImage.MaskX(filename, hdu=j)
                continue
            if md["EXTNAME"] == "IMAGE":
                extended_psf_part = afwImage.ImageF(filename, hdu=j)
            elif md["EXTNAME"] == "MASK":
                extended_psf_part = afwImage.MaskX(filename, hdu=j)
            extended_psf_parts.setdefault(md["REGION"], {})[md["EXTNAME"].lower()] = extended_psf_part
        # Handle default if present.
        if has_default:
            extended_psf = cls(afwImage.MaskedImageF(default_image, default_mask))
        else:
            extended_psf = cls()
        # Ensure we recovered an extended PSF for all focal plane regions.
        if len(extended_psf_parts) != len(focal_plane_region_names):
            raise ValueError(f"Number of per-region extended PSFs read ({len(extended_psf_parts)}) does not "
                             "match with the number of regions recorded in the metadata "
                             f"({len(focal_plane_region_names)}).")
        # Generate extended PSF regions mappings.
        for r_name in focal_plane_region_names:
            extended_psf_image = afwImage.MaskedImageF(**extended_psf_parts[r_name])
            detector_list = global_metadata.getArray(r_name)
            extended_psf.add_regional_extended_psf(extended_psf_image, r_name, detector_list)
        # Instantiate ExtendedPsf.
        return extended_psf

    @classmethod
    def readFits(cls, filename):
        """Alias for ``readFits``; exists for compatibility with the Butler.
        """
        return cls.read_fits(filename)


class StackBrightStarsConfig(pexConfig.Config):
    """Configuration parameters for StackBrightStarsTask.
    """
    subregion_size = pexConfig.ListField(
        dtype=int,
        doc="Size, in pixels, of the subregions over which the stacking will be "
            "iteratively performed.",
        default=(100, 100)
    )
    stacking_statistic = pexConfig.ChoiceField(
        dtype=str,
        doc="Type of statistic to use for stacking.",
        default="MEANCLIP",
        allowed={
            "MEAN": "mean",
            "MEDIAN": "median",
            "MEANCLIP": "clipped mean",
        }
    )
    num_sigma_clip = pexConfig.Field(
        dtype=float,
        doc="Sigma for outlier rejection; ignored if stacking_statistic != 'MEANCLIP'.",
        default=4
    )
    num_iter = pexConfig.Field(
        dtype=int,
        doc="Number of iterations of outlier rejection; ignored if stackingStatistic != 'MEANCLIP'.",
        default=3
    )
    bad_mask_planes = pexConfig.ListField(
        dtype=str,
        doc="Mask planes that, if set, lead to associated pixels not being included in the stacking of the "
            "bright star stamps.",
        default=('BAD', 'CR', 'CROSSTALK', 'EDGE', 'NO_DATA', 'SAT', 'SUSPECT', 'UNMASKEDNAN')
    )
    do_mag_cut = pexConfig.Field(
        dtype=bool,
        doc="Apply magnitude cut before stacking?",
        default=False
    )
    mag_limit = pexConfig.Field(
        dtype=float,
        doc="Magnitude limit, in Gaia G; all stars brighter than this value will be stacked",
        default=18
    )


class StackBrightStarsTask(pipeBase.Task):
    """Stack bright stars together to build an extended PSF model.
    """
    ConfigClass = StackBrightStarsConfig
    _DefaultName = "stack_bright_stars"

    def _set_up_stacking(self, example_stamp):
        """Configure stacking statistic and control from config fields.
        """
        stats_control = afwMath.StatisticsControl()
        stats_control.setNumSigmaClip(self.config.num_sigma_clip)
        stats_control.setNumIter(self.config.num_iter)
        if bad_masks := self.config.bad_mask_planes:
            and_mask = example_stamp.mask.getPlaneBitMask(bad_masks[0])
            for bm in bad_masks[1:]:
                and_mask = and_mask | example_stamp.mask.getPlaneBitMask(bm)
            stats_control.setAndMask(and_mask)
        stats_flags = afwMath.stringToStatisticsProperty(self.config.stacking_statistic)
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
            region_message = f' for region "{region_name}".'
        else:
            region_message = ''
        self.log.info('Building extended PSF from stamps extracted from %d detector images%s',
                      len(bss_ref_list), region_message)
        # read in example set of full stamps
        example_bss = bss_ref_list[0].get()
        example_stamp = example_bss[0].stamp_im
        # create model image
        ext_psf = afwImage.MaskedImageF(example_stamp.getBBox())
        # divide model image into smaller subregions
        subregion_size = Extent2I(*self.config.subregion_size)
        sub_bboxes = AssembleCoaddTask._subBBoxIter(ext_psf.getBBox(), subregion_size)
        # compute approximate number of subregions
        n_subregions = int(ext_psf.getDimensions()[0]/subregion_size[0] + 1)*int(
            ext_psf.getDimensions()[1]/subregion_size[1] + 1)
        self.log.info("Stacking will performed iteratively over approximately %d "
                      "smaller areas of the final model image.", n_subregions)
        # set up stacking statistic
        stats_control, stats_flags = self._set_up_stacking(example_stamp)
        # perform stacking
        for jbbox, bbox in enumerate(sub_bboxes):
            all_stars = None
            for bss_ref in bss_ref_list:
                read_stars = bss_ref.get(parameters={'bbox': bbox})
                if self.config.do_mag_cut:
                    read_stars = read_stars.selectByMag(magMax=self.config.mag_limit)
                if all_stars:
                    all_stars.extend(read_stars)
                else:
                    all_stars = read_stars
            # TODO: DM-27371 add weights to bright stars for stacking
            coadd_sub_bbox = afwMath.statisticsStack(all_stars.getMaskedImages(), stats_flags, stats_control)
            ext_psf.assign(coadd_sub_bbox, bbox)
        return ext_psf


class MeasureExtendedPsfConnections(pipeBase.PipelineTaskConnections,
                                    dimensions=("band", "instrument")):
    input_brightStarStamps = pipeBase.connectionTypes.Input(
        doc="Input list of bright star collections to be stacked.",
        name="brightStarStamps",
        storageClass="BrightStarStamps",
        dimensions=("visit", "detector"),
        deferLoad=True,
        multiple=True
    )
    extended_psf = pipeBase.connectionTypes.Output(
        doc="Extended PSF model built by stacking bright stars.",
        name="extended_psf",
        storageClass="ExtendedPsf",
        dimensions=("band",),
    )


class MeasureExtendedPsfConfig(pipeBase.PipelineTaskConfig,
                               pipelineConnections=MeasureExtendedPsfConnections):
    """Configuration parameters for MeasureExtendedPsfTask.
    """
    stack_bright_stars = pexConfig.ConfigurableField(
        target=StackBrightStarsTask,
        doc="Stack selected bright stars",
    )
    detectors_focal_plane_regions = pexConfig.DictField(
        keytype=int,
        itemtype=str,
        doc="Mapping from detector IDs to focal plane region names. If empty, a constant "
            "extended PSF model is built from all selected bright stars.",
        default={}
    )


class MeasureExtendedPsfTask(pipeBase.Task):
    """Build and save extended PSF model.

    The model is built by stacking bright star stamps, extracted and
    preprocessed by
    `lsst.pipe.tasks.processBrightStars.ProcessBrightStarsTask`.
    If a mapping from detector IDs to focal plane regions is provided,
    a different extended PSF model will be built for each focal plane
    region. If not, a single, constant extended PSF model is built using
    all available data.
    """
    ConfigClass = MeasureExtendedPsfConfig
    _DefaultName = "measureExtendedPsf"

    def __init__(self, initInputs=None, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)
        self.makeSubtask("stack_bright_stars")
        self.focal_plane_regions = {region: [] for region in
                                    set(self.config.detectors_focal_plane_regions.values())}
        for det, region in self.config.detectors_focal_plane_regions.items():
            self.focal_plane_regions[region].append(det)
        # make no assumption on what detector IDs should be, but if we come
        # across one where there are processed bright stars, but no
        # corresponding focal plane region, make sure we keep track of
        # it (eg to raise a warning only once)
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
        region_ref_list = {region: [] for region in self.focal_plane_regions.keys()}
        for dataset_handle in ref_list:
            det_id = dataset_handle.ref.dataId["detector"]
            if det_id in self.regionless_dets:
                continue
            try:
                region_name = self.config.detectors_focal_plane_regions[det_id]
            except KeyError:
                self.log.warning('Bright stars were available for detector %d, but it was missing '
                                 'from the "detectors_focal_plane_regions" config field, so they will not '
                                 'be used to build any of the extended PSF models', det_id)
                self.regionless_dets.append(det_id)
                continue
            region_ref_list[region_name].append(dataset_handle)
        return region_ref_list

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        input_data = butlerQC.get(inputRefs)
        bss_ref_list = input_data['input_brightStarStamps']
        # Handle default case of a single region with empty detector list
        if not self.config.detectors_focal_plane_regions:
            self.log.info("No detector groups were provided to MeasureExtendedPsfTask; computing a single, "
                          "constant extended PSF model over all available observations.")
            output_e_psf = ExtendedPsf(self.stack_bright_stars.run(bss_ref_list))
        else:
            output_e_psf = ExtendedPsf()
            region_ref_list = self.select_detector_refs(bss_ref_list)
            for region_name, ref_list in region_ref_list.items():
                if not ref_list:
                    # no valid references found
                    self.log.warning('No valid brightStarStamps reference found for region "%s"; '
                                     'skipping it.', region_name)
                    continue
                ext_psf = self.stack_bright_stars.run(ref_list, region_name)
                output_e_psf.add_regional_extended_psf(ext_psf, region_name,
                                                       self.focal_plane_regions[region_name])
        output = pipeBase.Struct(extended_psf=output_e_psf)
        butlerQC.put(output, outputRefs)
