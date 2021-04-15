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
#
"""Read preprocessed bright stars and stack them to build an extended
PSF model.
"""

from dataclasses import dataclass
from typing import List

from lsst.afw import image as afwImage
from lsst.afw import fits as afwFits
from lsst.daf.base import PropertyList


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
