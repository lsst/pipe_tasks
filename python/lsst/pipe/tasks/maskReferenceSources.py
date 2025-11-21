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

"""Pipeline for masking DIA sources based on a reference catalog
"""

__all__ = ["MaskReferenceSourcesTask",
           "MaskReferenceSourcesConfig",
           "MaskReferenceSourcesConnections"]

import numpy as np
import logging

from astropy.coordinates import SkyCoord
import astropy.units as u

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.meas.algorithms


class MaskReferenceSourcesConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "visit", "detector"),
):
    """Connections for MaskReferenceSources task."""

    # Input reference catalog
    astrometry_ref_cat = connectionTypes.PrerequisiteInput(
        doc="Reference catalog to use for source matching",
        name="gaia_dr3_20230707",
        storageClass="SimpleCatalog",
        dimensions=("skypix",),
        deferLoad=True,
        multiple=True,
    )

    # Input image for matching (default: science image)
    matching_image = connectionTypes.Input(
        doc="Image to use for reference source matching",
        name="calexp",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
    )

    # Input sources for matching (default: science sources)
    matching_sources = connectionTypes.Input(
        doc="Sources detected in matching image",
        name="src",
        storageClass="SourceCatalog",
        dimensions=("instrument", "visit", "detector"),
    )

    # Input difference image to mask
    difference_image = connectionTypes.Input(
        doc="Difference image to apply masks to",
        name="difference_image",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
    )

    # Output masked difference image
    masked_difference_image = connectionTypes.Output(
        doc="Difference image with reference source mask",
        name="difference_image_masked",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
    )


class MaskReferenceSourcesConfig(
    pipeBase.PipelineTaskConfig, pipelineConnections=MaskReferenceSourcesConnections
):
    """Configuration for MaskReferenceSources task."""

    matching_radius = pexConfig.Field(
        dtype=float,
        default=1.0,
        doc="Maximum separation for source matching in arcseconds",
    )
    reference_mag_column = pexConfig.Field(
        dtype=str,
        default="r_flux",
        doc="Column name for reference catalog flux (in nJy)",
    )
    reference_ra_column = pexConfig.Field(
        dtype=str,
        default="coord_ra",
        doc="Column name for reference catalog RA in degrees",
    )
    reference_dec_column = pexConfig.Field(
        dtype=str,
        default="coord_dec",
        doc="Column name for reference catalog Dec in degrees",
    )
    mask_plane_name = pexConfig.Field(
        dtype=str,
        default="REFERENCE",
        doc="Name of mask plane to create for reference source regions",
    )
    reference_buffer = pexConfig.Field(
        dtype=int,
        default=100,
        doc="Buffer in pixels to add around image bounds when loading reference catalog",
    )
    astrometry_ref_loader = pexConfig.ConfigField(
        dtype=lsst.meas.algorithms.LoadReferenceObjectsConfig,
        doc="Configuration of reference object loader for source matching",
    )

    def setDefaults(self):
        super().setDefaults()
        # Set to Gaii, but we don't use photometry currently
        self.astrometry_ref_loader.filterMap = {
            "u": "phot_g_mean",
            "g": "phot_g_mean",
            "r": "phot_g_mean",
            "i": "phot_g_mean",
            "z": "phot_g_mean",
            "y": "phot_g_mean",
        }


class MaskReferenceSourcesTask(pipeBase.PipelineTask):
    """Mask regions around reference sources in difference images."""

    ConfigClass = MaskReferenceSourcesConfig
    _DefaultName = "maskReferenceSources"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        """Run the task on quantum data."""
        inputs = butlerQC.get(inputRefs)

        # Create reference object loader following the standard pattern
        astrometry_loader = lsst.meas.algorithms.ReferenceObjectLoader(
            dataIds=[ref.datasetRef.dataId for ref in inputRefs.astrometry_ref_cat],
            refCats=inputs.pop("astrometry_ref_cat"),
            name=self.config.connections.astrometry_ref_cat,
            config=self.config.astrometry_ref_loader,
            log=self.log,
        )

        # Load reference catalog using loadPixelBox with buffer for edge sources
        original_bbox = inputs["matching_image"].getBBox()
        buffered_bbox = original_bbox.dilatedBy(self.config.reference_buffer)

        self.log.info(
            f"Loading reference catalog for bbox: {original_bbox} (buffered to {buffered_bbox})"
        )

        ref_result = astrometry_loader.loadPixelBox(
            bbox=buffered_bbox,
            wcs=inputs["matching_image"].getWcs(),
            filterName=inputs["matching_image"].getFilter().bandLabel,
        )

        self.log.info(f"Loaded {len(ref_result.refCat)} reference sources")

        outputs = self.run(
            ref_catalog=ref_result.refCat,
            difference_image=inputs["difference_image"],
            matching_sources=inputs["matching_sources"],
            matching_image=inputs["matching_image"],
        )

        butlerQC.put(outputs, outputRefs)

    def run(self, ref_catalog, difference_image, matching_sources, matching_image):
        """Mask reference sources in difference image.

        Parameters
        ----------
        ref_catalog : `lsst.afw.table.SourceCatalog`
            Reference catalog with sources to mask
        difference_image : `lsst.afw.image.ExposureF`
            Difference image to apply masks to
        matching_sources : `lsst.afw.table.SourceCatalog`
            Sources to match against reference catalog
        matching_image : `lsst.afw.image.ExposureF`
            Image where matching sources were detected

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results struct with:
            - ``masked_difference_image`` : Modified difference image with reference mask
        """
        self.log.info(f"Masking reference sources in difference image")
        self.log.info(f"Reference catalog has {len(ref_catalog)} sources")
        self.log.info(f"Matching against {len(matching_sources)} detected sources")

        # Create copy of difference image to modify
        masked_diff = difference_image.clone()

        # Add mask plane if it doesn't exist
        mask = masked_diff.mask
        try:
            mask_bit = mask.addMaskPlane(self.config.mask_plane_name)
            self.log.info(
                f"Added mask plane '{self.config.mask_plane_name}' with bit {mask_bit}"
            )
        except Exception:
            # Mask plane already exists
            mask_bit = mask.getPlaneBitMask(self.config.mask_plane_name)
            self.log.info(
                f"Using existing mask plane '{self.config.mask_plane_name}' with bit {mask_bit}"
            )

        # Match reference sources to the matching sources
        matches = self._matchSources(
            ref_catalog, matching_sources, matching_image.getWcs()
        )
        self.log.info(f"Found {len(matches)} matched sources")

        if len(matches) == 0:
            self.log.warn(f"No reference sources matched to detected sources")
            return pipeBase.Struct(masked_difference_image=masked_diff)

        # Apply masks for matched sources
        n_masked = self._applyReferenceMasks(matches, masked_diff)
        self.log.info(f"Masked {n_masked} reference source regions")

        return pipeBase.Struct(masked_difference_image=masked_diff)

    def _matchSources(self, ref_catalog, matching_sources, wcs):
        """Match reference catalog sources to detected sources using Astropy SkyCoord.

        Parameters
        ----------
        ref_catalog : `lsst.afw.table.SourceCatalog`
            Reference catalog sources
        matching_sources : `lsst.afw.table.SourceCatalog`
            Detected sources to match against
        wcs : `lsst.afw.geom.SkyWcs`
            WCS for coordinate conversion

        Returns
        -------
        matches : `list` of `tuple`
            List of (ref_source, matching_source) pairs that match
        """
        if len(ref_catalog) == 0 or len(matching_sources) == 0:
            return []

        # Extract reference source coordinates
        ref_ras = []
        ref_decs = []
        ref_sources = []

        for ref_src in ref_catalog:
            if (
                self.config.reference_ra_column in ref_src.schema
                and self.config.reference_dec_column in ref_src.schema
            ):
                # Convert to degrees - these might be Angle objects
                ra = ref_src[self.config.reference_ra_column].asDegrees()
                dec = ref_src[self.config.reference_dec_column].asDegrees()
            else:
                # Fallback to coord if specific columns not available
                coord = ref_src.getCoord()
                ra = coord.getRa().asDegrees()
                dec = coord.getDec().asDegrees()

            ref_ras.append(ra)
            ref_decs.append(dec)
            ref_sources.append(ref_src)

        # Extract detected source coordinates
        match_ras = []
        match_decs = []
        match_sources = []

        for src in matching_sources:
            coord = wcs.pixelToSky(Point2D(src.getX(), src.getY()))
            match_ras.append(coord.getRa().asDegrees())
            match_decs.append(coord.getDec().asDegrees())
            match_sources.append(src)

        # Create SkyCoord objects
        ref_coords = SkyCoord(ref_ras * u.deg, ref_decs * u.deg)
        match_coords = SkyCoord(match_ras * u.deg, match_decs * u.deg)

        # Perform matching with maximum separation
        max_sep = self.config.matching_radius * u.arcsec
        idx, d2d, d3d = match_coords.match_to_catalog_sky(ref_coords)

        # Build matches list for sources within matching radius
        matches = []
        n_too_far = 0
        for i, (match_idx, sep) in enumerate(zip(idx, d2d)):
            if sep <= max_sep:
                matches.append((ref_sources[match_idx], match_sources[i]))
            else:
                n_too_far += 1

        self.log.debug(
            f"Matching: {len(matches)} matched, {n_too_far} beyond {self.config.matching_radius} arcsec"
        )
        return matches

    def _applyReferenceMasks(self, matches, masked_exposure):
        """Apply reference masks to matched source regions.

        Parameters
        ----------
        matches : `list` of `tuple`
            List of (ref_source, matching_source) matched pairs
        masked_exposure : `lsst.afw.image.ExposureF`
            Difference image exposure to apply masks to

        Returns
        -------
        n_masked : `int`
            Number of regions masked
        """
        mask = masked_exposure.mask
        ref_mask_bit = mask.getPlaneBitMask(self.config.mask_plane_name)

        n_masked = 0

        for ref_src, matching_src in matches:
            # Use the source footprint from the matching source to define the mask region
            footprint = matching_src.getFootprint()
            if footprint is not None:
                self._maskSourceFootprint(footprint, mask, ref_mask_bit)
                n_masked += 1

        return n_masked

    def _maskSourceFootprint(self, footprint, mask, mask_bit):
        """Mask pixels in the source footprint.

        Parameters
        ----------
        footprint : `lsst.afw.detection.Footprint`
            Footprint of the source
        mask : `lsst.afw.image.Mask`
            Difference image mask to modify
        mask_bit : `int`
            Mask plane bit to set
        """
        # Get the footprint pixels - since matching and difference images have same WCS,
        # coordinates can be used directly
        bbox = mask.getBBox()
        n_clipped = 0
        n_masked = 0

        for span in footprint.getSpans():
            y = span.getY()
            for x in range(span.getMinX(), span.getMaxX() + 1):
                if bbox.contains(x, y):
                    mask.array[y, x] |= mask_bit
                    n_masked += 1
                else:
                    n_clipped += 1

        if n_clipped > 0:
            self.log.debug(
                f"Footprint clipped: {n_clipped} pixels outside bounds, {n_masked} pixels masked"
            )

    def _maskConnectedRegion(
        self, mask, seed_x, seed_y, detected_bit, detected_neg_bit, ref_mask_bit
    ):
        """Mask a connected region of DETECTED or DETECTED_NEGATIVE pixels.

        Parameters
        ----------
        mask : `lsst.afw.image.Mask`
            Mask to modify
        seed_x, seed_y : `int`
            Seed pixel coordinates
        detected_bit, detected_neg_bit : `int`
            Mask plane bits for DETECTED and DETECTED_NEGATIVE
        ref_mask_bit : `int`
            Mask plane bit for REFERENCE_MASK
        """
        height, width = mask.array.shape
        visited = np.zeros((height, width), dtype=bool)

        # Stack for flood fill algorithm
        stack = [(seed_x, seed_y)]
        target_bits = detected_bit | detected_neg_bit

        while stack:
            x, y = stack.pop()

            # Check bounds and if already visited
            if x < 0 or x >= width or y < 0 or y >= height or visited[y, x]:
                continue

            visited[y, x] = True
            mask_value = mask.array[y, x]

            # If this pixel has DETECTED or DETECTED_NEGATIVE, mark it and add neighbors
            if mask_value & target_bits:
                mask.array[y, x] |= ref_mask_bit

                # Add 8-connected neighbors to stack
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:  # Skip center pixel
                            stack.append((x + dx, y + dy))
