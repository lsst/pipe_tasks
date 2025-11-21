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

"""Pipeline for generating templates from archival DES data.
"""

__all__ = ["DesTemplateTask",
           "DesTemplateConfig",
           "DesTemplateConnections"]

import numpy as np

from astropy.table import Table
from astropy.io import fits

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.afw.math as afwMath
import lsst.afw.geom as afwGeom
from lsst.geom import Box2D, Point2D
from lsst.sphgeom import ConvexPolygon, UnitVector3d, LonLat
from lsst.afw.image import makePhotoCalibFromCalibZeroPoint
from lsst.meas.algorithms.installGaussianPsf import (
    InstallGaussianPsfTask,
    InstallGaussianPsfConfig,
)
from lsst.meas.algorithms import CoaddPsf, CoaddPsfConfig

from lsst.meas.base import IdGenerator
from lsst.geom import SpherePoint, degrees
import lsst.log


class DesTemplateConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "visit", "detector"),
):
    """Connections for DES template creation task."""

    bbox = connectionTypes.Input(
        doc="Bounding box of the exposure",
        name="pvi.bbox",
        storageClass="Box2I",
        dimensions=("instrument", "visit", "detector"),
    )
    wcs = connectionTypes.Input(
        doc="WCS of the exposure",
        name="pvi.wcs",
        storageClass="Wcs",
        dimensions=("instrument", "visit", "detector"),
    )
    template = connectionTypes.Output(
        doc="Template exposure created from DES tiles",
        name="desTemplate",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
    )


class DesTemplateConfig(
    pipeBase.PipelineTaskConfig, pipelineConnections=DesTemplateConnections
):
    """Configuration for DES template creation task."""

    tileFile = pexConfig.Field(
        dtype=str,
        default="merged_tiles_v3_deduplicated.csv",
        doc="Path to merged CSV file containing tile information with columns: "
        "tilename, band, survey, ra_cent, dec_cent, rac1-4, decc1-4, filepath",
    )
    dataSourcePriority = pexConfig.ListField(
        dtype=str,
        default=["DELVE", "DES", "DECADE"],
        doc="Priority order for surveys when duplicate tiles exist (first = highest priority)",
    )
    modelPsf = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Use modeled PSF instead of Gaussian PSF for templates",
    )
    gaussianPsfFwhm = pexConfig.Field(
        dtype=float,
        default=4.0,
        doc="FWHM for Gaussian PSF in pixels (used when modelPsf=False)",
    )
    gaussianPsfWidth = pexConfig.Field(
        dtype=int,
        default=21,
        doc="Width for Gaussian PSF in pixels (used when modelPsf=False)",
    )
    hasInverseVariancePlane = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Do the template files have a weight map or inverse variance plane",
    )


class DesTemplateTask(pipeBase.PipelineTask):
    """Create template exposures from survey tile data."""

    ConfigClass = DesTemplateConfig
    _DefaultName = "desTemplate"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Suppress FITS reader warnings for survey tiles
        lsst.log.setLevel("lsst.afw.image.MaskedImageFitsReader", lsst.log.ERROR)

        if self.config.modelPsf:
            from lsst.pipe.tasks.calibrateImage import (
                CalibrateImageTask,
                CalibrateImageConfig,
            )

            cal_config = CalibrateImageConfig()
            cal_config.psf_repair.doInterpolate = False
            cal_config.psf_repair.doCosmicRay = False
            cal_config.psf_detection.thresholdValue = 100.0
            self.calibrateTask = CalibrateImageTask(config=cal_config)
        else:
            install_psf_config = InstallGaussianPsfConfig()
            install_psf_config.fwhm = self.config.gaussianPsfFwhm
            install_psf_config.width = self.config.gaussianPsfWidth
            self.installPsfTask = InstallGaussianPsfTask(config=install_psf_config)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Extract band and physical_filter from dataId
        band = butlerQC.quantum.dataId["band"]
        physical_filter = butlerQC.quantum.dataId["physical_filter"]

        inputs = butlerQC.get(inputRefs)
        outputs = self.run(band=band, physical_filter=physical_filter, **inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, bbox, wcs, band, physical_filter):
        """Create template from survey tiles overlapping the target region.

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Bounding box of the target region
        wcs : `lsst.afw.geom.SkyWcs`
            WCS of the target region
        band : `str`
            Photometric band (e.g., 'r', 'g', 'i')
        physical_filter : `str`
            Physical filter name (e.g., 'r_03', 'g_01')

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results struct with:
            - ``template`` : Template exposure created from survey tiles
        """
        self.log.info("Creating survey template for target region")
        self.log.info(f"Using band '{band}' and physical_filter '{physical_filter}'")

        # Load tile catalog and filter by band
        tile_table = Table.read(self.config.tileFile, format="ascii")
        self.log.info(f"Loaded {len(tile_table)} total tiles from catalog")

        # Validate required columns exist
        required_columns = [
            "tilename",
            "band",
            "survey",
            "ra_cent",
            "dec_cent",
            "rac1",
            "rac2",
            "rac3",
            "rac4",
            "decc1",
            "decc2",
            "decc3",
            "decc4",
            "filepath",
        ]
        missing_columns = [
            col for col in required_columns if col not in tile_table.colnames
        ]
        if missing_columns:
            raise RuntimeError(
                f"Required columns missing from tile table: {missing_columns}. "
                "Please use the merged tiles CSV format."
            )

        # Filter by band first
        band_filtered_tiles = self._filterTilesByBand(tile_table, band)
        self.log.info(f"Found {len(band_filtered_tiles)} tiles for band '{band}'")

        if len(band_filtered_tiles) == 0:
            raise RuntimeError(f"No tiles found for band '{band}' in catalog")

        # Find overlapping tiles
        overlapping_tiles = self._findOverlappingTiles(bbox, wcs, band_filtered_tiles)
        self.log.info(f"Found {len(overlapping_tiles)} overlapping tiles")

        if not overlapping_tiles:
            center = wcs.pixelToSky(Point2D(bbox.getCenter()))
            target_ra = center.getRa().asDegrees()
            target_dec = center.getDec().asDegrees()
            self.log.info(
                f"Target was centered at RA={target_ra:.3f}, Dec={target_dec:.3f}"
            )
            raise RuntimeError("No survey tiles found overlapping with target region")

        # Resolve duplicate tiles and load template exposures
        resolved_tiles = self._resolveDuplicateTiles(
            overlapping_tiles, band_filtered_tiles
        )
        tile_exposures, tile_names = self._loadTileTemplates(resolved_tiles)
        self.log.info(f"Successfully loaded {len(tile_exposures)} template exposures")

        if not tile_exposures:
            raise RuntimeError("No valid template exposures found")

        # Create coadded template
        template = self._createCoaddFromTiles(
            tile_exposures, tile_names, wcs, bbox, physical_filter
        )

        self.log.info("Successfully created survey template")
        return pipeBase.Struct(template=template)

    def _tileOverlapsRegion(self, tile_corners, bbox, wcs):
        """Check if a tile overlaps with the target region using spherical geometry.

        Parameters
        ----------
        tile_corners : `list` of `tuple`
            List of (ra, dec) tuples for tile corners in degrees
        bbox : `lsst.geom.Box2I`
            Bounding box of target region
        wcs : `lsst.afw.geom.SkyWcs`
            WCS of target region

        Returns
        -------
        overlaps : `bool`
            True if tile overlaps with target region, False otherwise
        """
        # Get target region corners
        region_corners = [wcs.pixelToSky(Point2D(p)) for p in bbox.getCorners()]

        # Convert tile corners to unit vectors
        tile_unit_vectors = [
            UnitVector3d(LonLat.fromDegrees(ra, dec)) for ra, dec in tile_corners
        ]

        # Convert region corners to unit vectors
        region_unit_vectors = [
            UnitVector3d(
                LonLat.fromDegrees(s.getRa().asDegrees(), s.getDec().asDegrees())
            )
            for s in region_corners
        ]

        # Create convex polygons
        tile_polygon = ConvexPolygon(tile_unit_vectors)
        region_polygon = ConvexPolygon(region_unit_vectors)

        # Check for overlap
        return tile_polygon.overlaps(region_polygon)

    def _findOverlappingTiles(self, bbox, wcs, tile_table):
        """Find tiles that overlap with the target region.

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Bounding box of target region
        wcs : `lsst.afw.geom.SkyWcs`
            WCS of target region
        tile_table : `astropy.table.Table`
            Table containing tile information

        Returns
        -------
        overlapping_tiles : `list` of `tuple`
            List of (tile_name, distance_deg) for overlapping tiles, sorted by distance
        """
        # Get target region center for distance calculation
        center = wcs.pixelToSky(Point2D(bbox.getCenter()))
        target_ra = center.getRa().asDegrees()
        target_dec = center.getDec().asDegrees()
        p0 = SpherePoint(target_ra * degrees, target_dec * degrees)
        overlapping_tiles = []
        total_tiles_checked = 0

        self.log.info(f"Checking {len(tile_table)} tiles for overlap with exposure")

        for row in tile_table:
            tile_name = row["tilename"]
            total_tiles_checked += 1

            # Extract tile corners from table
            required_corner_columns = [
                "rac1",
                "decc1",
                "rac2",
                "decc2",
                "rac3",
                "decc3",
                "rac4",
                "decc4",
            ]

            if not all(col in tile_table.colnames for col in required_corner_columns):
                missing_corners = [
                    col
                    for col in required_corner_columns
                    if col not in tile_table.colnames
                ]
                raise RuntimeError(
                    f"Required corner coordinate columns missing from tile table: {missing_corners}"
                )

            # Use actual corner coordinates
            tile_corners = [
                (row["rac1"], row["decc1"]),
                (row["rac2"], row["decc2"]),
                (row["rac3"], row["decc3"]),
                (row["rac4"], row["decc4"]),
            ]

            # Check if tile overlaps with target region
            has_overlap = self._tileOverlapsRegion(tile_corners, bbox, wcs)

            if has_overlap:
                # Calculate distance to tile center for sorting
                tile_ra = row["ra_cent"]
                tile_dec = row["dec_cent"]

                p1 = SpherePoint(tile_ra * degrees, tile_dec * degrees)
                distance = p0.separation(p1).asDegrees()

                self.log.info(f"  OVERLAP: {tile_name} - distance={distance:.3f}°")

                overlapping_tiles.append((tile_name, distance))
            else:
                # Log nearby tiles that don't overlap for debugging
                tile_ra = row["ra_cent"]
                tile_dec = row["dec_cent"]
                p1 = SpherePoint(tile_ra * degrees, tile_dec * degrees)
                distance = p0.separation(p1).asDegrees()

                if distance < 2.0:  # Only log tiles within 2 degrees
                    self.log.info(
                        f"  No overlap: {tile_name} - distance={distance:.3f}° (RA={tile_ra:.3f}, Dec={tile_dec:.3f})"
                    )

        self.log.info(
            f"Checked {total_tiles_checked} tiles, found {len(overlapping_tiles)} with overlap"
        )

        # Sort by distance
        overlapping_tiles.sort(key=lambda x: x[1])

        # Return the overlapping tiles
        return overlapping_tiles

    def _filterTilesByBand(self, tile_table, band):
        """Filter tiles by the requested band and validate survey values.

        Parameters
        ----------
        tile_table : `astropy.table.Table`
            Table containing all tiles
        band : `str`
            Photometric band to filter for

        Returns
        -------
        filtered_table : `astropy.table.Table`
            Table containing only tiles for the specified band with valid surveys
        """
        if "band" not in tile_table.colnames:
            raise RuntimeError(
                "Required 'band' column not found in tile table. "
                "Please use the merged tiles CSV format."
            )

        if "survey" not in tile_table.colnames:
            raise RuntimeError(
                "Required 'survey' column not found in tile table. "
                "Please use the merged tiles CSV format."
            )

        # First filter by band
        band_mask = tile_table["band"] == band
        band_filtered = tile_table[band_mask]

        # Then validate survey values
        valid_surveys = set(self.config.dataSourcePriority)
        survey_column = band_filtered["survey"]

        # Vectorized membership check (case-sensitive)
        valid_mask = np.isin(survey_column, list(valid_surveys))

        invalid_count = len(band_filtered) - np.sum(valid_mask)
        if invalid_count > 0:
            self.log.warn(
                f"Found {invalid_count} tiles with unrecognized survey values, ignoring them"
            )

        filtered_table = band_filtered[valid_mask]

        self.log.info(
            f"Filtered {len(tile_table)} tiles to {len(filtered_table)} for band '{band}' with valid surveys"
        )
        return filtered_table

    def _resolveDuplicateTiles(self, overlapping_tiles, band_filtered_tiles):
        """Resolve duplicate tiles based on survey priority.

        Parameters
        ----------
        overlapping_tiles : `list` of `tuple`
            List of (tile_name, distance) tuples for overlapping tiles
        band_filtered_tiles : `astropy.table.Table`
            Band-filtered tile table with filepath and survey columns

        Returns
        -------
        resolved_tiles : `list` of `dict`
            List of resolved tile dictionaries with tilename, distance, filepath, survey
        """
        # Get priority order from config
        priority_order = self.config.dataSourcePriority

        resolved_tiles = []

        for tile_name, distance in overlapping_tiles:
            # Find all entries for this tilename in the band-filtered table
            tile_mask = band_filtered_tiles["tilename"] == tile_name
            tile_entries = band_filtered_tiles[tile_mask]

            if len(tile_entries) == 0:
                self.log.warn(f"No entries found for overlapping tile {tile_name}")
                continue
            elif len(tile_entries) == 1:
                # No duplicates, use the single entry
                entry = tile_entries[0]
                resolved_tiles.append(
                    {
                        "tilename": tile_name,
                        "distance": distance,
                        "filepath": entry["filepath"],
                        "survey": entry["survey"],
                    }
                )
            else:
                # Multiple entries, resolve based on priority order
                selected_entry = None

                # Check for each survey in priority order
                for preferred_survey in priority_order:
                    survey_entries = tile_entries[
                        tile_entries["survey"] == preferred_survey
                    ]
                    if len(survey_entries) > 0:
                        selected_entry = survey_entries[0]
                        self.log.info(
                            f"Using {preferred_survey} tile for {tile_name} (priority)"
                        )
                        break

                if selected_entry is None:
                    # No recognized survey, use first available
                    selected_entry = tile_entries[0]
                    self.log.info(
                        f"Using {selected_entry['survey']} tile for {tile_name} (fallback)"
                    )

                resolved_tiles.append(
                    {
                        "tilename": tile_name,
                        "distance": distance,
                        "filepath": selected_entry["filepath"],
                        "survey": selected_entry["survey"],
                    }
                )

        self.log.info(
            f"Resolved {len(resolved_tiles)} tiles from {len(overlapping_tiles)} overlapping tiles"
        )
        return resolved_tiles

    def _loadTileTemplates(self, resolved_tiles):
        """Load template exposures from resolved tiles.

        Parameters
        ----------
        resolved_tiles : `list` of `dict`
            List of resolved tile dictionaries with tilename, distance, filepath, survey

        Returns
        -------
        exposures : `list` of `lsst.afw.image.ExposureF`
            List of loaded template exposures
        tile_names : `list` of `str`
            List of tile names corresponding to exposures
        """
        exposures = []
        tile_names = []

        for tile_info in resolved_tiles:
            tile_name = tile_info["tilename"]
            filepath = tile_info["filepath"]
            survey = tile_info["survey"]

            try:
                self.log.info(f"Loading {survey} tile {tile_name} from {filepath}")
                exp = afwImage.ExposureF(filepath)

                if self.config.hasInverseVariancePlane:
                    # Need convert DES weight map to variance map for LSST exposure
                    mi = exp.maskedImage
                    bad_weight = mi.variance.array <= 0.0
                    mi.variance.array[~bad_weight] = (
                        1.0 / mi.variance.array[~bad_weight]
                    )
                    mi.variance.array[bad_weight] = np.nan
                    mi.mask.array[bad_weight] |= mi.mask.getPlaneBitMask("NO_DATA")

                # Set PSF
                if not self.config.modelPsf:
                    # Install Gaussian PSF
                    self.installPsfTask.run(exp)
                else:
                    # Use modeled PSF
                    cat = self.calibrateTask._compute_psf(exp, IdGenerator())

                # Set photometric calibration
                metadata = exp.getInfo().getMetadata()
                if "MAGZERO" in metadata:
                    zp = metadata["MAGZERO"]
                    flux0 = 10 ** (0.4 * zp)
                    calib = makePhotoCalibFromCalibZeroPoint(flux0, 0.0)
                    exp.setPhotoCalib(calib)
                else:
                    self.log.warn(
                        f"No MAGZERO found in {tile_name}, using default calibration"
                    )

                exposures.append(exp)
                tile_names.append(tile_name)

            except Exception as e:
                self.log.warn(f"Failed to load tile {tile_name}: {e}")
                continue

        return exposures, tile_names

    def _createCoaddFromTiles(
        self, tile_exposures, tile_names, target_wcs, target_bbox, physical_filter
    ):
        """Create coadded template from tile exposures.

        Parameters
        ----------
        tile_exposures : `list` of `lsst.afw.image.ExposureF`
            List of tile exposures
        tile_names : `list` of `str`
            List of tile names
        target_wcs : `lsst.afw.geom.SkyWcs`
            Target WCS for template
        target_bbox : `lsst.geom.Box2I`
            Target bounding box for template
        physical_filter : `str`
            Physical filter name for the template

        Returns
        -------
        template : `lsst.afw.image.ExposureF`
            Coadded template exposure
        """
        warper = afwMath.Warper.fromConfig(afwMath.Warper.ConfigClass())

        # Create tile catalog for PSF computation
        tile_schema = afwTable.ExposureTable.makeMinimalSchema()
        tile_key = tile_schema.addField("tile", type="String", size=12)
        weight_key = tile_schema.addField("weight", type=float)

        coadd_config = CoaddPsfConfig()

        # Statistics configuration
        stats_flags = afwMath.stringToStatisticsProperty("MEAN")
        stats_ctrl = afwMath.StatisticsControl()
        stats_ctrl.setNanSafe(True)
        stats_ctrl.setWeighted(True)
        stats_ctrl.setCalcErrorMosaicMode(True)

        tile_catalog = afwTable.ExposureCatalog(tile_schema)
        masked_images = []
        weights = []

        for i, (exp, tile) in enumerate(zip(tile_exposures, tile_names)):
            self.log.debug(f"Processing tile {tile} ({i+1}/{len(tile_exposures)})")

            # Warp template to target coordinate system
            warped = warper.warpExposure(target_wcs, exp, maxBBox=target_bbox)
            if warped.getBBox().getArea() == 0:
                self.log.debug(f"Skipping tile {tile}: no overlap after warping")
                continue

            # Create properly initialized exposure
            aligned_exp = afwImage.ExposureF(target_bbox, target_wcs)
            aligned_exp.maskedImage.set(
                np.nan, afwImage.Mask.getPlaneBitMask("NO_DATA"), np.nan
            )
            aligned_exp.maskedImage.assign(warped.maskedImage, warped.getBBox())
            masked_images.append(aligned_exp.maskedImage)

            # Calculate weight (inverse of mean variance)
            var_array = aligned_exp.variance.array
            finite_var = var_array[np.isfinite(var_array)]
            if len(finite_var) > 0:
                mean_var = np.mean(finite_var)
                weight = 1.0 / mean_var if mean_var > 0 else 1.0
            else:
                weight = 1.0
            weights.append(weight)

            # Add to tile catalog for PSF computation
            record = tile_catalog.addNew()
            record.set(tile_key, tile)
            record.set(weight_key, weight)
            record.setPsf(exp.getPsf())
            record.setWcs(exp.getWcs())
            record.setPhotoCalib(exp.getPhotoCalib())
            record.setBBox(exp.getBBox())

            polygon = afwGeom.Polygon(Box2D(exp.getBBox()).getCorners())
            record.setValidPolygon(polygon)

        if not masked_images:
            raise RuntimeError("No valid warped images for coadd creation")

        # Create coadd exposure
        coadd = afwImage.ExposureF(target_bbox, target_wcs)
        coadd.maskedImage.set(np.nan, afwImage.Mask.getPlaneBitMask("NO_DATA"), np.nan)
        xy0 = coadd.getXY0()

        # Perform statistical stacking
        coadd.maskedImage = afwMath.statisticsStack(
            masked_images, stats_flags, stats_ctrl, weights, clipped=0, maskMap=[]
        )
        coadd.maskedImage.setXY0(xy0)

        # Create and set PSF
        if len(tile_catalog) > 0:
            valid_mask = (
                coadd.maskedImage.mask.array
                & coadd.maskedImage.mask.getPlaneBitMask("NO_DATA")
            ) == 0
            if np.any(valid_mask):
                mask_for_centroid = afwImage.makeMaskFromArray(
                    valid_mask.astype(afwImage.MaskPixel)
                )
                psf_center = afwGeom.SpanSet.fromMask(
                    mask_for_centroid, 1
                ).computeCentroid()

                ctrl = coadd_config.makeControl()
                coadd_psf = CoaddPsf(
                    tile_catalog,
                    target_wcs,
                    psf_center,
                    ctrl.warpingKernelName,
                    ctrl.cacheSize,
                )
                coadd.setPsf(coadd_psf)

        # Set calibration and filter from first tile exposure and physical_filter
        if tile_exposures:
            coadd.setPhotoCalib(tile_exposures[0].getPhotoCalib())
            # Create filter label from physical_filter
            # Extract band from physical_filter (e.g., 'r_03' -> 'r')
            band = (
                physical_filter.split("_")[0]
                if "_" in physical_filter
                else physical_filter
            )
            filter_label = afwImage.FilterLabel(band=band, physical=physical_filter)
            coadd.setFilter(filter_label)

        return coadd
