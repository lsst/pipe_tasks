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

__all__ = ["HealSparseInputMapTask", "HealSparseInputMapConfig",
           "HealSparseMapFormatter", "HealSparsePropertyMapConnections",
           "HealSparsePropertyMapConfig", "HealSparsePropertyMapTask",
           "ConsolidateHealSparsePropertyMapConnections",
           "ConsolidateHealSparsePropertyMapConfig",
           "ConsolidateHealSparsePropertyMapTask"]

from collections import defaultdict
import dataclasses
import esutil
import warnings
import numbers
import numpy as np
import hpgeom as hpg
import healsparse as hsp

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.geom
import lsst.afw.geom as afwGeom
from lsst.daf.butler import Formatter
from lsst.skymap import BaseSkyMap
from lsst.utils.timer import timeMethod
from .healSparseMappingProperties import (BasePropertyMap, BasePropertyMapConfig,
                                          PropertyMapMap, compute_approx_psf_size_and_shape)


class HealSparseMapFormatter(Formatter):
    """Interface for reading and writing healsparse.HealSparseMap files."""
    unsupportedParameters = frozenset()
    supportedExtensions = frozenset({".hsp", ".fit", ".fits"})
    extension = '.hsp'

    def read(self, component=None):
        # Docstring inherited from Formatter.read.
        path = self.fileDescriptor.location.path

        if component == 'coverage':
            try:
                data = hsp.HealSparseCoverage.read(path)
            except (OSError, RuntimeError):
                raise ValueError(f"Unable to read healsparse map with URI {self.fileDescriptor.location.uri}")

            return data

        if self.fileDescriptor.parameters is None:
            pixels = None
            degrade_nside = None
        else:
            pixels = self.fileDescriptor.parameters.get('pixels', None)
            degrade_nside = self.fileDescriptor.parameters.get('degrade_nside', None)
        try:
            data = hsp.HealSparseMap.read(path, pixels=pixels, degrade_nside=degrade_nside)
        except (OSError, RuntimeError):
            raise ValueError(f"Unable to read healsparse map with URI {self.fileDescriptor.location.uri}")

        return data

    def write(self, inMemoryDataset):
        # Docstring inherited from Formatter.write.
        # Update the location with the formatter-preferred file extension
        self.fileDescriptor.location.updateExtension(self.extension)
        inMemoryDataset.write(self.fileDescriptor.location.path, clobber=True)


def _is_power_of_two(value):
    """Check that value is a power of two.

    Parameters
    ----------
    value : `int`
        Value to check.

    Returns
    -------
    is_power_of_two : `bool`
        True if value is a power of two; False otherwise, or
        if value is not an integer.
    """
    if not isinstance(value, numbers.Integral):
        return False

    # See https://stackoverflow.com/questions/57025836
    # Every power of 2 has exactly 1 bit set to 1; subtracting
    # 1 therefore flips every preceding bit.  If you and that
    # together with the original value it must be 0.
    return (value & (value - 1) == 0) and value != 0


class HealSparseInputMapConfig(pexConfig.Config):
    """Configuration parameters for HealSparseInputMapTask"""
    nside = pexConfig.Field(
        doc="Mapping healpix nside.  Must be power of 2.",
        dtype=int,
        default=32768,
        check=_is_power_of_two,
    )
    nside_coverage = pexConfig.Field(
        doc="HealSparse coverage map nside.  Must be power of 2.",
        dtype=int,
        default=256,
        check=_is_power_of_two,
    )
    bad_mask_min_coverage = pexConfig.Field(
        doc=("Minimum area fraction of a map healpixel pixel that must be "
             "covered by bad pixels to be removed from the input map. "
             "This is approximate."),
        dtype=float,
        default=0.5,
    )


class HealSparseInputMapTask(pipeBase.Task):
    """Task for making a HealSparse input map."""

    ConfigClass = HealSparseInputMapConfig
    _DefaultName = "healSparseInputMap"

    def __init__(self, **kwargs):
        pipeBase.Task.__init__(self, **kwargs)

        self.ccd_input_map = None

    def build_ccd_input_map(self, bbox, wcs, ccds):
        """Build a map from ccd valid polygons or bounding boxes.

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Bounding box for region to build input map.
        wcs : `lsst.afw.geom.SkyWcs`
            WCS object for region to build input map.
        ccds : `lsst.afw.table.ExposureCatalog`
            Exposure catalog with ccd data from coadd inputs.
        """
        with warnings.catch_warnings():
            # Healsparse will emit a warning if nside coverage is greater than
            # 128.  In the case of generating patch input maps, and not global
            # maps, high nside coverage works fine, so we can suppress this
            # warning.
            warnings.simplefilter("ignore")
            self.ccd_input_map = hsp.HealSparseMap.make_empty(nside_coverage=self.config.nside_coverage,
                                                              nside_sparse=self.config.nside,
                                                              dtype=hsp.WIDE_MASK,
                                                              wide_mask_maxbits=len(ccds))
        self._wcs = wcs
        self._bbox = bbox
        self._ccds = ccds

        pixel_scale = wcs.getPixelScale(bbox.getCenter()).asArcseconds()
        hpix_area_arcsec2 = hpg.nside_to_pixel_area(self.config.nside, degrees=True)*(3600.**2.)
        self._min_bad = self.config.bad_mask_min_coverage*hpix_area_arcsec2/(pixel_scale**2.)

        metadata = {}
        self._bits_per_visit_ccd = {}
        self._bits_per_visit = defaultdict(list)
        for bit, ccd_row in enumerate(ccds):
            metadata[f"B{bit:04d}CCD"] = ccd_row["ccd"]
            metadata[f"B{bit:04d}VIS"] = ccd_row["visit"]
            metadata[f"B{bit:04d}WT"] = ccd_row["weight"]

            self._bits_per_visit_ccd[(ccd_row["visit"], ccd_row["ccd"])] = bit
            self._bits_per_visit[ccd_row["visit"]].append(bit)

            ccd_poly = ccd_row.getValidPolygon()
            if ccd_poly is None:
                ccd_poly = afwGeom.Polygon(lsst.geom.Box2D(ccd_row.getBBox()))
            # Detectors need to be rendered with their own wcs.
            ccd_poly_radec = self._pixels_to_radec(ccd_row.getWcs(), ccd_poly.convexHull().getVertices())

            # Create a ccd healsparse polygon
            poly = hsp.Polygon(ra=ccd_poly_radec[: -1, 0],
                               dec=ccd_poly_radec[: -1, 1],
                               value=[bit])
            self.ccd_input_map.set_bits_pix(poly.get_pixels(nside=self.ccd_input_map.nside_sparse),
                                            [bit])

        # Cut down to the overall bounding box with associated wcs.
        bbox_afw_poly = afwGeom.Polygon(lsst.geom.Box2D(bbox))
        bbox_poly_radec = self._pixels_to_radec(self._wcs,
                                                bbox_afw_poly.convexHull().getVertices())
        bbox_poly = hsp.Polygon(ra=bbox_poly_radec[: -1, 0], dec=bbox_poly_radec[: -1, 1],
                                value=np.arange(self.ccd_input_map.wide_mask_maxbits))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bbox_poly_map = bbox_poly.get_map_like(self.ccd_input_map)
            self.ccd_input_map = hsp.and_intersection([self.ccd_input_map, bbox_poly_map])
        self.ccd_input_map.metadata = metadata

        # Create a temporary map to hold the count of bad pixels in each healpix pixel
        dtype = [(f"v{visit}", np.int64) for visit in self._bits_per_visit.keys()]

        with warnings.catch_warnings():
            # Healsparse will emit a warning if nside coverage is greater than
            # 128.  In the case of generating patch input maps, and not global
            # maps, high nside coverage works fine, so we can suppress this
            # warning.
            warnings.simplefilter("ignore")
            self._ccd_input_bad_count_map = hsp.HealSparseMap.make_empty(
                nside_coverage=self.config.nside_coverage,
                nside_sparse=self.config.nside,
                dtype=dtype,
                primary=dtype[0][0])

        self._ccd_input_pixels = self.ccd_input_map.valid_pixels

        # Don't set input bad map if there are no ccds which overlap the bbox.
        if len(self._ccd_input_pixels) > 0:
            # Ensure these are sorted.
            self._ccd_input_pixels = np.sort(self._ccd_input_pixels)

            self._ccd_input_bad_count_map[self._ccd_input_pixels] = np.zeros(1, dtype=dtype)

    def mask_warp_bbox(self, bbox, visit, mask, bit_mask_value):
        """Mask a subregion from a visit.
        This must be run after build_ccd_input_map initializes
        the overall map.

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Bounding box from region to mask.
        visit : `int`
            Visit number corresponding to warp with mask.
        mask : `lsst.afw.image.MaskX`
            Mask plane from warp exposure.
        bit_mask_value : `int`
            Bit mask to check for bad pixels.

        Raises
        ------
        RuntimeError : Raised if build_ccd_input_map was not run first.
        """
        if self.ccd_input_map is None:
            raise RuntimeError("Must run build_ccd_input_map before mask_warp_bbox")

        if len(self._ccd_input_pixels) == 0:
            # This tract has no coverage, so there is nothing to do.
            return

        # Find the bad pixels and convert to healpix
        bad_pixels = np.where(mask.array & bit_mask_value)
        if len(bad_pixels[0]) == 0:
            # No bad pixels
            return

        # Bad pixels come from warps which use the overall wcs.
        bad_ra, bad_dec = self._wcs.pixelToSkyArray(bad_pixels[1].astype(np.float64),
                                                    bad_pixels[0].astype(np.float64),
                                                    degrees=True)
        bad_hpix = hpg.angle_to_pixel(self.config.nside, bad_ra, bad_dec)

        # Check if any of these "bad" pixels are in the valid footprint.
        match_input, match_bad = esutil.numpy_util.match(self._ccd_input_pixels, bad_hpix, presorted=True)
        if len(match_bad) == 0:
            return

        bad_hpix = bad_hpix[match_bad]

        # Create a view of the column we need to add to.
        count_map_visit = self._ccd_input_bad_count_map[f"v{visit}"]
        # Add the bad pixels to the accumulator. Note that the view
        # cannot append pixels, but the match above ensures we are
        # only adding to pixels that are already in the coverage
        # map and initialized.
        count_map_visit.update_values_pix(bad_hpix, 1, operation="add")

    def finalize_ccd_input_map_mask(self):
        """Use accumulated mask information to finalize the masking of
        ccd_input_map.

        Raises
        ------
        RuntimeError : Raised if build_ccd_input_map was not run first.
        """
        if self.ccd_input_map is None:
            raise RuntimeError("Must run build_ccd_input_map before finalize_ccd_input_map_mask.")

        count_map_arr = self._ccd_input_bad_count_map[self._ccd_input_pixels]
        for visit in self._bits_per_visit:
            to_mask, = np.where(count_map_arr[f"v{visit}"] > self._min_bad)
            if to_mask.size == 0:
                continue
            self.ccd_input_map.clear_bits_pix(self._ccd_input_pixels[to_mask],
                                              self._bits_per_visit[visit])

        # Clear memory
        self._ccd_input_bad_count_map = None

    def _pixels_to_radec(self, wcs, pixels):
        """Convert pixels to ra/dec positions using a wcs.

        Parameters
        ----------
        wcs : `lsst.afw.geom.SkyWcs`
            WCS object.
        pixels : `list` [`lsst.geom.Point2D`]
            List of pixels to convert.

        Returns
        -------
        radec : `numpy.ndarray`
            Nx2 array of ra/dec positions associated with pixels.
        """
        sph_pts = wcs.pixelToSky(pixels)
        return np.array([(sph.getRa().asDegrees(), sph.getDec().asDegrees())
                         for sph in sph_pts])


class HealSparsePropertyMapConnections(pipeBase.PipelineTaskConnections,
                                       dimensions=("tract", "band", "skymap",),
                                       defaultTemplates={"coaddName": "deep",
                                                         "calexpType": "",
                                                         # If set, prefix replaces "{coaddName}Coadd_".
                                                         "prefix": ""}):
    input_maps = pipeBase.connectionTypes.Input(
        doc="Healsparse bit-wise coadd input maps",
        name="{coaddName}Coadd_inputMap",
        storageClass="HealSparseMap",
        dimensions=("tract", "patch", "skymap", "band"),
        multiple=True,
        deferLoad=True,
    )
    coadd_exposures = pipeBase.connectionTypes.Input(
        doc="Coadded exposures associated with input_maps",
        name="{coaddName}Coadd",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "band"),
        multiple=True,
        deferLoad=True,
    )
    visit_summaries = pipeBase.connectionTypes.Input(
        doc="Visit summary tables with aggregated statistics",
        name="finalVisitSummary",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit"),
        multiple=True,
        deferLoad=True,
    )
    sky_map = pipeBase.connectionTypes.Input(
        doc="Input definition of geometry/bbox and projection/wcs for coadded exposures",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )

    # Create output connections for all possible maps defined in the
    # registry.  The vars() trick used here allows us to set class attributes
    # programmatically.  Taken from
    # https://stackoverflow.com/questions/2519807/
    # setting-a-class-attribute-with-a-given-name-in-python-while-defining-the-class
    for name in BasePropertyMap.registry:
        vars()[f"{name}_map_min"] = pipeBase.connectionTypes.Output(
            doc=f"Minimum-value map of {name}",
            name=f"{{coaddName}}Coadd_{name}_map_min",
            storageClass="HealSparseMap",
            dimensions=("tract", "skymap", "band"),
        )
        vars()[f"{name}_map_max"] = pipeBase.connectionTypes.Output(
            doc=f"Maximum-value map of {name}",
            name=f"{{coaddName}}Coadd_{name}_map_max",
            storageClass="HealSparseMap",
            dimensions=("tract", "skymap", "band"),
        )
        vars()[f"{name}_map_mean"] = pipeBase.connectionTypes.Output(
            doc=f"Mean-value map of {name}",
            name=f"{{coaddName}}Coadd_{name}_map_mean",
            storageClass="HealSparseMap",
            dimensions=("tract", "skymap", "band"),
        )
        vars()[f"{name}_map_weighted_mean"] = pipeBase.connectionTypes.Output(
            doc=f"Weighted mean-value map of {name}",
            name=f"{{coaddName}}Coadd_{name}_map_weighted_mean",
            storageClass="HealSparseMap",
            dimensions=("tract", "skymap", "band"),
        )
        vars()[f"{name}_map_sum"] = pipeBase.connectionTypes.Output(
            doc=f"Sum-value map of {name}",
            name=f"{{coaddName}}Coadd_{name}_map_sum",
            storageClass="HealSparseMap",
            dimensions=("tract", "skymap", "band"),
        )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        # Not all possible maps in the registry will be configured to run.
        # Here we remove the unused connections.
        for name in BasePropertyMap.registry:
            if name not in config.property_maps:
                prop_config = BasePropertyMapConfig()
                prop_config.do_min = False
                prop_config.do_max = False
                prop_config.do_mean = False
                prop_config.do_weighted_mean = False
                prop_config.do_sum = False
            else:
                prop_config = config.property_maps[name]

            if not prop_config.do_min:
                self.outputs.remove(f"{name}_map_min")
            if not prop_config.do_max:
                self.outputs.remove(f"{name}_map_max")
            if not prop_config.do_mean:
                self.outputs.remove(f"{name}_map_mean")
            if not prop_config.do_weighted_mean:
                self.outputs.remove(f"{name}_map_weighted_mean")
            if not prop_config.do_sum:
                self.outputs.remove(f"{name}_map_sum")

        if config.connections.prefix:
            # If the 'prefix' connection template is set, replace
            # '{coaddName}Coadd_' with that; this is a fully-backwards
            # compatible way of overriding more than just the
            # currently-configurable part of the prefix.
            for connection_name in self.outputs:
                old_connection = getattr(self, connection_name)
                new_dataset_type_name = old_connection.name.replace(
                    f"{self.config.connections.coaddName}Coadd_",
                    config.connections.prefix,
                )
                setattr(
                    self,
                    connection_name,
                    dataclasses.replace(old_connection, name=new_dataset_type_name)
                )


class HealSparsePropertyMapConfig(pipeBase.PipelineTaskConfig,
                                  pipelineConnections=HealSparsePropertyMapConnections):
    """Configuration parameters for HealSparsePropertyMapTask"""
    property_maps = BasePropertyMap.registry.makeField(
        multi=True,
        default=["exposure_time",
                 "psf_size",
                 "psf_e1",
                 "psf_e2",
                 "psf_maglim",
                 "sky_noise",
                 "sky_background",
                 "dcr_dra",
                 "dcr_ddec",
                 "dcr_e1",
                 "dcr_e2",
                 "epoch"],
        doc="Property map computation objects",
    )

    def setDefaults(self):
        self.property_maps["exposure_time"].do_sum = True
        self.property_maps["psf_size"].do_weighted_mean = True
        self.property_maps["psf_e1"].do_weighted_mean = True
        self.property_maps["psf_e2"].do_weighted_mean = True
        self.property_maps["psf_maglim"].do_weighted_mean = True
        self.property_maps["sky_noise"].do_weighted_mean = True
        self.property_maps["sky_background"].do_weighted_mean = True
        self.property_maps["dcr_dra"].do_weighted_mean = True
        self.property_maps["dcr_ddec"].do_weighted_mean = True
        self.property_maps["dcr_e1"].do_weighted_mean = True
        self.property_maps["dcr_e2"].do_weighted_mean = True
        self.property_maps["epoch"].do_mean = True
        self.property_maps["epoch"].do_min = True
        self.property_maps["epoch"].do_max = True


class HealSparsePropertyMapTask(pipeBase.PipelineTask):
    """Task to compute Healsparse property maps.

    This task will compute individual property maps (per tract, per
    map type, per band).  These maps cover the full coadd tract, and
    are not truncated to the inner tract region.
    """
    ConfigClass = HealSparsePropertyMapConfig
    _DefaultName = "healSparsePropertyMapTask"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.property_maps = PropertyMapMap()
        for name, config, PropertyMapClass in self.config.property_maps.apply():
            self.property_maps[name] = PropertyMapClass(config, name)

    @timeMethod
    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        sky_map = inputs.pop("sky_map")

        tract = butlerQC.quantum.dataId["tract"]
        band = butlerQC.quantum.dataId["band"]

        input_map_dict = {ref.dataId["patch"]: ref for ref in inputs["input_maps"]}
        coadd_dict = {ref.dataId["patch"]: ref for ref in inputs["coadd_exposures"]}

        visit_summary_dict = {ref.dataId["visit"]: ref.get()
                              for ref in inputs["visit_summaries"]}

        self.run(sky_map, tract, band, coadd_dict, input_map_dict, visit_summary_dict)

        # Write the outputs
        for name, property_map in self.property_maps.items():
            if property_map.config.do_min:
                butlerQC.put(property_map.min_map,
                             getattr(outputRefs, f"{name}_map_min"))
            if property_map.config.do_max:
                butlerQC.put(property_map.max_map,
                             getattr(outputRefs, f"{name}_map_max"))
            if property_map.config.do_mean:
                butlerQC.put(property_map.mean_map,
                             getattr(outputRefs, f"{name}_map_mean"))
            if property_map.config.do_weighted_mean:
                butlerQC.put(property_map.weighted_mean_map,
                             getattr(outputRefs, f"{name}_map_weighted_mean"))
            if property_map.config.do_sum:
                butlerQC.put(property_map.sum_map,
                             getattr(outputRefs, f"{name}_map_sum"))

    def run(self, sky_map, tract, band, coadd_dict, input_map_dict, visit_summary_dict):
        """Run the healsparse property task.

        Parameters
        ----------
        sky_map : Sky map object
        tract : `int`
            Tract number.
        band : `str`
            Band name for logging.
        coadd_dict : `dict` [`int`: `lsst.daf.butler.DeferredDatasetHandle`]
            Dictionary of coadd exposure references.  Keys are patch numbers.
        input_map_dict : `dict` [`int`: `lsst.daf.butler.DeferredDatasetHandle`]
            Dictionary of input map references.  Keys are patch numbers.
        visit_summary_dict : `dict` [`int`: `lsst.afw.table.ExposureCatalog`]
            Dictionary of visit summary tables.  Keys are visit numbers.

        Raises
        ------
        RepeatableQuantumError
            If visit_summary_dict is missing any visits or detectors found in an
            input map.  This leads to an inconsistency between what is in the coadd
            (via the input map) and the visit summary tables which contain data
            to compute the maps.
        """
        tract_info = sky_map[tract]

        tract_maps_initialized = False

        for patch in input_map_dict.keys():
            self.log.info("Making maps for band %s, tract %d, patch %d.",
                          band, tract, patch)

            patch_info = tract_info[patch]

            input_map = input_map_dict[patch].get()

            # Initialize the tract maps as soon as we have the first input
            # map for getting nside information.
            if not tract_maps_initialized:
                # We use the first input map nside information to initialize
                # the tract maps
                nside_coverage = self._compute_nside_coverage_tract(tract_info)
                nside = input_map.nside_sparse

                do_compute_approx_psf = False
                # Initialize the tract maps
                for property_map in self.property_maps:
                    property_map.initialize_tract_maps(nside_coverage, nside)
                    if property_map.requires_psf:
                        do_compute_approx_psf = True

                tract_maps_initialized = True

            if input_map.valid_pixels.size == 0:
                self.log.warning("No valid pixels for band %s, tract %d, patch %d; skipping.",
                                 band, tract, patch)
                continue

            coadd_photo_calib = coadd_dict[patch].get(component="photoCalib")
            coadd_inputs = coadd_dict[patch].get(component="coaddInputs")

            coadd_zeropoint = 2.5*np.log10(coadd_photo_calib.getInstFluxAtZeroMagnitude())

            # Crop input_map to the inner polygon of the patch
            poly_vertices = patch_info.getInnerSkyPolygon(tract_info.getWcs()).getVertices()
            patch_radec = self._vertices_to_radec(poly_vertices)
            patch_poly = hsp.Polygon(ra=patch_radec[:, 0], dec=patch_radec[:, 1],
                                     value=np.arange(input_map.wide_mask_maxbits))
            with warnings.catch_warnings():
                # Healsparse will emit a warning if nside coverage is greater than
                # 128.  In the case of generating patch input maps, and not global
                # maps, high nside coverage works fine, so we can suppress this
                # warning.
                warnings.simplefilter("ignore")
                patch_poly_map = patch_poly.get_map_like(input_map)
                input_map = hsp.and_intersection([input_map, patch_poly_map])

            valid_pixels, vpix_ra, vpix_dec = input_map.valid_pixels_pos(return_pixels=True)

            # Check if there are no valid pixels for the inner (unique) patch region
            if valid_pixels.size == 0:
                continue

            # Initialize the value accumulators
            for property_map in self.property_maps:
                property_map.initialize_values(valid_pixels.size)
                property_map.zeropoint = coadd_zeropoint

            # Initialize the weight and counter accumulators
            total_weights = np.zeros(valid_pixels.size)
            total_inputs = np.zeros(valid_pixels.size, dtype=np.int32)

            for bit, ccd_row in enumerate(coadd_inputs.ccds):
                # Which pixels in the map are used by this visit/detector
                inmap, = np.where(input_map.check_bits_pix(valid_pixels, [bit]))

                # Check if there are any valid pixels in the map from this deteector.
                if inmap.size == 0:
                    continue

                # visit, detector_id, weight = input_dict[bit]
                visit = ccd_row["visit"]
                detector_id = ccd_row["ccd"]
                weight = ccd_row["weight"]

                x, y = ccd_row.getWcs().skyToPixelArray(vpix_ra[inmap], vpix_dec[inmap], degrees=True)
                scalings = self._compute_calib_scale(ccd_row, x, y)

                if do_compute_approx_psf:
                    psf_array = compute_approx_psf_size_and_shape(ccd_row, vpix_ra[inmap], vpix_dec[inmap])
                else:
                    psf_array = None

                total_weights[inmap] += weight
                total_inputs[inmap] += 1

                # Retrieve the correct visitSummary row
                if visit not in visit_summary_dict:
                    msg = f"Visit {visit} not found in visit_summaries."
                    raise pipeBase.RepeatableQuantumError(msg)
                row = visit_summary_dict[visit].find(detector_id)
                if row is None:
                    msg = f"Visit {visit} / detector_id {detector_id} not found in visit_summaries."
                    raise pipeBase.RepeatableQuantumError(msg)

                # Accumulate the values
                for property_map in self.property_maps:
                    property_map.accumulate_values(inmap,
                                                   vpix_ra[inmap],
                                                   vpix_dec[inmap],
                                                   weight,
                                                   scalings,
                                                   row,
                                                   psf_array=psf_array)

            # Finalize the mean values and set the tract maps
            for property_map in self.property_maps:
                property_map.finalize_mean_values(total_weights, total_inputs)
                property_map.set_map_values(valid_pixels)

    def _compute_calib_scale(self, ccd_row, x, y):
        """Compute calibration scaling values.

        Parameters
        ----------
        ccd_row : `lsst.afw.table.ExposureRecord`
            Exposure metadata for a given detector exposure.
        x : `np.ndarray`
            Array of x positions.
        y : `np.ndarray`
            Array of y positions.

        Returns
        -------
        calib_scale : `np.ndarray`
            Array of calibration scale values.
        """
        photo_calib = ccd_row.getPhotoCalib()
        bf = photo_calib.computeScaledCalibration()
        if bf.getBBox() == ccd_row.getBBox():
            # Track variable calibration over the detector
            calib_scale = photo_calib.getCalibrationMean()*bf.evaluate(x, y)
        else:
            # Spatially constant calibration
            calib_scale = photo_calib.getCalibrationMean()

        return calib_scale

    def _vertices_to_radec(self, vertices):
        """Convert polygon vertices to ra/dec.

        Parameters
        ----------
        vertices : `list` [ `lsst.sphgeom.UnitVector3d` ]
            Vertices for bounding polygon.

        Returns
        -------
        radec : `numpy.ndarray`
            Nx2 array of ra/dec positions (in degrees) associated with vertices.
        """
        lonlats = [lsst.sphgeom.LonLat(x) for x in vertices]
        radec = np.array([(x.getLon().asDegrees(), x.getLat().asDegrees()) for
                          x in lonlats])
        return radec

    def _compute_nside_coverage_tract(self, tract_info):
        """Compute the optimal coverage nside for a tract.

        Parameters
        ----------
        tract_info : `lsst.skymap.tractInfo.ExplicitTractInfo`
            Tract information object.

        Returns
        -------
        nside_coverage : `int`
            Optimal coverage nside for a tract map.
        """
        num_patches = tract_info.getNumPatches()

        # Compute approximate patch area
        patch_info = tract_info.getPatchInfo(0)
        vertices = patch_info.getInnerSkyPolygon(tract_info.getWcs()).getVertices()
        radec = self._vertices_to_radec(vertices)
        delta_ra = np.max(radec[:, 0]) - np.min(radec[:, 0])
        delta_dec = np.max(radec[:, 1]) - np.min(radec[:, 1])
        patch_area = delta_ra*delta_dec*np.cos(np.deg2rad(np.mean(radec[:, 1])))

        tract_area = num_patches[0]*num_patches[1]*patch_area
        # Start with a fairly low nside and increase until we find the approximate area.
        nside_coverage_tract = 32
        while hpg.nside_to_pixel_area(nside_coverage_tract, degrees=True) > tract_area:
            nside_coverage_tract = 2*nside_coverage_tract
        # Step back one, but don't go bigger pixels than nside=32 or smaller
        # than 128 (recommended by healsparse).
        nside_coverage_tract = int(np.clip(nside_coverage_tract/2, 32, 128))

        return nside_coverage_tract


class ConsolidateHealSparsePropertyMapConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("band", "skymap",),
    defaultTemplates={
        "coaddName": "deep",
        # If set, prefix replaces "{coaddName}Coadd_".
        "prefix": ""
    }
):
    sky_map = pipeBase.connectionTypes.Input(
        doc="Input definition of geometry/bbox and projection/wcs for coadded exposures",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )

    # Create output connections for all possible maps defined in the
    # registry.  The vars() trick used here allows us to set class attributes
    # programatically.  Taken from
    # https://stackoverflow.com/questions/2519807/
    # setting-a-class-attribute-with-a-given-name-in-python-while-defining-the-class
    for name in BasePropertyMap.registry:
        vars()[f"{name}_map_min"] = pipeBase.connectionTypes.Input(
            doc=f"Minimum-value map of {name}",
            name=f"{{coaddName}}Coadd_{name}_map_min",
            storageClass="HealSparseMap",
            dimensions=("tract", "skymap", "band"),
            multiple=True,
            deferLoad=True,
        )
        vars()[f"{name}_consolidated_map_min"] = pipeBase.connectionTypes.Output(
            doc=f"Minumum-value map of {name}",
            name=f"{{coaddName}}Coadd_{name}_consolidated_map_min",
            storageClass="HealSparseMap",
            dimensions=("skymap", "band"),
        )
        vars()[f"{name}_map_max"] = pipeBase.connectionTypes.Input(
            doc=f"Maximum-value map of {name}",
            name=f"{{coaddName}}Coadd_{name}_map_max",
            storageClass="HealSparseMap",
            dimensions=("tract", "skymap", "band"),
            multiple=True,
            deferLoad=True,
        )
        vars()[f"{name}_consolidated_map_max"] = pipeBase.connectionTypes.Output(
            doc=f"Minumum-value map of {name}",
            name=f"{{coaddName}}Coadd_{name}_consolidated_map_max",
            storageClass="HealSparseMap",
            dimensions=("skymap", "band"),
        )
        vars()[f"{name}_map_mean"] = pipeBase.connectionTypes.Input(
            doc=f"Mean-value map of {name}",
            name=f"{{coaddName}}Coadd_{name}_map_mean",
            storageClass="HealSparseMap",
            dimensions=("tract", "skymap", "band"),
            multiple=True,
            deferLoad=True,
        )
        vars()[f"{name}_consolidated_map_mean"] = pipeBase.connectionTypes.Output(
            doc=f"Minumum-value map of {name}",
            name=f"{{coaddName}}Coadd_{name}_consolidated_map_mean",
            storageClass="HealSparseMap",
            dimensions=("skymap", "band"),
        )
        vars()[f"{name}_map_weighted_mean"] = pipeBase.connectionTypes.Input(
            doc=f"Weighted mean-value map of {name}",
            name=f"{{coaddName}}Coadd_{name}_map_weighted_mean",
            storageClass="HealSparseMap",
            dimensions=("tract", "skymap", "band"),
            multiple=True,
            deferLoad=True,
        )
        vars()[f"{name}_consolidated_map_weighted_mean"] = pipeBase.connectionTypes.Output(
            doc=f"Minumum-value map of {name}",
            name=f"{{coaddName}}Coadd_{name}_consolidated_map_weighted_mean",
            storageClass="HealSparseMap",
            dimensions=("skymap", "band"),
        )
        vars()[f"{name}_map_sum"] = pipeBase.connectionTypes.Input(
            doc=f"Sum-value map of {name}",
            name=f"{{coaddName}}Coadd_{name}_map_sum",
            storageClass="HealSparseMap",
            dimensions=("tract", "skymap", "band"),
            multiple=True,
            deferLoad=True,
        )
        vars()[f"{name}_consolidated_map_sum"] = pipeBase.connectionTypes.Output(
            doc=f"Minumum-value map of {name}",
            name=f"{{coaddName}}Coadd_{name}_consolidated_map_sum",
            storageClass="HealSparseMap",
            dimensions=("skymap", "band"),
        )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        # Not all possible maps in the registry will be configured to run.
        # Here we remove the unused connections.
        for name in BasePropertyMap.registry:
            if name not in config.property_maps:
                prop_config = BasePropertyMapConfig()
                prop_config.do_min = False
                prop_config.do_max = False
                prop_config.do_mean = False
                prop_config.do_weighted_mean = False
                prop_config.do_sum = False
            else:
                prop_config = config.property_maps[name]

            if not prop_config.do_min:
                self.inputs.remove(f"{name}_map_min")
                self.outputs.remove(f"{name}_consolidated_map_min")
            if not prop_config.do_max:
                self.inputs.remove(f"{name}_map_max")
                self.outputs.remove(f"{name}_consolidated_map_max")
            if not prop_config.do_mean:
                self.inputs.remove(f"{name}_map_mean")
                self.outputs.remove(f"{name}_consolidated_map_mean")
            if not prop_config.do_weighted_mean:
                self.inputs.remove(f"{name}_map_weighted_mean")
                self.outputs.remove(f"{name}_consolidated_map_weighted_mean")
            if not prop_config.do_sum:
                self.inputs.remove(f"{name}_map_sum")
                self.outputs.remove(f"{name}_consolidated_map_sum")

        if config.connections.prefix:
            # If the 'prefix' connection template is set, replace
            # '{coaddName}Coadd_' with that; this is a fully-backwards
            # compatible way of overriding more than just the
            # currently-configurable part of the prefix.
            for connection_name in self.inputs | self.outputs:
                if connection_name == "sky_map":
                    continue
                old_connection = getattr(self, connection_name)
                new_dataset_type_name = old_connection.name.replace(
                    f"{self.config.connections.coaddName}Coadd_",
                    config.connections.prefix,
                )
                setattr(
                    self,
                    connection_name,
                    dataclasses.replace(old_connection, name=new_dataset_type_name)
                )


class ConsolidateHealSparsePropertyMapConfig(pipeBase.PipelineTaskConfig,
                                             pipelineConnections=ConsolidateHealSparsePropertyMapConnections):
    """Configuration parameters for ConsolidateHealSparsePropertyMapTask"""
    property_maps = BasePropertyMap.registry.makeField(
        multi=True,
        default=["exposure_time",
                 "psf_size",
                 "psf_e1",
                 "psf_e2",
                 "psf_maglim",
                 "sky_noise",
                 "sky_background",
                 "dcr_dra",
                 "dcr_ddec",
                 "dcr_e1",
                 "dcr_e2",
                 "epoch"],
        doc="Property map computation objects",
    )
    nside_coverage = pexConfig.Field(
        doc="Consolidated HealSparse coverage map nside.  Must be power of 2.",
        dtype=int,
        default=32,
        check=_is_power_of_two,
    )

    def setDefaults(self):
        self.property_maps["exposure_time"].do_sum = True
        self.property_maps["psf_size"].do_weighted_mean = True
        self.property_maps["psf_e1"].do_weighted_mean = True
        self.property_maps["psf_e2"].do_weighted_mean = True
        self.property_maps["psf_maglim"].do_weighted_mean = True
        self.property_maps["sky_noise"].do_weighted_mean = True
        self.property_maps["sky_background"].do_weighted_mean = True
        self.property_maps["dcr_dra"].do_weighted_mean = True
        self.property_maps["dcr_ddec"].do_weighted_mean = True
        self.property_maps["dcr_e1"].do_weighted_mean = True
        self.property_maps["dcr_e2"].do_weighted_mean = True
        self.property_maps["epoch"].do_mean = True
        self.property_maps["epoch"].do_min = True
        self.property_maps["epoch"].do_max = True


class ConsolidateHealSparsePropertyMapTask(pipeBase.PipelineTask):
    """Task to consolidate HealSparse property maps.

    This task will take all the individual tract-based maps (per map type,
    per band) and consolidate them into one survey-wide map (per map type,
    per band).  Each tract map is truncated to its inner region before
    consolidation.
    """
    ConfigClass = ConsolidateHealSparsePropertyMapConfig
    _DefaultName = "consolidateHealSparsePropertyMapTask"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.property_maps = PropertyMapMap()
        for name, config, PropertyMapClass in self.config.property_maps.apply():
            self.property_maps[name] = PropertyMapClass(config, name)

    @timeMethod
    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        sky_map = inputs.pop("sky_map")

        # These need to be consolidated one at a time to conserve memory.
        for name in self.config.property_maps.names:
            for type_ in ['min', 'max', 'mean', 'weighted_mean', 'sum']:
                map_type = f"{name}_map_{type_}"
                if map_type in inputs:
                    input_refs = {ref.dataId['tract']: ref
                                  for ref in inputs[map_type]}
                    consolidated_map = self.consolidate_map(sky_map, input_refs)
                    butlerQC.put(consolidated_map,
                                 getattr(outputRefs, f"{name}_consolidated_map_{type_}"))

    def consolidate_map(self, sky_map, input_refs):
        """Consolidate the healsparse property maps.

        Parameters
        ----------
        sky_map : Sky map object
        input_refs : `dict` [`int`: `lsst.daf.butler.DeferredDatasetHandle`]
            Dictionary of tract_id mapping to dataref.

        Returns
        -------
        consolidated_map : `healsparse.HealSparseMap`
            Consolidated HealSparse map.
        """
        # First, we read in the coverage maps to know how much memory
        # to allocate
        cov_mask = None
        nside_coverage_inputs = None
        for tract_id in input_refs:
            cov = input_refs[tract_id].get(component='coverage')
            if cov_mask is None:
                cov_mask = cov.coverage_mask
                nside_coverage_inputs = cov.nside_coverage
            else:
                cov_mask |= cov.coverage_mask

        cov_pix_inputs, = np.where(cov_mask)

        # Compute the coverage pixels for the desired nside_coverage
        if nside_coverage_inputs == self.config.nside_coverage:
            cov_pix = cov_pix_inputs
        elif nside_coverage_inputs > self.config.nside_coverage:
            # Converting from higher resolution coverage to lower
            # resolution coverage.
            bit_shift = hsp.utils._compute_bitshift(self.config.nside_coverage,
                                                    nside_coverage_inputs)
            cov_pix = np.right_shift(cov_pix_inputs, bit_shift)
        else:
            # Converting from lower resolution coverage to higher
            # resolution coverage.
            bit_shift = hsp.utils._compute_bitshift(nside_coverage_inputs,
                                                    self.config.nside_coverage)
            cov_pix = np.left_shift(cov_pix_inputs, bit_shift)

        # Now read in each tract map and build the consolidated map.
        consolidated_map = None
        for tract_id in input_refs:
            input_map = input_refs[tract_id].get()
            if consolidated_map is None:
                consolidated_map = hsp.HealSparseMap.make_empty(
                    self.config.nside_coverage,
                    input_map.nside_sparse,
                    input_map.dtype,
                    sentinel=input_map._sentinel,
                    cov_pixels=cov_pix,
                    metadata=input_map.metadata,
                )

            # Only use pixels that are properly inside the tract.
            vpix, ra, dec = input_map.valid_pixels_pos(return_pixels=True)
            vpix_tract_ids = sky_map.findTractIdArray(ra, dec, degrees=True)

            in_tract = (vpix_tract_ids == tract_id)

            consolidated_map[vpix[in_tract]] = input_map[vpix[in_tract]]

        return consolidated_map
