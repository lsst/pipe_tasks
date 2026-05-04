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
from __future__ import annotations

__all__ = ("AllSkyFitsHipsTaskConnections", "AllSkyFitsHipsTaskConfig", "AllSkyFitsHipsTask")

from PIL import Image
import io
import math
import re
import hpgeom as hpg
import numpy as np
import healsparse as hsp
from dataclasses import replace
from datetime import datetime
from collections.abc import Iterable


from lsst.resources import ResourcePath
from lsst.pex.config import Field, ConfigField, ChoiceField
from lsst.sphgeom import RangeSet

from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
    QuantumContext,
    InputQuantizedConnection,
    OutputQuantizedConnection,
    TaskMetadata,
)
from lsst.pipe.base.connectionTypes import Input


from ..rgb2hips._properties import HipsPropertiesConfig, _write_property
from ..healSparseMapping import _is_power_of_two


class AllSkyFitsHipsTaskConnections(
    PipelineTaskConnections,
    dimensions=tuple(
        "band",
    ),
    defaultTemplates={"input_task_label": "generateLowOrderHips"},
):
    low_order_metadata = Input(
        doc="Metadata produced by the LowOrderHipsTask",
        name="{input_task_label}_metadata",
        storageClass="TaskMetadata",
        multiple=True,
        deferLoad=True,
        dimensions=tuple(),
    )
    input_hips = Input(
        doc="Hips pixels at level 8 used to build higher orders",
        name="fits_picture_hips8",
        storageClass="NumpyArray",
        multiple=True,
        deferLoad=True,
        dimensions=("healpix8", "band"),
    )

    def __init__(self, *, config: AllSkyFitsHipsTaskConfig):
        # Set the input dimensions to whatever the minimum order healpix
        # to produce is.
        self.low_order_metadata = replace(
            self.low_order_metadata, dimensions=set((f"healpix{config.min_order}",))
        )


class AllSkyFitsHipsTaskConfig(PipelineTaskConfig, pipelineConnections=AllSkyFitsHipsTaskConnections):
    hips_base_uri = Field[str](
        doc="URI to HiPS base for output.",
        optional=False,
    )
    color_ordering = Field[str](
        doc=(
            "A string of the astrophysical bands that correspond to the RGB channels in the color image "
            "inputs to high_order_hips task. This is in making the hips metadata"
        ),
        optional=False,
    )
    properties = ConfigField[HipsPropertiesConfig](
        doc="Configuration for properties file.",
    )
    allsky_tilesize = Field[int](
        dtype=int,
        doc="Allsky tile size; must be power of 2. HiPS standard recommends 64x64 tiles.",
        default=64,
        check=_is_power_of_two,
    )
    max_order = Field[int](doc="The maximum order hips that was produced", default=11)
    shift_order = Field[int](
        doc="Shift order of hips, right now must be 9 configuration for future options", default=9
    )
    min_order = Field[int](
        doc="Minimum healpix order for HiPS tree.",
        default=3,
    )

    def validate(self):
        if self.shift_order != 9:
            raise ValueError("Shift order must be 9.")
        return super().validate()


class AllSkyFitsHipsTask(PipelineTask):
    """Pipeline task for generating all-sky HealPix (HiPS) tiles and associated metadata."""

    _DefaultName = "allSkyHipsTask"
    ConfigClass = AllSkyFitsHipsTaskConfig
    config: ConfigClass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hips_base_path = ResourcePath(self.config.hips_base_uri, forceDirectory=True)

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        band_name = butlerQC.quantum.dataId["band"]

        self.hips_base_path = self.hips_base_path.join(f"band_{band_name}", forceDirectory=True)

        # Extract the healpix8 pixel ids.
        hpx8_pixels = []
        for ref in inputRefs.input_hips:
            hpx8_pixels.append((ref.dataId["healpix8"]))

        # Scale pixel IDS to higher order (hpx11) that were already produced
        hpx8_rangeset = RangeSet(hpx8_pixels)
        hpx11_rangeset = hpx8_rangeset.scaled(4**3)
        hpx11_pixels = set()
        for begin, end in hpx11_rangeset:
            hpx11_pixels.update(range(begin, end))
        hpx11_pixels = np.array([s for s in hpx11_pixels])

        low_order_metadata = butlerQC.get(inputRefs.low_order_metadata)

        outputs = self.run(low_order_metadata, hpx11_pixels, band_name)
        butlerQC.put(outputs, outputRefs)

    def run(self, low_order_metadata: Iterable[TaskMetadata], hpx11_pixels, band_name) -> Struct:
        """Generate all-sky HealPix tiles and metadata.

        Parameters
        ----------
        low_order_metadata : Iterable[TaskMetadata]
            Low-order metadata from previous processing steps.
        hpx11_pixels : array-like
            Array of HPX11 pixel IDs to process.
        band_name : `str`
            The band in which this data was collected

        Returns
        -------
        Struct
            This task produces no outputs so an empty struct is returned
        """
        self._write_properties_and_moc(
            self.config.max_order, hpx11_pixels, self.config.shift_order, band_name
        )
        self._write_allsky_file(self.config.min_order)
        return Struct()

    def _write_properties_and_moc(self, max_order, pixels, shift_order, band):
        """Write HiPS properties file and MOC.

        Parameters
        ----------
        max_order : `int`
            Maximum HEALPix order.
        pixels : `np.ndarray` (N,)
            Array of pixels used.
        shift_order : `int`
            HPX shift order.
        band : `str`
            Band (or color).
        """
        area = hpg.nside_to_pixel_area(2**max_order, degrees=True) * len(pixels)

        initial_ra = self.config.properties.initial_ra
        initial_dec = self.config.properties.initial_dec
        initial_fov = self.config.properties.initial_fov

        if initial_ra is None or initial_dec is None or initial_fov is None:
            # We want to point to an arbitrary pixel in the footprint.
            # Just take the median pixel value for simplicity.
            temp_pixels = pixels.copy()
            if temp_pixels.size % 2 == 0:
                temp_pixels = np.append(temp_pixels, [temp_pixels[0]])
            medpix = int(np.median(temp_pixels))
            _initial_ra, _initial_dec = hpg.pixel_to_angle(2**max_order, medpix)
            _initial_fov = hpg.nside_to_resolution(2**max_order, units="arcminutes") / 60.0

            if initial_ra is None or initial_dec is None:
                initial_ra = _initial_ra
                initial_dec = _initial_dec
            if initial_fov is None:
                initial_fov = _initial_fov

        self._write_hips_properties_file(
            self.config.properties,
            band,
            max_order,
            shift_order,
            area,
            initial_ra,
            initial_dec,
            initial_fov,
        )

        # Write the MOC coverage
        self._write_hips_moc_file(
            max_order,
            pixels,
        )

    def _write_hips_properties_file(
        self,
        properties_config,
        band,
        max_order,
        shift_order,
        area,
        initial_ra,
        initial_dec,
        initial_fov,
    ):
        """Write HiPS properties file.

        Parameters
        ----------
        properties_config : `lsst.pipe.tasks.hips.HipsPropertiesConfig`
            Configuration for properties values.
        band : `str`
            Name of band(s) for HiPS tree.
        max_order : `int`
            Maximum HEALPix order.
        shift_order : `int`
            HPX shift order.
        area : `float`
            Coverage area in square degrees.
        initial_ra : `float`
            Initial HiPS RA position (degrees).
        initial_dec : `float`
            Initial HiPS Dec position (degrees).
        initial_fov : `float`
            Initial HiPS display size (degrees).
        """

        bitpix = 32
        hbitpix = 32

        date_iso8601 = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        pixel_scale = hpg.nside_to_resolution(2 ** (max_order + shift_order), units="degrees")

        uri = self.hips_base_path.join("properties")
        with ResourcePath.temporary_uri(suffix=uri.getExtension()) as temporary_uri:
            with open(temporary_uri.ospath, "w") as fh:
                _write_property(
                    fh,
                    "creator_did",
                    properties_config.creator_did_template.format(band=band),
                )
                if properties_config.obs_collection is not None:
                    _write_property(fh, "obs_collection", properties_config.obs_collection)
                _write_property(
                    fh,
                    "obs_title",
                    properties_config.obs_title_template.format(band=band),
                )
                if properties_config.obs_description_template is not None:
                    _write_property(
                        fh,
                        "obs_description",
                        properties_config.obs_description_template.format(band=band),
                    )
                if len(properties_config.prov_progenitor) > 0:
                    for prov_progenitor in properties_config.prov_progenitor:
                        _write_property(fh, "prov_progenitor", prov_progenitor)
                if properties_config.obs_ack is not None:
                    _write_property(fh, "obs_ack", properties_config.obs_ack)
                _write_property(fh, "obs_regime", "Optical")
                _write_property(fh, "data_pixel_bitpix", str(bitpix))
                _write_property(fh, "dataproduct_type", "image")
                _write_property(fh, "moc_sky_fraction", str(area / 41253.0))
                _write_property(fh, "data_ucd", "phot.flux")
                _write_property(fh, "hips_creation_date", date_iso8601)
                _write_property(fh, "hips_builder", "lsst.pipe.tasks.hips.GenerateHipsTask")
                _write_property(fh, "hips_creator", "Vera C. Rubin Observatory")
                _write_property(fh, "hips_version", "1.4")
                _write_property(fh, "hips_release_date", date_iso8601)
                _write_property(fh, "hips_frame", "equatorial")
                _write_property(fh, "hips_order", str(max_order))
                _write_property(fh, "hips_tile_width", str(2**shift_order))
                _write_property(fh, "hips_status", "private master clonableOnce")
                _write_property(fh, "hips_tile_format", "fits")
                _write_property(fh, "dataproduct_subtype", "color")
                _write_property(fh, "hips_pixel_bitpix", str(hbitpix))
                _write_property(fh, "hips_pixel_scale", str(pixel_scale))
                _write_property(fh, "hips_initial_ra", str(initial_ra))
                _write_property(fh, "hips_initial_dec", str(initial_dec))
                _write_property(fh, "hips_initial_fov", str(initial_fov))
                if band in properties_config.spectral_ranges:
                    em_min = properties_config.spectral_ranges[band].lambda_min / 1e9
                else:
                    self.log.warning("blue band %s not in self.config.spectral_ranges.", band)
                    em_min = 3e-7
                if band in properties_config.spectral_ranges:
                    em_max = properties_config.spectral_ranges[band].lambda_max / 1e9
                else:
                    self.log.warning("red band %s not in self.config.spectral_ranges.", band)
                    em_max = 1e-6
                _write_property(fh, "em_min", str(em_min))
                _write_property(fh, "em_max", str(em_max))
                if properties_config.t_min is not None:
                    _write_property(fh, "t_min", properties_config.t_min)
                if properties_config.t_max is not None:
                    _write_property(fh, "t_max", properties_config.t_max)

            uri.transfer_from(temporary_uri, transfer="copy", overwrite=True)

    def _write_hips_moc_file(self, max_order, pixels, min_uniq_order=1):
        """Write HiPS MOC file.

        Parameters
        ----------
        max_order : `int`
            Maximum HEALPix order.
        pixels : `np.ndarray`
            Array of pixels covered.
        min_uniq_order : `int`, optional
            Minimum HEALPix order for looking for fully covered pixels.
        """
        # WARNING: In general PipelineTasks are not allowed to do any outputs
        # outside of the butler. This task has been given (temporary)
        # Special Dispensation because of the nature of HiPS outputs until
        # a more controlled solution can be found.

        # Make a healsparse map which provides easy degrade/comparisons.
        hspmap = hsp.HealSparseMap.make_empty(2**min_uniq_order, 2**max_order, dtype=np.int8)
        hspmap[pixels] = 1

        uri = self.hips_base_path.join("Moc.fits")
        with ResourcePath.temporary_uri(suffix=uri.getExtension()) as temporary_uri:
            hspmap.write_moc(temporary_uri.ospath)
            uri.transfer_from(temporary_uri, transfer="copy", overwrite=True)

    def _write_allsky_file(self, allsky_order):
        """Write an Allsky.png file.

        Parameters
        ----------
        allsky_order : `int`
            HEALPix order of the minimum order to make allsky file.
        """
        tile_size = self.config.allsky_tilesize

        # The Allsky file format is described in
        # https://www.ivoa.net/documents/HiPS/20170519/REC-HIPS-1.0-20170519.pdf
        # From S4.3.2:
        # The Allsky file is built as an array of tiles, stored side by side in
        # the left-to-right order. The width of this array must be the square
        # root of the number of the tiles of the order. For instance, the width
        # of this array at order 3 is 27 ( (int)sqrt(768) ). To avoid having a
        # too large Allsky file, the resolution of each tile may be reduced but
        # must stay a power of two (typically 64x64 pixels rather than 512x512).

        n_tiles = hpg.nside_to_npixel(hpg.order_to_nside(allsky_order))
        n_tiles_wide = int(np.floor(np.sqrt(n_tiles)))
        n_tiles_high = int(np.ceil(n_tiles / n_tiles_wide))

        allsky_image = None

        allsky_order_uri = self.hips_base_path.join(f"Norder{allsky_order}", forceDirectory=True)
        pixel_regex = re.compile(r"Npix([0-9]+)\.fits$")

        image_uris = list(
            ResourcePath.findFileResources(
                candidates=[allsky_order_uri],
                file_filter=pixel_regex,
            )
        )

        for image_uri in image_uris:
            matches = re.match(pixel_regex, image_uri.basename())
            pix_num = int(matches.group(1))
            tile_image = Image.open(io.BytesIO(image_uri.read()))
            row = math.floor(pix_num // n_tiles_wide)
            column = pix_num % n_tiles_wide
            box = (column * tile_size, row * tile_size, (column + 1) * tile_size, (row + 1) * tile_size)
            tile_image_shrunk = tile_image.resize((tile_size, tile_size))

            if allsky_image is None:
                allsky_image = Image.new(
                    tile_image.mode,
                    (n_tiles_wide * tile_size, n_tiles_high * tile_size),
                )
            allsky_image.paste(tile_image_shrunk, box)

        uri = allsky_order_uri.join(f"Allsky.fits")

        with ResourcePath.temporary_uri(suffix=uri.getExtension()) as temporary_uri:
            allsky_image.save(temporary_uri.ospath)

            uri.transfer_from(temporary_uri, transfer="copy", overwrite=True)
