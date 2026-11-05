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

__all__ = ("AllSkyHipsTaskConnections", "AllSkyHipsTaskConfig", "AllSkyHipsTask")

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


from ._properties import HipsPropertiesConfig, _write_property
from ._utils import downsample_hips_array
from ..healSparseMapping import _is_power_of_two


class AllSkyHipsTaskConnections(
    PipelineTaskConnections,
    dimensions=tuple(),
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
        name="rgb_picture_hips8",
        storageClass="NumpyArray",
        multiple=True,
        deferLoad=True,
        dimensions=("healpix8",),
    )

    def __init__(self, *, config: AllSkyHipsTaskConfig):
        # Set the input dimensions to whatever the minimum order healpix
        # to produce is.
        self.low_order_metadata = replace(
            self.low_order_metadata, dimensions=set((f"healpix{config.min_order}",))
        )


class AllSkyHipsTaskConfig(PipelineTaskConfig, pipelineConnections=AllSkyHipsTaskConnections):
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
    file_extension = ChoiceField[str](
        doc="Extension for the presisted image.",
        allowed={
            "png": "Use the png image extension",
            "webp": "Use the webp image extension",
            "avif": "Use the avif image extension (PQ-encoded HDR tiles; requires is_hdr=True)",
        },
        default="png",
    )
    is_hdr = Field[bool](
        doc=(
            "The tree is PQ-encoded HDR delivered as AVIF (must match the "
            "tile tasks' is_hdr). The Allsky preview is then composited "
            "from the healpix8 float arrays via the same PQ writer as the "
            "tiles, never from decoded tiles."
        ),
        default=False,
    )
    lum_norm = Field[float](
        doc=(
            "HDR only: the linear pixel value that encodes to lum_norm "
            "nits (paper white) under PQ. Default 100.0 (linear 1.0 -> "
            "100 nits). Should match the tile tasks' lum_norm."
        ),
        default=100.0,
        check=lambda x: x > 0,
    )
    clamp_values = Field[float](
        doc=(
            "HDR only: optional upper clamp (in linear input units) applied "
            "before PQ encoding of the Allsky preview. None disables clamping."
        ),
        default=None,
        optional=True,
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
    bluest_band = Field[str](
        doc="Band corresponding to bluest color in image",
        default="g",
    )
    reddest_band = Field[str](
        doc="Band corresponding to reddest color in image",
        default="i",
    )
    max_order = Field[int](doc="The maximum order hips that was produced", default=11)
    data_bitpix = ChoiceField[str](
        doc="The dtype of the original input data",
        allowed={
            "float64": "double precision",
            "float32": "single precision",
            "uint8": "8 bit uint",
            "uint16": "16 bit uint",
            "uint32": "32 bit uint",
            "uint64": "64 bit uint",
        },
        default="float32",
    )
    hips_bitpix = ChoiceField[str](
        doc="The dtype of the hips images",
        allowed={
            "float64": "double precision",
            "float32": "single precision",
            "uint8": "8 bit uint",
            "uint16": "16 bit uint",
            "uint32": "32 bit uint",
            "uint64": "64 bit uint",
        },
        default="uint8",
    )
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
        if self.is_hdr and self.file_extension != "avif":
            raise ValueError(
                f"is_hdr=True requires file_extension='avif' (got '{self.file_extension}'): "
                "the HDR Allsky preview is written by the PQ-encoded AVIF writer."
            )
        if self.file_extension == "avif" and not self.is_hdr:
            raise ValueError(
                "file_extension='avif' requires is_hdr=True: the avif path is the PQ-encoded "
                "HDR writer; no SDR avif encoding is provided."
            )
        return super().validate()


class AllSkyHipsTask(PipelineTask):
    """Pipeline task for generating all-sky HealPix (HiPS) tiles and associated metadata."""

    _DefaultName = "allSkyHipsTask"
    ConfigClass = AllSkyHipsTaskConfig
    config: ConfigClass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hips_base_path = ResourcePath(self.config.hips_base_uri, forceDirectory=True)
        self.hips_base_path = self.hips_base_path.join(
            f"color_{self.config.color_ordering}", forceDirectory=True
        )

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        # Extract the healpix8 pixel ids (and their float arrays, which the
        # HDR AllSky build composites directly — no tile re-reading).
        hpx8_pixels = []
        hpx8_data = []
        for ref in inputRefs.input_hips:
            hpx8_pixels.append((ref.dataId["healpix8"]))
            hpx8_data.append((ref.dataId["healpix8"], butlerQC.get(ref)))

        # Scale pixel IDS to higher order (hpx11) that were already produced
        hpx8_rangeset = RangeSet(hpx8_pixels)
        hpx11_rangeset = hpx8_rangeset.scaled(4**3)
        hpx11_pixels = set()
        for begin, end in hpx11_rangeset:
            hpx11_pixels.update(range(begin, end))
        hpx11_pixels = np.array([s for s in hpx11_pixels])

        low_order_metadata = butlerQC.get(inputRefs.low_order_metadata)

        outputs = self.run(low_order_metadata, hpx11_pixels, hpx8_data)
        butlerQC.put(outputs, outputRefs)

    def run(
        self,
        low_order_metadata: Iterable[TaskMetadata],
        hpx11_pixels,
        input_hips: Iterable[tuple[int, object]] | None = None,
    ) -> Struct:
        """Generate all-sky HealPix tiles and metadata.

        Parameters
        ----------
        low_order_metadata : Iterable[TaskMetadata]
            Low-order metadata from previous processing steps.
        hpx11_pixels : array-like
            Array of HPX11 pixel IDs to process.
        input_hips : Iterable of [``int``, array-like], optional
            (healpix8 pixel id, linear float array or deferred handle)
            pairs from HighOrderHipsTask's butler output. Required when
            ``config.is_hdr`` (the preview is composited from these).

        Returns
        -------
        Struct
            This task produces no outputs so an empty strct is returned
        """
        self._write_properties_and_moc(
            self.config.max_order, hpx11_pixels, self.config.shift_order, self.config.color_ordering
        )
        if self.config.is_hdr:
            self._write_allsky_file_hdr(self.config.min_order, input_hips)
        else:
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

        match self.config.data_bitpix:
            case "float64":
                bitpix = -64
            case "float32":
                bitpix = -32
            case "uint8":
                bitpix = 8
            case "uint16":
                bitpix = 16
            case "uint32":
                bitpix = 32
            case "uint64":
                bitpix = 64

        match self.config.hips_bitpix:
            case "float64":
                hbitpix = -64
            case "float32":
                hbitpix = -32
            case "uint8":
                hbitpix = 8
            case "uint16":
                hbitpix = 16
            case "uint32":
                hbitpix = 32
            case "uint64":
                hbitpix = 64

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
                _write_property(fh, "hips_tile_format", self.config.file_extension)
                _write_property(fh, "dataproduct_subtype", "color")
                _write_property(fh, "hips_pixel_bitpix", str(hbitpix))
                _write_property(fh, "hips_pixel_scale", str(pixel_scale))
                _write_property(fh, "hips_initial_ra", str(initial_ra))
                _write_property(fh, "hips_initial_dec", str(initial_dec))
                _write_property(fh, "hips_initial_fov", str(initial_fov))
                if self.config.bluest_band in properties_config.spectral_ranges:
                    em_min = properties_config.spectral_ranges[self.config.bluest_band].lambda_min / 1e9
                else:
                    self.log.warning("blue band %s not in self.config.spectral_ranges.", band)
                    em_min = 3e-7
                if self.config.reddest_band in properties_config.spectral_ranges:
                    em_max = properties_config.spectral_ranges[self.config.reddest_band].lambda_max / 1e9
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
        if self.config.file_extension == "png":
            pixel_regex = re.compile(r"Npix([0-9]+)\.png$")
        elif self.config.file_extension == "webp":
            pixel_regex = re.compile(r"Npix([0-9]+)\.webp$")
        else:
            raise RuntimeError("Unknown file extension")

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

        uri = allsky_order_uri.join(f"Allsky.{self.config.file_extension}")

        with ResourcePath.temporary_uri(suffix=uri.getExtension()) as temporary_uri:
            allsky_image.save(temporary_uri.ospath)

            uri.transfer_from(temporary_uri, transfer="copy", overwrite=True)

    def _write_allsky_file_hdr(self, allsky_order, input_hips):
        """Write the Allsky preview for an HDR (AVIF/PQ) tree.

        Unlike the SDR path in ``_write_allsky_file``, this never reads
        tiles back from disk: the preview is composited from the healpix8
        linear float arrays HighOrderHipsTask stored in the butler (HDR
        intact), each area-downsampled into its slot on the HiPS Allsky
        grid and encoded once through the same PQ writer as the tiles —
        so the viewer's first-paint background is bit-family-identical to
        the tiles streamed over it. Grid layout follows IVOA HIPS S4.3.2
        (tiles left-to-right, order rows).

        Parameters
        ----------
        allsky_order : `int`
            HEALPix order of the preview grid (must be < 8).
        input_hips : `Iterable` of [``int``, array-like]
            (healpix8 pixel id, (256, 256, 3) linear float array) pairs,
            or deferred handles resolving to those arrays.

        Raises
        ------
        ValueError
            If ``allsky_order`` cannot host a healpix8 subdivision of the
            preview tiles, or no input arrays are provided.
        """
        if allsky_order >= 8:
            raise ValueError(
                f"min_order={allsky_order} >= 8 leaves no healpix8 float arrays to composite; "
                "the HDR AllSky preview is built from the order-8 butler arrays."
            )
        tile_size = self.config.allsky_tilesize
        depth = 8 - allsky_order
        sub_px = tile_size >> depth
        if sub_px < 1:
            raise ValueError(
                f"allsky_tilesize {tile_size} is smaller than one {depth}-level subdivision slot "
                f"at order {allsky_order}; raise allsky_tilesize or lower min_order."
            )
        n_tiles = hpg.nside_to_npixel(hpg.order_to_nside(allsky_order))
        n_tiles_wide = int(np.floor(np.sqrt(n_tiles)))
        n_tiles_high = int(np.ceil(n_tiles / n_tiles_wide))

        # Nested-order subdivision template: (row, col) of every order-8
        # child inside an order-`allsky_order` parent. Same quadrant rule
        # as the HighOrderHipsTask tile pyramid: ids [b, c, d, e] of a
        # split pixel lay out as [[b, d], [c, e]].
        tmp_pixels = np.array([[0]])
        for _ in range(depth):
            tmp_array = np.zeros(np.array(tmp_pixels.shape) * 2)
            for ii in range(tmp_pixels.shape[0]):
                for jj in range(tmp_pixels.shape[1]):
                    tmp_array_view = tmp_array[ii * 2 : ii * 2 + 2, jj * 2 : jj * 2 + 2]
                    tmp_range_set = RangeSet(int(tmp_pixels[ii, jj]))
                    tmp_array_view[:, :] = (
                        np.array([x for x in range(*tmp_range_set.scaled(4)[0])], dtype=int)[[0, 2, 1, 3]]
                    ).reshape(2, 2)
            tmp_pixels = tmp_array
        sub_positions = {
            int(sub_id): (row, col)
            for row, rowvec in enumerate(tmp_pixels)
            for col, sub_id in enumerate(rowvec)
        }

        canvas = np.zeros((n_tiles_high * tile_size, n_tiles_wide * tile_size, 3), dtype=np.float32)
        filled = 0
        for pixel_id, data in input_hips:
            array = data.get() if hasattr(data, "get") else np.asarray(data)
            # Non-finite and negative pixels (e.g. warp undershoot) map to 0
            # — PQ cannot represent them.
            array = np.where(np.isfinite(array), array, 0.0)
            np.maximum(array, 0.0, out=array)
            reduced = downsample_hips_array(array, (sub_px, sub_px), "area")
            ancestor = pixel_id >> (2 * depth)
            row = ancestor // n_tiles_wide
            col = ancestor % n_tiles_wide
            sub_row, sub_col = sub_positions[pixel_id & ((1 << (2 * depth)) - 1)]
            canvas[
                row * tile_size + sub_row * sub_px : row * tile_size + (sub_row + 1) * sub_px,
                col * tile_size + sub_col * sub_px : col * tile_size + (sub_col + 1) * sub_px,
            ] = reduced
            filled += 1
        if filled == 0:
            raise ValueError(
                "is_hdr AllSky build received no healpix8 float arrays (input_hips is empty); "
                "refusing to write an empty preview."
            )

        uri = self.hips_base_path.join(f"Norder{allsky_order}", forceDirectory=True).join(
            f"Allsky.{self.config.file_extension}"
        )
        from stellaRGB._color_image import ColorImage  # lazy: drags in imagecodecs

        allsky_image = ColorImage(canvas, metadata={"is_hdr": True})
        with ResourcePath.temporary_uri(suffix=".avif") as temporary_uri:
            allsky_image.write_array_hdr_avif(
                temporary_uri.ospath,
                clamp_values=self.config.clamp_values,
                lum_norm=self.config.lum_norm,
            )
            uri.transfer_from(temporary_uri, transfer="copy", overwrite=True)
