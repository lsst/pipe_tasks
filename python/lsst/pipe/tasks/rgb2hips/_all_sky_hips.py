from __future__ import annotations

__all__ = ("AllSkyHipsTaskConnections", "AllSkyHipsTaskConfig", "AllSkyHipsTask")

from PIL import Image
import io
import math
import re
import hpgeom as hpg
import numpy as np
import healsparse as hsp
from datetime import datetime
from collections.abc import Iterable
from astropy.io import fits


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
from ..healSparseMapping import _is_power_of_two


class AllSkyHipsTaskConnections(
    PipelineTaskConnections,
    dimensions=tuple(
        "instrument",
    ),
    defaultTemplates={"task_label": "lowOrderHipsTask"},
):
    low_order_metadata = Input(
        doc="Metadata produced by the LowOrderHipsTask",
        name="{task_label}_metadata",
        storageClass="TaskMetadata",
        multiple=True,
        deferLoad=True,
        dimensions=tuple(
            "instrument",
        ),
    )
    input_hips = Input(
        doc="Hips pixels at level 8 used to build higher orders",
        name="rgb_picture_hips8",
        storageClass="NumpyArray",
        multiple=True,
        deferLoad=True,
        dimensions=("healpix8",),
    )


class AllSkyHipsTaskConfig(PipelineTaskConfig, pipelineConnections=AllSkyHipsTaskConnections):
    hips_base_uri = Field[str](
        doc="URI to HiPS base for output.",
        optional=False,
    )
    color_ordering = Field[str](doc="The bands used to construct the input images", optional=False)
    file_extension = ChoiceField[str](
        doc="Extension for the presisted image, must be png or webp",
        allowed={"png": "Use the png image extension", "webp": "Use the webp image extension"},
        default="png",
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
    redest_band = Field[str](
        doc="Band corresponding to reddes color in image",
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
    hips_tile_width = Field[int](doc="The width of one hips tile", default=512)
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


class AllSkyHipsTask(PipelineTask):
    _DefaultName = "_allSkyHipsTask"
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
        # get the healpix8 pixel ids
        hpx8_pixels = []
        for ref in inputRefs.input_hips:
            hpx8_pixels.append((ref.dataId["healpix8"]))

        # turn that into the hpx11 pixels that were produced
        hpx8_rangeset = RangeSet(hpx8_pixels)
        hpx11_rangeset = hpx8_rangeset.scaled(4**3)
        hpx11_pixels = set()
        for begin, end in hpx11_rangeset:
            hpx11_pixels.update(range(begin, end))
        hpx11_pixels = list(hpx11_pixels)

        low_order_metadata = butlerQC.get(inputRefs.low_order_metadata)

        outputs = self.run(low_order_metadata, hpx11_pixels)
        butlerQC.put(outputs, outputRefs)

    def run(self, low_order_metadata: Iterable[TaskMetadata], hpx11_pixels) -> Struct:
        self._write_properties_and_moc(
            self.config.max_order, hpx11_pixels, self.config.shift_order, self.config.color_ordering
        )
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
                _write_property(fh, "hips_tile_width", str(self.config.hips_tile_width))
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
                if self.config.redest_band in properties_config.spectral_ranges:
                    em_max = properties_config.spectral_ranges[self.config.redest_band].lambda_max / 1e9
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

        # Make the initial list of UNIQ pixels
        uniq = 4 * (4**max_order) + pixels

        # Make a healsparse map which provides easy degrade/comparisons.
        hspmap = hsp.HealSparseMap.make_empty(2**min_uniq_order, 2**max_order, dtype=np.float32)
        hspmap[pixels] = 1.0

        # Loop over orders, degrade each time, and look for pixels with full coverage.
        for uniq_order in range(max_order - 1, min_uniq_order - 1, -1):
            hspmap = hspmap.degrade(2**uniq_order, reduction="sum")
            pix_shift = np.right_shift(pixels, 2 * (max_order - uniq_order))
            # Check if any of the pixels at uniq_order have full coverage.
            (covered,) = np.isclose(hspmap[pix_shift], 4 ** (max_order - uniq_order)).nonzero()
            if covered.size == 0:
                # No pixels at uniq_order are fully covered, we're done.
                break
            # Replace the UNIQ pixels that are fully covered.
            uniq[covered] = 4 * (4**uniq_order) + pix_shift[covered]

        # Remove duplicate pixels.
        uniq = np.unique(uniq)

        # Output to fits.
        tbl = np.zeros(uniq.size, dtype=[("UNIQ", "i8")])
        tbl["UNIQ"] = uniq

        order = np.log2(tbl["UNIQ"] // 4).astype(np.int32) // 2
        moc_order = np.max(order)

        hdu = fits.BinTableHDU(tbl)
        hdu.header["PIXTYPE"] = "HEALPIX"
        hdu.header["ORDERING"] = "NUNIQ"
        hdu.header["COORDSYS"] = "C"
        hdu.header["MOCORDER"] = moc_order
        hdu.header["MOCTOOL"] = "lsst.pipe.tasks.hips.GenerateHipsTask"

        uri = self.hips_base_path.join("Moc.fits")

        with ResourcePath.temporary_uri(suffix=uri.getExtension()) as temporary_uri:
            hdu.writeto(temporary_uri.ospath)

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

        png_uris = list(
            ResourcePath.findFileResources(
                candidates=[allsky_order_uri],
                file_filter=pixel_regex,
            )
        )

        for png_uri in png_uris:
            matches = re.match(pixel_regex, png_uri.basename())
            pix_num = int(matches.group(1))
            tile_image = Image.open(io.BytesIO(png_uri.read()))
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
