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

__all__ = ("LowOrderHipsTaskConnections", "LowOrderHipsTaskConfig", "LowOrderHipsTask")

import numpy as np

from lsst.daf.butler import DeferredDatasetHandle
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
    QuantumContext,
    InputQuantizedConnection,
    OutputQuantizedConnection,
)

from lsst.pex.config import Field, ChoiceField
from lsst.pipe.base.connectionTypes import Input
from lsst.resources import ResourcePath

from collections.abc import Iterable

from numpy.typing import NDArray

from ._utils import _write_hips_image, downsample_hips_array


class LowOrderHipsTaskConnections(PipelineTaskConnections, dimensions=tuple()):
    input_hips = Input(
        doc="Hips pixels at level 8 used to build higher orders",
        name="rgb_picture_hips8",
        storageClass="NumpyArray",
        multiple=True,
        deferLoad=True,
        dimensions=("healpix8",),
    )

    def __init__(self, *, config: LowOrderHipsTaskConfig):
        # Set the quantum dimensions to whatever the minimum order healpix
        # to produce is.
        self.dimensions = set(
            (f"healpix{config.min_order}",),
        )


class LowOrderHipsTaskConfig(PipelineTaskConfig, pipelineConnections=LowOrderHipsTaskConnections):
    min_order = Field[int](
        doc="Minimum healpix order for HiPS tree.",
        default=3,
    )
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
        doc="Extension for the presisted image",
        allowed={
            "png": "Use the png image extension",
            "webp": "Use the webp image extension",
            "avif": "Use the avif image extension (PQ-encoded HDR tiles; requires is_hdr=True)",
        },
        default="png",
    )
    array_type = ChoiceField[str](
        doc="The dataset type for the output image array",
        default="uint8",
        allowed={
            "uint8": "Use 8 bit arrays, 255 max",
            "uint16": "Use 16 bit arrays, 65535 max",
            "half": "Use 16 bit float arrays, 1 max",
            "float": "Use 32 bit float arrays, 1 max",
        },
    )
    downsample_kernel = ChoiceField[str](
        doc=(
            "Kernel for the level-on-level cascade (order-7 tiles are "
            "assembled from the children, each order is then resampled to "
            "256x256 to feed the next order down). NOTE: unlike the "
            "high-order pyramid (single pass per level from full "
            "resolution), this cascade compounds the kernel once per order "
            "step, so its properties (edge erosion for \"area\", ringing "
            "and noise pass-through for \"lanczos\") accumulate with the "
            "number of levels. \"lanczos\" (default) preserves the "
            "historical edge contrast. \"area\" is the exact box-average "
            "prefilter: flux-conserving, alias-free, minimum-variance for "
            "white noise, but it smooths faint gradients."
        ),
        allowed={
            "area": "Exact box-average prefilter: anti-aliased, "
            "minimum-noise, smooths faint edges",
            "lanczos": "cv2 INTER_LANCZOS4 (no prefilter, default): "
            "sharper edges, passes ringing and high-frequency noise",
        },
        default="lanczos",
    )
    is_hdr = Field[bool](
        doc=(
            "The healpix8 input arrays carry linear HDR data (values may "
            "exceed 1) and tiles are written PQ-encoded as AVIF (requires "
            "file_extension=\"avif\"). Must match the HighOrderHipsTask "
            "setting that produced rgb_picture_hips8."
        ),
        default=False,
    )
    lum_norm = Field[float](
        doc=(
            "HDR only: the linear pixel value that encodes to lum_norm nits "
            "(paper white) under PQ. Default 100.0 (linear 1.0 -> 100 "
            "nits). Should match HighOrderHipsTask.lum_norm."
        ),
        default=100.0,
        check=lambda x: x > 0,
    )
    clamp_values = Field[float](
        doc=(
            "HDR only: optional upper clamp (in linear input units) applied "
            "before PQ encoding. None disables clamping."
        ),
        default=None,
        optional=True,
    )

    def validate(self):
        if self.min_order >= 8:
            raise ValueError("The minimum order must be less than 8.")
        if self.is_hdr and self.file_extension != "avif":
            raise ValueError(
                f"is_hdr=True requires file_extension='avif' (got '{self.file_extension}'): "
                "SDR tile formats clip to [0, 1] and would destroy the HDR signal."
            )
        if self.file_extension == "avif" and not self.is_hdr:
            raise ValueError(
                "file_extension='avif' requires is_hdr=True: the avif path is the PQ-encoded "
                "HDR writer; no SDR avif encoding is provided."
            )
        if self.is_hdr and self.downsample_kernel != "area":
            raise ValueError(
                "is_hdr=True requires downsample_kernel='area': lanczos undershoot produces "
                "negative linear values that PQ cannot encode, and the cascade compounds "
                "whatever kernel is chosen once per order."
            )
        return super().validate()


class LowOrderHipsTask(PipelineTask):
    """`PipelineTask` to create low order hips tiles.

    This task reads in healpix 8 tiles, which have already been down sampled,
    and assembles them into progressively lower hips order tiles.

    This task has special permission to write to locations outside the butler.
    Don't emulate this in other tasks.
    """

    _DefaultName = "lowOrderHipsTask"
    ConfigClass = LowOrderHipsTaskConfig

    config: ConfigClass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hips_base_path = ResourcePath(self.config.hips_base_uri, forceDirectory=True)
        self.hips_base_path = self.hips_base_path.join(
            f"color_{self.config.color_ordering}", forceDirectory=True
        )

    def run(self, hpx_container: Iterable[tuple[DeferredDatasetHandle, int]]) -> Struct:
        """Produce Hips images with hips order 8 inputs to the configured min_order.

        Parameters
        ----------
        hpx_container : `Iterable` of `tuple` of `DeferredDatasetHanle`, `int`
            This is an iterable of handles to already down-sampled hpx order 8
            arrays and their corresponding order 8 pixel id.

        Returns
        -------
        result : `Struct`
            This tasks does not produce an output, so will return an empty `Struct`

        """
        # loop over each order, assembling the previous order tiles into
        # an array, and writing the image. Resample each image smaller,
        # and continue downward in order.
        # This must 7 here based on the outputs of HighOrderHipsTask being
        # healpix order 8 pixels.
        for order in range(7, self.config.min_order - 1, -1):
            self.log.info("Processing order %d", order)
            # sort the previous order's pixels into a mapping with keys of
            # this order's pixel to the corresponding previous orders pixels
            # that are contained within that key.
            hpx_next_mapping = self._create_sorted_container(hpx_container)

            hpx_next_container = []
            npix = 512
            size_thresh = len(hpx_next_mapping) // 10
            size_counter = 0
            percent_counter = 0
            for hpx_next_id, hpx_next_items in hpx_next_mapping.items():
                # Print out a log message every so often for a liveness
                # check
                if size_counter > size_thresh:
                    percent_counter += 10
                    self.log.info("Done %d percent", percent_counter)
                    size_counter = 0
                # allocate a container for the pixel being assembled
                hpx_next_array = np.zeros((npix, npix, 3), dtype=np.float32)
                for img_prev, hpx_prev_id in hpx_next_items:
                    if order == 7:
                        # These are saved out in float32 from the previous task
                        img_prev: NDArray = img_prev.get()
                    # determine which sub pixel quadrant this belongs to in the next orders
                    # pixel and assign.
                    sub_index = hpx_prev_id - np.left_shift(hpx_next_id, 2)
                    match sub_index:
                        case 0:
                            hpx_next_array[0 : npix // 2 :, 0 : npix // 2] = img_prev
                        case 1:
                            hpx_next_array[npix // 2 :, 0 : npix // 2] = img_prev
                        case 2:
                            hpx_next_array[0 : npix // 2, npix // 2 :] = img_prev
                        case 3:
                            hpx_next_array[npix // 2 :, npix // 2 :] = img_prev
                # Write out the hips image
                _write_hips_image(
                    hpx_next_array,
                    hpx_next_id,
                    order,
                    self.hips_base_path,
                    self.config.file_extension,
                    self.config.array_type,
                    is_hdr=self.config.is_hdr,
                    lum_norm=self.config.lum_norm,
                    clamp_values=self.config.clamp_values,
                )
                size_counter += 1

                # resample the image to a smaller grid and store it for the next order
                zoomed = downsample_hips_array(
                    hpx_next_array, (256, 256), self.config.downsample_kernel
                )

                hpx_next_container.append((zoomed, hpx_next_id))
            hpx_container = hpx_next_container
        return Struct()

    def _create_sorted_container(
        self,
        hpx_container: Iterable[tuple[NDArray | DeferredDatasetHandle, int]],
    ) -> dict[int, Iterable[tuple[NDArray | DeferredDatasetHandle, int]]]:
        """Sort a list of [images (or handels), hpx_id] into corresponding pixels at a higher order."""
        hpx_output_mapping = {}
        for pair in hpx_container:
            hpx_output_id = np.right_shift(pair[1], 2)
            hpx_output_container = hpx_output_mapping.setdefault(hpx_output_id, [])
            hpx_output_container.append(pair)
        return hpx_output_mapping

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        # get the hips handles and their pixel
        hpx_container = []
        for ref in inputRefs.input_hips:
            hpx_container.append((butlerQC.get(ref), ref.dataId["healpix8"]))

        outputs = self.run(hpx_container)
        butlerQC.put(outputs, outputRefs)
