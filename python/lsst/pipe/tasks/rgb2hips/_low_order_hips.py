from lsst.daf.butler import DeferredDatasetHandle

__all__ = ("LowOrderHipsTaskConnections", "LowOrderHipsTaskConfig", "LowOrderHipsTask")

import numpy as np
from skimage.transform import resize

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

from ._utils import _write_hips_image


class LowOrderHipsTaskConnections(PipelineTaskConnections, dimensions=("instrument",)):
    input_hips = Input(
        doc="Hips pixels at level 8 used to build higher orders",
        name="rgb_picture_hips8",
        storageClass="NumpyArray",
        multiple=True,
        deferLoad=True,
        dimensions=("healpix8",),
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
    color_ordering = Field[str](doc="The bands used to construct the input images", optional=False)
    file_extension = ChoiceField[str](
        doc="Extension for the presisted image, must be png or webp",
        allowed={"png": "Use the png image extension", "webp": "Use the webp image extension"},
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


class LowOrderHipsTask(PipelineTask):
    _DefaultName = "lowOrderHipsTask"
    ConfigClass = LowOrderHipsTaskConfig

    config: ConfigClass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hips_base_path = ResourcePath(self.config.hips_base_uri, forceDirectory=True)
        self.hips_base_path = self.hips_base_path.join(
            f"color_{self.config.color_ordering}", forceDirectory=True
        )

    def run(self, hpx_container) -> Struct:
        # do level 7 specifically, because we need to load in 8 handles
        for order in range(7, self.config.min_order - 1, -1):
            self.log.info("Processing order %d", order)
            hpx_next_mapping = self._create_sorted_container(hpx_container)

            hpx_next_container = []
            npix = 512
            # now loop over all order 9 pixels
            size_thresh = len(hpx_next_mapping) // 10
            size_counter = 0
            percent_counter = 0
            for hpx_next_id, hpx_next_items in hpx_next_mapping.items():
                if size_counter > size_thresh:
                    percent_counter += 10
                    self.log.info("Done %d percent", percent_counter)
                    size_counter = 0
                hpx_next_array = np.zeros((npix, npix, 3), dtype=np.float32)
                for img_prev, hpx_prev_id in hpx_next_items:
                    sub_index = hpx_prev_id - np.left_shift(hpx_next_id, 2)
                    if order == 7:
                        # These are saved out in float32 from the previous task
                        img_prev: NDArray = img_prev.get()
                    match sub_index:
                        case 0:
                            hpx_next_array[npix // 2 :, 0 : npix // 2] = img_prev
                        case 1:
                            hpx_next_array[0 : npix // 2 :, 0 : npix // 2] = img_prev
                        case 2:
                            hpx_next_array[npix // 2 :, npix // 2 :] = img_prev
                        case 3:
                            hpx_next_array[0 : npix // 2, npix // 2 :] = img_prev
                _write_hips_image(
                    hpx_next_array,
                    hpx_next_id,
                    order,
                    self.hips_base_path,
                    self.config.file_extension,
                    self.config.array_type,
                )
                size_counter += 1
                hpx_next_container.append((resize(hpx_next_array, (256, 256, 3)), hpx_next_id))
            hpx_container = hpx_next_container
        return Struct()

    def _create_sorted_container(
        self,
        hpx_container: Iterable[tuple[NDArray | DeferredDatasetHandle, int]],
    ) -> dict[int, Iterable[tuple[NDArray | DeferredDatasetHandle, int]]]:
        # do level 7 specifically, because we need to load in 8 handles
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
