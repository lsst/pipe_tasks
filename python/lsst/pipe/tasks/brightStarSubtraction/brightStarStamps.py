# This file is part of meas_algorithms.
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

"""Collection of small images (postage stamps) centered on bright stars."""

from __future__ import annotations

__all__ = [
    "BrightStarStampInfo",
    "BrightStarStampSerializationModel",
    "BrightStarStampsSerializationModel",
    "BrightStarStamp",
    "BrightStarStamps",
]

import functools
from collections.abc import Sequence
from types import EllipsisType
from typing import Any

from astro_metadata_translator import ObservationInfo
from pydantic import BaseModel, Field

from lsst.images import (
    Box,
    Image,
    ImageSerializationModel,
    Mask,
    MaskedImage,
    MaskedImageSerializationModel,
    MaskSchema,
    Projection,
)
from lsst.images.serialization import ArchiveTree, InputArchive, MetadataValue, OutputArchive, Quantity
from lsst.images.utils import is_none


class BrightStarStampInfo(BaseModel):
    """Information about a bright star in a `BrightStarStamp`.

    Attributes
    ----------
    visit : `int`, optional
        The visit during which the bright star was observed.
    detector : `int`, optional
        The detector on which the bright star was observed.
    ref_id : `int`, optional
        The reference catalog ID for the bright star.
    ref_mag : `float`, optional
        The reference magnitude for the bright star.
    position_x : `float`, optional
        The x-coordinate of the bright star in the focal plane.
    position_y : `float`, optional
        The y-coordinate of the bright star in the focal plane.
    focal_plane_radius : `~lsst.images.utils.Quantity`, optional
        The radius of the bright star from the center of the focal plane.
    focal_plane_angle : `~lsst.images.utils.Quantity`, optional
        The angle of the bright star in the focal plane,
        measured from the +x axis.
    """

    visit: int | None = None
    detector: int | None = None
    ref_id: int | None = None
    ref_mag: float | None = None
    position_x: float | None = None
    position_y: float | None = None
    focal_plane_radius: Quantity | None = None
    focal_plane_angle: Quantity | None = None

    def __str__(self) -> str:
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"BrightStarStampInfo({attrs})"

    __repr__ = __str__


class BrightStarStampSerializationModel[P: BaseModel](MaskedImageSerializationModel[P]):
    """A Pydantic model used to represent a serialized `BrightStarStamp`."""

    psf_kernel_image: ImageSerializationModel[P] | None = Field(
        default=None,
        exclude_if=is_none,
        description="Kernel image of the PSF at the stamp center.",
    )
    stamp_info: BrightStarStampInfo = Field(description="Information about the bright star in the stamp.")


class BrightStarStampsSerializationModel[P: BaseModel](ArchiveTree):
    """A Pydantic model used to represent serialized `BrightStarStamps`."""

    stamps: list[BrightStarStampSerializationModel[P]] = Field(
        default_factory=list,
        description="The bright star stamps in this collection.",
    )


class BrightStarStamp(MaskedImage):
    """A postage stamp centered on a bright star, with associated metadata.

    Parameters
    ----------
    image : `~lsst.images.Image`
        The main data image for this bright star stamp.
    mask : `~lsst.images.Mask`, optional
        Bitmask that annotates the main image's pixels.
    variance : `~lsst.images.Image`, optional
        Per-pixel variance estimates for the image.
    mask_schema : `~lsst.images.MaskSchema`, optional
        Schema for the mask, required if a mask is provided.
    projection : `~lsst.images.Projection`, optional
        Projection to map pixels to the sky.
    obs_info : `~astro_metadata_translator.ObservationInfo`, optional
        Standardized description of visit metadata.
    metadata : `dict` [`str`, `MetadataValue`], optional
        Additional metadata to associate with this stamp.
    psf_kernel_image : `~lsst.images.Image`, optional
        Kernel image of the PSF at the stamp center.
    stamp_info : `BrightStarStampInfo`, optional
        Information about the bright star in the stamp.

    Attributes
    ----------
    psf_kernel_image : `~lsst.images.Image`
        Kernel image of the PSF at the stamp center.
    stamp_info : `BrightStarStampInfo`
        Information about the bright star in this stamp.
    """

    def __init__(
        self,
        image: Image,
        *,
        mask: Mask | None = None,
        variance: Image | None = None,
        mask_schema: MaskSchema | None = None,
        projection: Projection | None = None,
        obs_info: ObservationInfo | None = None,
        metadata: dict[str, MetadataValue] | None = None,
        psf_kernel_image: Image | None = None,
        stamp_info: BrightStarStampInfo | None = None,
    ):
        super().__init__(
            image,
            mask=mask,
            variance=variance,
            mask_schema=mask_schema,
            projection=projection,
            obs_info=obs_info,
            metadata=metadata,
        )

        self._psf_kernel_image = psf_kernel_image
        self._stamp_info = stamp_info or BrightStarStampInfo()

    def __getitem__(self, bbox: Box | EllipsisType) -> BrightStarStamp:
        if bbox is ...:
            return self
        super().__getitem__(bbox)
        return self._transfer_metadata(
            BrightStarStamp(
                # Projection and obs_info propagate from the image.
                self.image[bbox],
                mask=self.mask[bbox],
                variance=self.variance[bbox],
                psf_kernel_image=self.psf_kernel_image,
                stamp_info=self.stamp_info,
            ),
            bbox=bbox,
        )

    def __str__(self) -> str:
        return f"BrightStarStamp({self.image!s}, {list(self.mask.schema.names)}, {self.stamp_info})"

    def __repr__(self) -> str:
        return (
            f"BrightStarStamp({self.image!r}, mask_schema={self.mask.schema!r}, "
            f"stamp_info={self.stamp_info!r})"
        )

    @property
    def psf_kernel_image(self) -> Image:
        """Kernel image of the PSF at the stamp center."""
        if self._psf_kernel_image is None:
            raise RuntimeError("No PSF kernel image is attached to this BrightStarStamp.")
        return self._psf_kernel_image

    @property
    def stamp_info(self) -> BrightStarStampInfo:
        """Return the BrightStarStampInfo associated with this stamp."""
        return self._stamp_info

    def copy(self) -> BrightStarStamp:
        """Deep-copy the bright star stamp, metadata, and stamp info."""
        return self._transfer_metadata(
            BrightStarStamp(
                image=self._image.copy(),
                mask=self._mask.copy(),
                variance=self._variance.copy(),
                psf_kernel_image=self._psf_kernel_image,
                stamp_info=self._stamp_info.model_copy(),
            ),
            copy=True,
        )

    def serialize(self, archive: OutputArchive[Any]) -> BrightStarStampSerializationModel:
        masked_image_model = super().serialize(archive)
        serialized_psf_kernel_image = (
            archive.serialize_direct(
                "psf_kernel_image",
                functools.partial(self._psf_kernel_image.serialize, save_projection=False),
            )
            if self._psf_kernel_image is not None
            else None
        )
        return BrightStarStampSerializationModel(
            **masked_image_model.model_dump(),
            psf_kernel_image=serialized_psf_kernel_image,
            stamp_info=self.stamp_info,
        )

    @staticmethod
    def deserialize(
        model: BrightStarStampSerializationModel[Any],
        archive: InputArchive[Any],
        *,
        bbox: Box | None = None,
    ) -> BrightStarStamp:
        masked_image = MaskedImage.deserialize(model, archive, bbox=bbox)
        psf_kernel_image = (
            Image.deserialize(model.psf_kernel_image, archive) if model.psf_kernel_image is not None else None
        )
        return BrightStarStamp(
            masked_image.image,
            mask=masked_image.mask,
            variance=masked_image.variance,
            psf_kernel_image=psf_kernel_image,
            stamp_info=model.stamp_info,
        )._finish_deserialize(model)


class BrightStarStamps(Sequence[BrightStarStamp]):
    """A collection of bright star stamps.

    Parameters
    ----------
    stamps : `Iterable` [`BrightStarStamp`]
        Collection of `BrightStarStamp` instances.
    metadata : `dict` [`str`, `MetadataValue`], optional
        Global metadata associated with the collection.

    Attributes
    ----------
    metadata : `dict` [`str`, `MetadataValue`]
        Global metadata associated with the collection.
    ref_id_map : `dict` [`int`, `BrightStarStamp`]
        A mapping from reference IDs to `BrightStarStamp` objects.
        Only includes stamps with valid reference IDs.
    """

    def __init__(
        self,
        stamps: Sequence[BrightStarStamp],
        metadata: dict[str, MetadataValue] | None = None,
    ):
        self._stamps = list(stamps)
        self._metadata = {} if metadata is None else dict(metadata)
        self._ref_id_map = {
            stamp.stamp_info.ref_id: stamp for stamp in self if stamp.stamp_info.ref_id is not None
        }

    def __len__(self):
        return len(self._stamps)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return BrightStarStamps(self._stamps[index], metadata=self._metadata)
        return self._stamps[index]

    def __iter__(self):
        return iter(self._stamps)

    def __str__(self) -> str:
        return f"BrightStarStamps(length={len(self)})"

    __repr__ = __str__

    @property
    def metadata(self):
        """Return the collection's global metadata as a dict."""
        return self._metadata

    @property
    def ref_id_map(self):
        """Map reference IDs to `BrightStarStamp` objects."""
        return self._ref_id_map

    def serialize(self, archive: OutputArchive[Any]) -> BrightStarStampsSerializationModel:
        return BrightStarStampsSerializationModel(
            stamps=[
                archive.serialize_direct(f"stamp_{index}", stamp.serialize)
                for index, stamp in enumerate(self._stamps)
            ],
            metadata=self._metadata,
        )

    @staticmethod
    def deserialize(
        model: BrightStarStampsSerializationModel[Any],
        archive: InputArchive[Any],
    ) -> BrightStarStamps:
        return BrightStarStamps(
            [BrightStarStamp.deserialize(stamp_model, archive) for stamp_model in model.stamps],
            metadata=model.metadata,
        )

    @staticmethod
    def _get_archive_tree_type[P: BaseModel](
        pointer_type: type[P],
    ) -> type[BrightStarStampsSerializationModel[P]]:
        return BrightStarStampsSerializationModel[pointer_type]
