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

__all__ = [
    "ExtendedPsfCandidateInfo",
    "ExtendedPsfCandidateSerializationModel",
    "ExtendedPsfCandidatesSerializationModel",
    "ExtendedPsfCandidate",
    "ExtendedPsfCandidates",
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
    fits,
)
from lsst.images.serialization import ArchiveTree, InputArchive, MetadataValue, OutputArchive, Quantity
from lsst.images.utils import is_none
from lsst.resources import ResourcePathExpression


class ExtendedPsfCandidateInfo(BaseModel):
    """Information about a star in an `ExtendedPsfCandidate`.

    Attributes
    ----------
    visit : `int`, optional
        The visit during which the star was observed.
    detector : `int`, optional
        The detector on which the star was observed.
    ref_id : `int`, optional
        The reference catalog ID for the star.
    ref_mag : `float`, optional
        The reference magnitude for the star.
    position_x : `float`, optional
        The x-coordinate of the star in the focal plane.
    position_y : `float`, optional
        The y-coordinate of the star in the focal plane.
    focal_plane_radius : `~lsst.images.utils.Quantity`, optional
        The radius of the star from the center of the focal plane.
    focal_plane_angle : `~lsst.images.utils.Quantity`, optional
        The angle of the star in the focal plane, measured from the +x axis.
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
        return f"ExtendedPsfCandidateInfo({attrs})"

    __repr__ = __str__


class ExtendedPsfCandidateSerializationModel[P: BaseModel](MaskedImageSerializationModel[P]):
    """A Pydantic model to represent a serialized `ExtendedPsfCandidate`."""

    psf_kernel_image: ImageSerializationModel[P] | None = Field(
        default=None,
        exclude_if=is_none,
        description="Kernel image of the PSF at the cutout center.",
    )
    star_info: ExtendedPsfCandidateInfo = Field(description="Information about the star in the cutout.")


class ExtendedPsfCandidatesSerializationModel[P: BaseModel](ArchiveTree):
    """A Pydantic model to represent serialized `ExtendedPsfCandidates`."""

    candidates: list[ExtendedPsfCandidateSerializationModel[P]] = Field(
        default_factory=list,
        description="The candidate cutouts in this collection.",
    )


class ExtendedPsfCandidate(MaskedImage):
    """A cutout centered on a star, with associated metadata.

    Parameters
    ----------
    image : `~lsst.images.Image`
        The main data image for this star cutout.
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
        Additional metadata to associate with this cutout.
    psf_kernel_image : `~lsst.images.Image`, optional
        Kernel image of the PSF at the cutout center.
    star_info : `ExtendedPsfCandidateInfo`, optional
        Information about the star in the cutout.

    Attributes
    ----------
    psf_kernel_image : `~lsst.images.Image`
        Kernel image of the PSF at the cutout center.
    star_info : `ExtendedPsfCandidateInfo`
        Information about the star in this cutout.
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
        star_info: ExtendedPsfCandidateInfo | None = None,
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
        self._star_info = star_info or ExtendedPsfCandidateInfo()

    def __getitem__(self, bbox: Box | EllipsisType) -> ExtendedPsfCandidate:
        if bbox is ...:
            return self
        super().__getitem__(bbox)
        return self._transfer_metadata(
            ExtendedPsfCandidate(
                # Projection and obs_info propagate from the image.
                self.image[bbox],
                mask=self.mask[bbox],
                variance=self.variance[bbox],
                psf_kernel_image=self.psf_kernel_image,
                star_info=self.star_info,
            ),
            bbox=bbox,
        )

    def __str__(self) -> str:
        return f"ExtendedPsfCandidate({self.image!s}, {list(self.mask.schema.names)}, {self.star_info})"

    def __repr__(self) -> str:
        return (
            f"ExtendedPsfCandidate({self.image!r}, mask_schema={self.mask.schema!r}, "
            f"star_info={self.star_info!r})"
        )

    @property
    def psf_kernel_image(self) -> Image:
        """Kernel image of the PSF at the cutout center."""
        if self._psf_kernel_image is None:
            raise RuntimeError("No PSF kernel image is attached to this ExtendedPsfCandidate.")
        return self._psf_kernel_image

    @property
    def star_info(self) -> ExtendedPsfCandidateInfo:
        """Return the ExtendedPsfCandidateInfo associated with this star."""
        return self._star_info

    def copy(self) -> ExtendedPsfCandidate:
        """Deep-copy the star cutout, metadata, and star info."""
        return self._transfer_metadata(
            ExtendedPsfCandidate(
                image=self._image.copy(),
                mask=self._mask.copy(),
                variance=self._variance.copy(),
                psf_kernel_image=self._psf_kernel_image,
                star_info=self._star_info.model_copy(),
            ),
            copy=True,
        )

    def serialize(self, archive: OutputArchive[Any]) -> ExtendedPsfCandidateSerializationModel:
        masked_image_model = super().serialize(archive)
        serialized_psf_kernel_image = (
            archive.serialize_direct(
                "psf_kernel_image",
                functools.partial(self._psf_kernel_image.serialize, save_projection=False),
            )
            if self._psf_kernel_image is not None
            else None
        )
        return ExtendedPsfCandidateSerializationModel(
            **masked_image_model.model_dump(),
            psf_kernel_image=serialized_psf_kernel_image,
            star_info=self.star_info,
        )

    @staticmethod
    def _get_archive_tree_type[P: BaseModel](
        pointer_type: type[P],
    ) -> type[ExtendedPsfCandidateSerializationModel[P]]:
        return ExtendedPsfCandidateSerializationModel[pointer_type]

    @staticmethod
    def deserialize(
        model: ExtendedPsfCandidateSerializationModel[Any],
        archive: InputArchive[Any],
        *,
        bbox: Box | None = None,
    ) -> ExtendedPsfCandidate:
        masked_image = MaskedImage.deserialize(model, archive, bbox=bbox)
        psf_kernel_image = (
            Image.deserialize(model.psf_kernel_image, archive) if model.psf_kernel_image is not None else None
        )
        return ExtendedPsfCandidate(
            masked_image.image,
            mask=masked_image.mask,
            variance=masked_image.variance,
            psf_kernel_image=psf_kernel_image,
            star_info=model.star_info,
        )._finish_deserialize(model)


class ExtendedPsfCandidates(Sequence[ExtendedPsfCandidate]):
    """A collection of star cutouts.

    Parameters
    ----------
    candidates : `Iterable` [`ExtendedPsfCandidate`]
        Collection of `ExtendedPsfCandidate` instances.
    metadata : `dict` [`str`, `MetadataValue`], optional
        Global metadata associated with the collection.

    Attributes
    ----------
    metadata : `dict` [`str`, `MetadataValue`]
        Global metadata associated with the collection.
    ref_id_map : `dict` [`int`, `ExtendedPsfCandidate`]
        A mapping from reference IDs to `ExtendedPsfCandidate` objects.
        Only includes candidates with valid reference IDs.
    """

    def __init__(
        self,
        candidates: Sequence[ExtendedPsfCandidate],
        metadata: dict[str, MetadataValue] | None = None,
    ):
        self._candidates = list(candidates)
        self._metadata = {} if metadata is None else dict(metadata)
        self._ref_id_map = {
            candidate.star_info.ref_id: candidate
            for candidate in self
            if candidate.star_info.ref_id is not None
        }

    def __len__(self):
        return len(self._candidates)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return ExtendedPsfCandidates(self._candidates[index], metadata=self._metadata)
        return self._candidates[index]

    def __iter__(self):
        return iter(self._candidates)

    def __str__(self) -> str:
        return f"ExtendedPsfCandidates(length={len(self)})"

    __repr__ = __str__

    @property
    def metadata(self):
        """Return the collection's global metadata as a dict."""
        return self._metadata

    @property
    def ref_id_map(self):
        """Map reference IDs to `ExtendedPsfCandidate` objects."""
        return self._ref_id_map

    @classmethod
    def read_fits(cls, url: ResourcePathExpression) -> ExtendedPsfCandidates:
        """Read a collection from a FITS file.

        Parameters
        ----------
        url
            URL of the file to read; may be any type supported by
            `lsst.resources.ResourcePath`.
        """
        return fits.read(cls, url).deserialized

    def write_fits(self, filename: str) -> None:
        """Write the collection to a FITS file.

        Parameters
        ----------
        filename
            Name of the file to write to. Must not already exist.
        """
        fits.write(self, filename)

    def serialize(self, archive: OutputArchive[Any]) -> ExtendedPsfCandidatesSerializationModel:
        return ExtendedPsfCandidatesSerializationModel(
            candidates=[
                archive.serialize_direct(f"candidate_{index}", candidate.serialize)
                for index, candidate in enumerate(self._candidates)
            ],
            metadata=self._metadata,
        )

    @staticmethod
    def deserialize(
        model: ExtendedPsfCandidatesSerializationModel[Any],
        archive: InputArchive[Any],
    ) -> ExtendedPsfCandidates:
        return ExtendedPsfCandidates(
            [
                ExtendedPsfCandidate.deserialize(candidate_model, archive)
                for candidate_model in model.candidates
            ],
            metadata=model.metadata,
        )

    @staticmethod
    def _get_archive_tree_type[P: BaseModel](
        pointer_type: type[P],
    ) -> type[ExtendedPsfCandidatesSerializationModel[P]]:
        return ExtendedPsfCandidatesSerializationModel[pointer_type]
