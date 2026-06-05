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

__all__ = (
    "ExtendedPsfImageInfo",
    "ExtendedPsfImageSerializationModel",
    "ExtendedPsfImage",
)

import functools
from types import EllipsisType
from typing import Any, ClassVar

import numpy as np
from astropy.units import UnitBase
from pydantic import BaseModel, Field

from lsst.images import Box, GeneralizedImage, Image, ImageSerializationModel
from lsst.images.serialization import ArchiveTree, InputArchive, MetadataValue, OutputArchive

from .extended_psf_fit import ExtendedPsfFit, ExtendedPsfMoffatFit


class ExtendedPsfImageInfo(BaseModel):
    """Additional information about an `ExtendedPsfImage`.

    Attributes
    ----------
    n_stars : `int`, optional
        Number of stars used to construct the extended PSF image.
    """

    n_stars: int | None = None

    def __str__(self) -> str:
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"ExtendedPsfImageInfo({attrs})"

    __repr__ = __str__


class ExtendedPsfImage(GeneralizedImage):
    """A multi-plane image with data (image) and variance planes, and the
    results of a profile fit to the image.

    Parameters
    ----------
    image : `~lsst.images.Image`
        The main image plane.
    variance : `~lsst.images.Image`, optional
        The per-pixel uncertainty of the main image as an image of variance
        values. Must have the same bounding box as ``image`` if provided, and
        its units must be the square of ``image.unit`` or `None`.
        Values default to ``1.0``. Any attached projection is replaced
        (possibly by `None`).
    info : `ExtendedPsfImageInfo`, optional
        Additional information about how the extended PSF image was
        constructed.
    fit : `ExtendedPsfFit`, optional
        The results of a profile fit to the image.
    metadata : `dict` [`str`, `MetadataValue`], optional
        Arbitrary flexible metadata to associate with the image.

    Attributes
    ----------
    image : `~lsst.images.Image`
        The main image plane.
    variance : `~lsst.images.Image`
        The per-pixel uncertainty of the main image as an image of variance
        values.
    bbox : `~lsst.images.Box`
        The bounding box shared by both image planes.
    unit : `astropy.units.Unit` or `None`
        The units of the image plane, or `None` if the image is dimensionless.
    projection : `None`
        The projection that maps the pixel grid to the sky. Always `None` for
        `ExtendedPsfImage`.
    info : `ExtendedPsfImageInfo`
        Additional information about how the extended PSF image was
        constructed.
    fit : `ExtendedPsfFit`
        The results of a profile fit to the image.
    """

    def __init__(
        self,
        image: Image,
        *,
        variance: Image | None = None,
        info: ExtendedPsfImageInfo | None = None,
        fit: ExtendedPsfFit | None = None,
        metadata: dict[str, MetadataValue] | None = None,
    ):
        super().__init__(metadata)
        if variance is None:
            variance = Image(
                1.0,
                dtype=np.float32,
                bbox=image.bbox,
                unit=None if image.unit is None else image.unit**2,
            )
        else:
            if image.bbox != variance.bbox:
                raise ValueError(f"Image ({image.bbox}) and variance ({variance.bbox}) bboxes do not agree.")
            if image.unit is None:
                if variance.unit is not None:
                    raise ValueError(f"Image has no units but variance does ({variance.unit}).")
            elif variance.unit is None:
                variance = variance.view(unit=image.unit**2)
            elif variance.unit != image.unit**2:
                raise ValueError(
                    f"Variance unit ({variance.unit}) should be the square of the image unit ({image.unit})."
                )
        if info is None:
            info = ExtendedPsfImageInfo()
        if fit is None:
            fit = ExtendedPsfFit(chi2=np.nan, reduced_chi2=np.nan)
        self._image = image
        self._variance = variance
        self._info = info
        self._fit = fit

    @property
    def image(self) -> Image:
        """The main image plane (`Image`)."""
        return self._image

    @property
    def variance(self) -> Image:
        """The variance plane (`Image`)."""
        return self._variance

    @property
    def bbox(self) -> Box:
        """The bounding box shared by both image planes (`Box`)."""
        return self._image.bbox

    @property
    def unit(self) -> UnitBase | None:
        """The units of the image plane (`astropy.units.Unit` | `None`)."""
        return self._image.unit

    @property
    def projection(self) -> None:
        """The projection that maps the pixel grid to the sky.

        ExtendedPsfImage does not support attached projections,
        so this always returns `None`.
        """
        return None

    @property
    def info(self) -> ExtendedPsfImageInfo:
        """Additional information about the image (`ExtendedPsfImageInfo`)."""
        return self._info

    @property
    def fit(self) -> ExtendedPsfFit:
        """The results of a profile fit to the image."""
        return self._fit

    def __getitem__(self, bbox: Box | EllipsisType) -> ExtendedPsfImage:
        super().__getitem__(bbox)
        if bbox is ...:
            return self
        return self._transfer_metadata(
            ExtendedPsfImage(
                self.image[bbox],
                variance=self.variance[bbox],
                info=self.info,
                fit=self.fit,
            ),
            bbox=bbox,
        )

    def __setitem__(self, bbox: Box | EllipsisType, value: ExtendedPsfImage) -> None:
        self._image[bbox] = value.image
        self._variance[bbox] = value.variance

    def __str__(self) -> str:
        return f"ExtendedPsfImage({self.image!s}, info={self.info!r}, fit={self.fit!r})"

    __repr__ = __str__

    def copy(self) -> ExtendedPsfImage:
        """Deep-copy the profile image and metadata."""
        return self._transfer_metadata(
            ExtendedPsfImage(
                image=self._image.copy(),
                variance=self._variance.copy(),
                info=self._info.model_copy(),
                fit=self._fit.model_copy(),
            ),
            copy=True,
        )

    def serialize(self, archive: OutputArchive[Any]) -> ExtendedPsfImageSerializationModel:
        """Serialize the Extended PSF image to an output archive.

        Parameters
        ----------
        archive
            Archive to write to.
        """
        serialized_image = archive.serialize_direct(
            "image", functools.partial(self.image.serialize, save_projection=False)
        )
        serialized_variance = archive.serialize_direct(
            "variance", functools.partial(self.variance.serialize, save_projection=False)
        )
        serialized_info = self.info
        serialized_fit = self.fit
        return ExtendedPsfImageSerializationModel(
            image=serialized_image,
            variance=serialized_variance,
            info=serialized_info,
            fit=serialized_fit,
            metadata=self.metadata,
        )

    @staticmethod
    def deserialize(
        model: ExtendedPsfImageSerializationModel[Any], archive: InputArchive[Any], *, bbox: Box | None = None
    ) -> ExtendedPsfImage:
        """Deserialize an image from an input archive.

        Parameters
        ----------
        model
            A Pydantic model representation of the image, holding references
            to data stored in the archive.
        archive
            Archive to read from.
        bbox
            Bounding box of a subimage to read instead.
        """
        return model.deserialize(archive, bbox=bbox)

    @staticmethod
    def _get_archive_tree_type[P: BaseModel](
        pointer_type: type[P],
    ) -> type[ExtendedPsfImageSerializationModel[P]]:
        """Return the serialization model type for this object for an archive
        type that uses the given pointer type.
        """
        return ExtendedPsfImageSerializationModel[pointer_type]  # type: ignore


class ExtendedPsfImageSerializationModel[P: BaseModel](ArchiveTree):
    """A Pydantic model used to represent a serialized `ExtendedPsfImage`."""

    SCHEMA_NAME: ClassVar[str] = "extended_psf_image"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = ExtendedPsfImage

    image: ImageSerializationModel[P] = Field(
        description="The main data image.",
    )
    variance: ImageSerializationModel[P] = Field(
        description="Per-pixel variance estimates for the main image."
    )
    info: ExtendedPsfImageInfo = Field(
        description="Additional information about the extended PSF image.",
    )
    fit: ExtendedPsfMoffatFit | ExtendedPsfFit = Field(
        description="The results of an extended PSF fit to the image.",
    )

    @property
    def bbox(self) -> Box:
        """The bounding box of the image."""
        return self.image.bbox

    def deserialize(self, archive: InputArchive[Any], *, bbox: Box | None = None) -> ExtendedPsfImage:
        """Deserialize an image from an input archive.

        Parameters
        ----------
        archive
            Archive to read from.
        bbox
            Bounding box of a subimage to read instead.
        """
        image = self.image.deserialize(archive, bbox=bbox)
        variance = self.variance.deserialize(archive, bbox=bbox)
        return ExtendedPsfImage(
            image,
            variance=variance,
            info=self.info,
            fit=self.fit,
        )._finish_deserialize(self)
