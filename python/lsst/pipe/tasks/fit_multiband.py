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

__all__ = ["CatalogExposure", "CatalogExposureConfig", ]

from functools import cached_property
from pydantic import Field, ConfigDict
from pydantic.dataclasses import dataclass

import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.daf.butler as dafButler


CatalogExposureConfig = ConfigDict(arbitrary_types_allowed=True)


@dataclass(frozen=True, kw_only=True, config=CatalogExposureConfig)
class CatalogExposure:
    """A class to store a catalog, exposure, and metadata for a given dataId.

    The intent is to store an exposure and an associated measurement catalog.
    Users may omit one but not both (e.g. if the intent is just to attach
    a dataId and metadata to a catalog or exposure).
    """
    @cached_property
    def band(self) -> str:
        return self.dataId['band']

    @cached_property
    def calib(self) -> afwImage.PhotoCalib | None:
        return None if self.exposure is None else self.exposure.getPhotoCalib()

    dataId: dafButler.DataCoordinate | dict = Field(
        title="A DataCoordinate or dict containing a 'band' item")
    catalog: afwTable.SourceCatalog | None = Field(None, title="The measurement catalog, if any")
    exposure: afwImage.Exposure | None = Field(None, title="The exposure, if any")
    id_tract_patch: int = Field(0, title="A unique ID for this tract-patch pair")
    metadata: dict = Field(default_factory=dict, title="Arbitrary metadata")

    def __post_init__(self):
        if self.catalog is None and self.exposure is None:
            raise ValueError("Must specify at least one of catalog/exposure")
        if 'band' not in self.dataId:
            raise ValueError(f"dataId={self.dataId} must have a band")
