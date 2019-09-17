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
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.daf.butler import DatasetType
from lsst.skymap import skyMapRegistry

from sqlalchemy.exc import IntegrityError


class MakeGen3SkyMapConfig(pexConfig.Config):
    """Config for MakeGen3SkyMapTask
    """
    datasetTypeName = pexConfig.Field(
        doc="Name assigned to created skymap in butler registry",
        dtype=str,
        default="deepCoadd_skyMap",
    )
    name = pexConfig.Field(
        doc="Name assigned to created skymap in butler registry",
        dtype=str,
        default=None,
        optional=True
    )
    skyMap = skyMapRegistry.makeField(
        doc="type of skyMap",
        default="dodeca",
    )

    def validate(self):
        if self.name is None:
            raise ValueError("The name field must be set to the name of the specific "
                             "skymap to use when writing to the butler")


class MakeGen3SkyMapTask(pipeBase.Task):
    ConfigClass = MakeGen3SkyMapConfig
    _DefaultName = "makeGen3SkyMap"

    """This is a task to construct and optionally save a SkyMap into a gen3
    butler repository.

    Parameters
    ----------
    config : `MakeGen3SkyMapConfig` or None
        Instance of a configuration class specifying task options, a default
        config is created if value is None
    """

    def __init__(self, *, config=None, **kwargs):
        super().__init__(config=config, **kwargs)

    def run(self, butler):
        """Construct and optionally save a SkyMap into a gen3 repository
        Parameters
        ----------
        butler : `lsst.daf.butler.Butler`
            Butler repository to which the new skymap will be written
        """
        skyMap = self.config.skyMap.apply()
        skyMap.logSkyMapInfo(self.log)
        skyMapHash = skyMap.getSha1()
        self.log.info(f"Inserting SkyMap {self.config.name} with hash={skyMapHash}")
        with butler.registry.transaction():
            try:
                skyMap.register(self.config.name, butler.registry)
            except IntegrityError as err:
                raise RuntimeError("A skymap with the same name or hash already exists.") from err
            butler.registry.registerDatasetType(DatasetType(name=self.config.datasetTypeName,
                                                            dimensions=["skymap"],
                                                            storageClass="SkyMap",
                                                            universe=butler.registry.dimensions))
            butler.put(skyMap, self.config.datasetTypeName, {"skymap": self.config.name})
        return pipeBase.Struct(
            skyMap=skyMap
        )
