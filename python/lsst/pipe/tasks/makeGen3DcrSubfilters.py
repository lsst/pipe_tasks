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

from sqlalchemy.exc import IntegrityError


class MakeGen3DcrSubfiltersConfig(pexConfig.Config):
    """Config for MakeGen3DcrSubfiltersTask.
    """
    numSubfilters = pexConfig.Field(
        doc="The number of subfilters to be used for chromatic modeling.",
        dtype=int,
        default=3,
        optional=False
    )
    filterNames = pexConfig.ListField(
        doc="The filters to add chromatic subfilters to in the registry.",
        dtype=str,
        default=["g"],
        optional=False
    )


class MakeGen3DcrSubfiltersTask(pipeBase.Task):
    ConfigClass = MakeGen3DcrSubfiltersConfig
    _DefaultName = "makeGen3DcrSubfilters"

    """This is a task to construct the set of subfilters for chromatic modeling.

    Parameters
    ----------
    config : `MakeGen3DcrSubfiltersConfig` or None
        Instance of a configuration class specifying task options, a default
        config is created if value is None
    """

    def __init__(self, *, config=None, **kwargs):
        super().__init__(config=config, **kwargs)

    def run(self, butler):
        """Construct a set of subfilters for chromatic modeling.

        Parameters
        ----------
        butler : `lsst.daf.butler.Butler`
            Butler repository to add the subfilter definitions to.
        """
        with butler.registry.transaction():
            try:
                self.register(butler.registry)
            except IntegrityError as err:
                raise RuntimeError(f"Subfilters for at least one filter of {self.config.filterNames} "
                                   "are already defined.") from err

    def register(self, registry):
        """Add Subfilters to the given registry.

        Parameters
        ----------
        registry : `lsst.daf.butler.Registry`
            The registry to add to.
        """
        record = []
        for filterName in self.config.filterNames:
            self.log.info(f"Initializing filter {filterName} with "
                          f"{self.config.numSubfilters} subfilters")
            for sub in range(self.config.numSubfilters):
                record.append({
                              "band": filterName,
                              "subfilter": sub
                              })
        registry.insertDimensionData("subfilter", *record)
