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

__all__ = [
    "NumberDeblendedSourcesMetricTask", "NumberDeblendedSourcesMetricConfig",
]


import numpy as np
import astropy.units as u

from lsst.pipe.base import Struct, connectionTypes
from lsst.verify import Measurement
from lsst.verify.gen2tasks import register
from lsst.verify.tasks import MetricTask, MetricConfig, MetricConnections, MetricComputationError


class NumberDeblendedSourcesMetricConnections(
        MetricConnections,
        defaultTemplates={"package": "pipe_tasks",
                          "metric": "numDeblendedSciSources"},
        dimensions={"instrument", "visit", "detector"},
):
    sources = connectionTypes.Input(
        doc="The catalog of science sources.",
        name="src",
        storageClass="SourceCatalog",
        dimensions={"instrument", "visit", "detector"},
    )


class NumberDeblendedSourcesMetricConfig(
        MetricConfig,
        pipelineConnections=NumberDeblendedSourcesMetricConnections):
    pass


@register("numDeblendedSciSources")
class NumberDeblendedSourcesMetricTask(MetricTask):
    """Task that computes the number of science sources that have
    been deblended.

    This task only counts sources that existed prior to any deblending;
    i.e., if deblending was run more than once or with multiple iterations,
    only the "top-level" deblended sources are counted, and not any
    intermediate ones. If sky source information is present, sky sources
    are excluded.

    Notes
    -----
    The task excludes any non-sky sources in the catalog, but it does
    not require that the catalog include a ``sky_sources`` column.
    """
    _DefaultName = "numDeblendedSciSources"
    ConfigClass = NumberDeblendedSourcesMetricConfig

    def run(self, sources):
        """Count the number of deblended science sources.

        Parameters
        ----------
        sources : `lsst.afw.table.SourceCatalog` or `None`
            A science source catalog, which may be empty or `None`.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            A `~lsst.pipe.base.Struct` containing the following component:

            ``measurement``
                the total number of deblended science sources
                (`lsst.verify.Measurement`). If no deblending information is
                available in ``sources``, this is `None`.

        Raises
        ------
        MetricComputationError
            Raised if ``sources`` is missing mandatory keys for
            source catalogs.
        """
        if sources is None:
            self.log.info("Nothing to do: no catalogs found.")
            meas = None
        elif "deblend_nChild" not in sources.schema:
            self.log.info("Nothing to do: no deblending performed.")
            meas = None
        else:
            try:
                deblended = ((sources["parent"] == 0)           # top-level source
                             & (sources["deblend_nChild"] > 0)  # deblended
                             )
                if "sky_source" in sources.schema:
                    # E712 is not applicable, because
                    # afw.table.SourceRecord.ColumnView is not a bool.
                    deblended = deblended & (sources["sky_source"] == False)  # noqa: E712
            except LookupError as e:
                # Probably "parent"; all other columns already checked
                raise MetricComputationError("Invalid input catalog") from e
            else:
                nDeblended = np.count_nonzero(deblended)
                meas = Measurement(self.config.metricName, nDeblended * u.dimensionless_unscaled)

        return Struct(measurement=meas)
