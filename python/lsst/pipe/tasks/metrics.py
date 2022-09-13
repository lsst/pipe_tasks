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
    "NumberDeblendChildSourcesMetricTask", "NumberDeblendChildSourcesMetricConfig",
]


import numpy as np
import astropy.units as u

from lsst.pipe.base import Struct, connectionTypes
from lsst.verify import Measurement
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
        sources : `lsst.afw.table.SourceCatalog`
            A science source catalog, which may be empty.

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
        if "deblend_nChild" not in sources.schema:
            self.log.info("Nothing to do: no deblending performed.")
            meas = None
        else:
            try:
                deblended = ((sources["parent"] == 0)           # top-level source
                             & (sources["deblend_nChild"] > 0)  # deblended
                             )
                deblended = _filterSkySources(sources, deblended)
            except LookupError as e:
                # Probably "parent"; all other columns already checked
                raise MetricComputationError("Invalid input catalog") from e
            else:
                nDeblended = np.count_nonzero(deblended)
                meas = Measurement(self.config.metricName, nDeblended * u.dimensionless_unscaled)

        return Struct(measurement=meas)


class NumberDeblendChildSourcesMetricConnections(
        MetricConnections,
        defaultTemplates={"package": "pipe_tasks",
                          "metric": "numDeblendChildSciSources"},
        dimensions={"instrument", "visit", "detector"},
):
    sources = connectionTypes.Input(
        doc="The catalog of science sources.",
        name="src",
        storageClass="SourceCatalog",
        dimensions={"instrument", "visit", "detector"},
    )


class NumberDeblendChildSourcesMetricConfig(
        MetricConfig,
        pipelineConnections=NumberDeblendChildSourcesMetricConnections):
    pass


class NumberDeblendChildSourcesMetricTask(MetricTask):
    """Task that computes the number of science sources created
    through deblending.

    This task only counts final deblending products; i.e., if deblending was
    run more than once or with multiple iterations, only the final set of
    deblended sources are counted, and not any intermediate ones.
    If sky source information is present, sky sources are excluded.

    Notes
    -----
    The task excludes any non-sky sources in the catalog, but it does
    not require that the catalog include a ``sky_sources`` column.
    """
    _DefaultName = "numDeblendChildSciSources"
    ConfigClass = NumberDeblendChildSourcesMetricConfig

    def run(self, sources):
        """Count the number of science sources created by deblending.

        Parameters
        ----------
        sources : `lsst.afw.table.SourceCatalog`
            A science source catalog, which may be empty.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            A `~lsst.pipe.base.Struct` containing the following component:

            ``measurement``
                the total number of science sources from deblending
                (`lsst.verify.Measurement`). If no deblending information is
                available in ``sources``, this is `None`.

        Raises
        ------
        MetricComputationError
            Raised if ``sources`` is missing mandatory keys for
            source catalogs.
        """
        # Use deblend_parentNChild rather than detect_fromBlend because the
        # latter need not be defined in post-deblending catalogs.
        if "deblend_parentNChild" not in sources.schema or "deblend_nChild" not in sources.schema:
            self.log.info("Nothing to do: no deblending performed.")
            meas = None
        else:
            try:
                children = ((sources["deblend_parentNChild"] > 1)  # deblend child
                            & (sources["deblend_nChild"] == 0)     # not deblended
                            )
                children = _filterSkySources(sources, children)
            except LookupError as e:
                # Probably "parent"; all other columns already checked
                raise MetricComputationError("Invalid input catalog") from e
            else:
                nChildren = np.count_nonzero(children)
                meas = Measurement(self.config.metricName, nChildren * u.dimensionless_unscaled)

        return Struct(measurement=meas)


def _filterSkySources(catalog, selection):
    """Filter out any sky sources from a vector of selected sources.

    If no sky source information is available, all sources are assumed to
    be non-sky.

    Parameters
    ----------
    catalog : `lsst.afw.table.SourceCatalog`
        The catalog to filter.
    selection : `numpy.ndarray` [`bool`], (N,)
        A vector of existing source selections, of the same length as
        ``catalog``, where selected sources are marked `True`.

    Returns
    -------
    filtered : `numpy.ndarray` [`bool`], (N,)
        A version of ``selection`` with any sky sources filtered out
        (set to `False`). May be the same vector as ``selection`` if
        no changes were made.
    """
    if "sky_source" in catalog.schema:
        # E712 is not applicable, because afw.table.SourceRecord.ColumnView
        # is not a bool.
        return selection & (catalog["sky_source"] == False)  # noqa: E712
    else:
        return selection
