#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2014 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.table as afwTable
from lsst.meas.base import SingleFramePlugin
from lsst.meas.base import generateAlgorithmName
from lsst.pex.exceptions import NotFoundError


class CalibrateConfig(pexConfig.Config):
    """!Configuration for CalibrateTask"""
    pass


class CalibrateTask(pipeBase.Task):
    """!Calibration of measured quantities

    This task will receive a table of sources which have been measured with a
    meas_base derived measurement task. It should:

    * Look up measurement plugins which have been used to produce those
      measurements;
    * Check for calibration plugins which convert those measurements to some
      calibrated form;
    * Save the output to a new table, which is returned.

    """
    ConfigClass = CalibrateConfig
    _DefaultName = "calibrate"

    def __init__(self, *args, **kwargs):
        """!Construct a CalibrateTask
        """
        pipeBase.Task.__init__(self, *args, **kwargs)

    @pipeBase.timeMethod
    def run(self, sources, plugins):
        """!Calibrate a source table
        @param[in] sources: The source table for calibration.
        @param[in] plugins: The PluginMap used for measurement.
        @return a pipeBase Struct containing:
        - sources: The calibrated source table
        """
        mapper = afwTable.SchemaMapper(sources.schema)
        mapper.addMinimalSchema(afwTable.SourceTable.makeMinimalSchema())

        # Special case field added by detection step
        if "flags_negative" in sources.schema.getNames():
            mapper.addMapping(sources.schema.find('flags_negative').key)

        # mapFunctions is a list of functions which are applied to a
        # SourceRecord to map it into the output table.
        mapFunctions = []
        for name, plugin in plugins.iteritems():
            mapFunctions.extend(plugin.genCalibrate(sources.schema, mapper))

        # Should we also include the possibility of plugins which are not tied
        # to a specific measurment plugin?
        # Even: extend the measurement framework so that we can have plugins
        # which do not perform measurements but simply add calibration steps?
        # If so, what inputs do they need? Is the source table schema
        # adequate, or do we need to include e.g. the input images or further
        # configuration information?

        # Iterate over the input catalogue, mapping/transforming sources to
        # the new schema.
        newSources = afwTable.SourceCatalog(mapper.getOutputSchema())
        newSources.reserve(len(sources))
        for oldSource in sources:
            newSource = newSources.addNew()
            newSource.assign(oldSource, mapper)
            for fn in mapFunctions:
                fn(oldSource, newSource)

        # WARNING! The slots can be defined on the sources table even if the
        # source columns can't exist; they then _can't_ be defined on the
        # newSources table!
        # Should use a central definition of the slot names rather than
        # hardcoding here.
        for slot in ("PsfFlux", "ApFlux", "InstFlux", "ModelFlux", "Centroid", "Shape"):
            defn = getattr(sources.table, 'get' + slot[0].upper() + slot[1:] + "Definition")
            try:
                getattr(newSources.table, 'define' + slot[0].upper() + slot[1:])(defn())
            except NotFoundError:
                print "Fields not available for slot %s" % (slot,)

        return pipeBase.Struct(sources=newSources)
