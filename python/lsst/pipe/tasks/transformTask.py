#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2014-2015 LSST Corporation.
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
import lsst.afw.table as afwTable
import lsst.meas.base as measBase
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

class TransformConfig(pexConfig.Config):
    pass

class TransformTask(pipeBase.Task):
    ConfigClass = TransformConfig
    _DefaultName = "transform"

    def __init__(self, *args, **kwargs):
        # Need to extract these kwargs, or Task.__init__() chokes.
        measConfig = kwargs.pop('measConfig')
        self.pluginRegistry = kwargs.pop('pluginRegistry')
        pipeBase.Task.__init__(self, *args, **kwargs)

        # A list of measurement plugins & configurations used.
        self.measPlugins = [(name, measConfig.value.plugins.get(name))
                            for name in measConfig.value.plugins.names]

    @pipeBase.timeMethod
    def run(self, sourceList, wcs, calib):
        """!Generate source table transformations

        @param[in] mapper:  A SchemaMapper containing the input and output schemas.
        @param[in] plugins: A list of (name, configuration) pairs describing
                            the measurement plugins to be transformed.
        @return a pipeBase Struct containing:
        - mapper:    The updated SchemaMapper.
        - tranforms: An iterable of callables which perform record
                     transformations.
        """
        # Define a mapper which copies basic values across.
        # (It seems like it would be nice to do this in __init__ but I think
        # we can't, since we need the input schema from the source list.)
        mapper = afwTable.SchemaMapper(sourceList.schema)
        mapper.addMapping(sourceList.schema.find('id').key)
        mapper.addMapping(sourceList.schema.find('coord').key)

        transforms = [self.pluginRegistry.get(name).PluginClass.getTransformClass()(name, mapper, cfg, wcs, calib)
                      for name, cfg in self.measPlugins]

        # Iterate over the input catalogue, mapping/transforming sources to
        # the new schema.
        newSources = afwTable.BaseCatalog(mapper.getOutputSchema())
        newSources.reserve(len(sourceList))
        for oldSource in sourceList:
            newSource = newSources.addNew()
            newSource.assign(oldSource, mapper)
            for transform in transforms:
                transform(oldSource, newSource)

        return newSources


class TransformInterfaceConfig(pexConfig.Config):
    """!Configuration for TransformInterfaceTask
    """
    transform = pexConfig.ConfigurableField(
        doc="Subtask which performs transformations",
        target=TransformTask
    )


class TransformInterfaceTask(pipeBase.CmdLineTask):
    ConfigClass = TransformInterfaceConfig
    RunnerClass = pipeBase.ButlerInitializedTaskRunner
    _DefaultName = "transformInterface"

    @classmethod
    def _makeArgumentParser(cls):
        # Reconfigures the default argument parser to use dataset type fpC (an
        # SDSS "corrected frame") so that we can use it with the DM stack demo
        # data.
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", datasetType="fpC",
                               help="data ID, e.g. --id run=1 camcol=2 field=345")
        # We probably need additional arguments in here to handle setting
        # alternative WCS and calibration information that isn't provided in
        # the base dataRef.
        return parser

    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, config=kwargs['config'], log=kwargs['log'])
        self.makeSubtask('transform', measConfig=kwargs['butler'].get('processCcd_config').measurement,
                         pluginRegistry=measBase.sfm.SingleFramePlugin.registry)

    @pipeBase.timeMethod
    def run(self, dataRef):
        """!Transform input table.
        """
        # Note: the source table already has the algorithm metadata attached
        # to it; there should be no need to do anything further.
        sourceList = dataRef.get('src')
        wcs = dataRef.get('calexp').getWcs()
        calib = dataRef.get('calexp').getCalib()

        return self.transform.run(sourceList, wcs, calib)

    def _getConfigName(self):
        return None

    def _getMetadataName(self):
        return None
