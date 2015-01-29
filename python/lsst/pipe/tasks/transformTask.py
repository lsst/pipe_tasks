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
    copyFields = pexConfig.ListField(
        dtype = str,
        doc = "Fields to copy without tranformation",
        default = ('id', 'coord')
    )

## \addtogroup LSST_task_documentation
## \{
## \page pipeTasks_transformTask
## \ref TransformTask "TransformTask"
##      Transform raw measurements to calibrated quantities.
## \}

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
    def run(self, sourceCat, wcs, calib):
        """!Transform raw source measurements to calibrated quantities.

        @param[in] sourceCat: SourceCatalog of sources to transform.
        @param[in] wcs: The world coordinate system under which
                        transformations will take place.
        @param[in] calib: The calibration under which transformations will
                          take place.

        @return A BaseCatalog containing the transformed measurements.
        """
        # Define a mapper which copies basic values across.
        mapper = afwTable.SchemaMapper(sourceCat.schema)
        for field in self.config.copyFields:
            mapper.addMapping(sourceCat.schema.find(field).key)

        transforms = [self.pluginRegistry.get(name).PluginClass.getTransformClass()(name, mapper, cfg, wcs, calib)
                      for name, cfg in self.measPlugins]

        # Iterate over the input catalogue, mapping/transforming sources to
        # the new schema.
        newSources = afwTable.BaseCatalog(mapper.getOutputSchema())
        newSources.reserve(len(sourceCat))
        for oldSource in sourceCat:
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
        sourceCat = dataRef.get('src')
        wcs = dataRef.get('calexp').getWcs()
        calib = dataRef.get('calexp').getCalib()

        return self.transform.run(sourceCat, wcs, calib)

    def _getConfigName(self):
        return None

    def _getMetadataName(self):
        return None
