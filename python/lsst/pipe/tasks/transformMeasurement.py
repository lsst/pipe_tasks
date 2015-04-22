#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2008-2015 AURA/LSST.
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
"""
Tasks for transforming raw measurement outputs to calibrated quantities.
"""
import lsst.afw.table as afwTable
import lsst.meas.base as measBase
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

def makeContiguous(catalog):
    """!Return a version of the input catalog which is contiguous in memory."""
    if not catalog.isContiguous():
        return catalog.copy(deep=True)
    else:
        return catalog


class TransformConfig(pexConfig.Config):
    """!Configuration for TransformTask."""
    copyFields = pexConfig.ListField(
        dtype=str,
        doc="Fields to copy from input to output catalog without transformation",
        default=('id', 'coord')
    )

## \addtogroup LSST_task_documentation
## \{
## \page pipeTasks_TransformTask
## \ref TransformTask "TransformTask"
##      Task for transforming raw measurement outputs to calibrated quantities.
## \}

class TransformTask(pipeBase.Task):
    """!Transform a SourceCatalog containing raw measurements to calibrated form.

    Given a set of measurement algorithms with their associated configuration,
    the table of source measurements they have produced, and information about
    an associated WCS and calibration, transform the raw measurement output to
    a calibrated form.

    Transformations are defined on a per-measurement-plugin basis. In
    addition, a configurable set of fields may be simply copied from the input
    to the output catalog.
    """
    ConfigClass = TransformConfig
    _DefaultName = "transform"

    def __init__(self, measConfig, pluginRegistry, inputSchema, *args, **kwargs):
        """!Initialize TransformTask.

        @param[in] measConfig      Configuration for the measurement task which
                                   produced the measurments being transformed.
        @param[in] pluginRegistry  A PluginRegistry which maps plugin names to measurement algorithms.
        @param[in] inputSchema     The schema of the input catalog.
        @param[in] *args           Passed through to pipeBase.Task.__init__()
        @param[in] *kwargs         Passed through to pipeBase.Task.__init__()
        """
        pipeBase.Task.__init__(self, *args, **kwargs)

        # Define a mapper and add the basic fields to be copied.
        self.mapper = afwTable.SchemaMapper(inputSchema)
        for field in self.config.copyFields:
            self.mapper.addMapping(inputSchema.find(field).key)

        # Build a list of all transforms that will be applied to the input. We
        # will iterate over this in run().

        self.transforms = []
        for name in measConfig.plugins.names:
            config = measConfig.plugins.get(name)
            transformClass = pluginRegistry.get(name).PluginClass.getTransformClass()
            self.transforms.append(transformClass(config, name, self.mapper))

    def getSchemaCatalogs(self):
        """!Return a dict containing an empty catalog representative of this task's output."""
        transformedSrc = afwTable.BaseCatalog(self.mapper.getOutputSchema())
        return {'transformedSrc': transformedSrc}

    def run(self, inputCat, wcs, calib):
        """!Transform raw source measurements to calibrated quantities.

        @param[in] inputCat  SourceCatalog of sources to transform.
        @param[in] wcs       The world coordinate system under which transformations will take place.
        @param[in] calib     The calibration under which transformations will take place.

        @return A BaseCatalog containing the transformed measurements.
        """
        outputCat = afwTable.BaseCatalog(self.mapper.getOutputSchema())
        outputCat.extend(inputCat, mapper=self.mapper)

        # Transforms may use a ColumnView on the input and output catalogs,
        # which requires that the data be contiguous in memory.
        inputCat = makeContiguous(inputCat)
        outputCat = makeContiguous(outputCat)

        for transform in self.transforms:
            transform(inputCat, outputCat, wcs, calib)
        return outputCat


class RunTransformConfigBase(object):
    """!Configuration for RunTransformTaskBase derivatives."""
    transform = pexConfig.ConfigurableField(
        doc="Subtask which performs transformations",
        target=TransformTask
    )


class RunTransformTaskBase(object):
    """!Basic interface for TransformTask.

    Provide the skeleton of command-line task which can be used to run TransformTask.

    - Loads a plugin registry based on configuration;
    - Loads configuration for the measurement task which was applied from a repository;
    - Loads the SourceCatalog input schema from a repository;
    - For each input dataRef, reads the SourceCatalog, WCS and calibration from the
      repository and executes TransformTask.

    This is not a fully-fledged command line task: it requires specialization to a particular
    source type.  That can be conveniently provided by makeTransformCmdLineTask.
    """
    RunnerClass = pipeBase.ButlerInitializedTaskRunner

    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, config=kwargs['config'], log=kwargs['log'])
        if self.config.wasForced:
            pluginRegistry = measBase.forcedMeasurement.ForcedPlugin.registry
        else:
            pluginRegistry = measBase.sfm.SingleFramePlugin.registry
        self.makeSubtask('transform', pluginRegistry=pluginRegistry,
                         measConfig=kwargs['butler'].get(self.config.measConfig).measurement.value,
                         inputSchema=kwargs['butler'].get(self.config.sourceType + "_schema").schema)

    @pipeBase.timeMethod
    def run(self, dataRef):
        """!Transform the source catalog referred to by dataRef.

        The result is both returned and written as dataset type "transformed" + the input
        source type to the provided dataRef.

        @param[in] dataRef  Data reference for source catalog & calibrated exposure (calexp).

        @returns A BaseCatalog containing the transformed measurements.
        """
        inputCat = dataRef.get(self.config.sourceType)
        wcs = dataRef.get(self.config.calexpType).getWcs()
        calib = dataRef.get(self.config.calexpType).getCalib()
        outputCat = self.transform.run(inputCat, wcs, calib)
        dataRef.put(outputCat, "transformed" + self.config.sourceType.capitalize())
        return outputCat


def makeTransformCmdLineTask(sourceType, calexpType, measConfigType, outputType, wasForced):
    """!Generate and return a specialization of RunTransformTaskBase.

    RunTransformTaskBase provides a means of transforming arbitrary source types (src, forced_src,
    deepCoadd_src, ...). However, a specialization is required for each type we wish to transform. Here, we
    generate a specialization for the source type required.

    @param[in] sourceType      Name of source type to transform (e.g src, forced_src, etc).
    @param[in] calexpType      Name of exposure with calibration data (e.g. calexp, etc)
    @param[in] measConfigType  Name of measurement task configuration (e.g. processCCd_config, etc)
    @param[in] outputType      Name of output type to write (e.g. transformedSrc, etc)
    @param[in] wasForced       Boolean indicating if source measurements were forced.

    @returns A tuple of (Config, CmdLineTask) suitable for transforming sources of the given type.
    """
    sourceTypeField = pexConfig.Field(dtype=str, default=sourceType,
                                      doc="Dataset type of source to transform")
    calexpTypeField = pexConfig.Field(dtype=str, default=calexpType,
                                      doc="Dataset type of calibration exposure")
    measConfigField = pexConfig.Field(dtype=str, default="processCcd_config",
                                      doc="Dataset type of measurement operation configuration")
    outputTypeField = pexConfig.Field(dtype=str, default=outputType,
                                      doc="Dataset type of output sources")
    forcedField = pexConfig.Field(dtype=bool, default=wasForced, doc="Measurement operation was forced")

    configClass = type("%sTransformConfig" % sourceType.capitalize(),
                       (RunTransformConfigBase, pexConfig.Config),
                       {"sourceType": sourceTypeField, "calexpType": calexpTypeField,
                        "outputType": outputTypeField, "measConfig": measConfigField,
                        "wasForced": forcedField})
    taskClass = type("%sTransformTask" % (sourceType.capitalize(),),
                     (RunTransformTaskBase, pipeBase.CmdLineTask),
                     {"wasForced": wasForced, "ConfigClass": configClass,
                      "_DefaultName": "transform%sMeasurement" % (sourceType.capitalize(),)})
    return configClass, taskClass


# Explicitly instantiate command line transformation types for common source types. Camera-specific
# transformation tasks can be defined following this pattern in the appropriate obs_ package.
SrcTransformConfig, SrcTransformTask = makeTransformCmdLineTask(
                                                "src", "calexp", "processCcd_config", "transformedSrc", False)
ForcedSrcTransformConfig, ForcedSrcTransformTask = makeTransformCmdLineTask(
                                       "forced_src", "calexp", "forced_config", "transformedForced_src", True)
DeepCoaddSrcTransformConfig, DeepCoaddSrcTransformTask = makeTransformCmdLineTask(
           "deepCoadd_src", "deepCoadd_calexp", "deep_processCoadd_config", "transformedDeepCoadd_src", False)
DeepCoaddForcedSrcTransformConfig, DeepCoaddForcedSrcTransformTask = makeTransformCmdLineTask(
                                        "deepCoadd_forced_src", "deepCoadd_calexp", "deepCoadd_forced_config",
                                                                      "transformedDeepCoadd_Forced_src", True)
