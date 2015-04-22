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

    def __init__(self, measConfig, pluginRegistry, inputSchema, outputType, *args, **kwargs):
        """!Initialize TransformTask.

        @param[in] measConfig      Configuration for the measurement task which
                                   produced the measurments being transformed.
        @param[in] pluginRegistry  A PluginRegistry which maps plugin names to measurement algorithms.
        @param[in] inputSchema     The schema of the input catalog.
        @param[in] outputType      The butler dataset type of the output catalog.
        @param[in] *args           Passed through to pipeBase.Task.__init__()
        @param[in] *kwargs         Passed through to pipeBase.Task.__init__()
        """
        pipeBase.Task.__init__(self, *args, **kwargs)

        # This task can be used to generate multiple different output dataset types. We
        # need to be able to specify the output type together with its schema.
        self.outputType = outputType

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
        return {self.outputType: transformedSrc}

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


class RunTransformConfig(pexConfig.Config):
    """!Configuration for RunTransformTaskBase derivatives."""
    transform = pexConfig.ConfigurableField(
        doc="Subtask which performs transformations",
        target=TransformTask
    )
    inputConfigType = pexConfig.Field(
        dtype=str,
        doc="Dataset type of measurement operation configuration",
    )


class RunTransformTaskBase(pipeBase.CmdLineTask):
    """!Basic interface for TransformTask.

    Provide the skeleton of command-line task which can be used to run TransformTask.

    - Loads a plugin registry based on configuration;
    - Loads configuration for the measurement task which was applied from a repository;
    - Loads the SourceCatalog input schema from a repository;
    - For each input dataRef, reads the SourceCatalog, WCS and calibration from the
      repository and executes TransformTask.

    This is not a fully-fledged command line task: it requires specialization to a particular
    source type by defining the variables indicated below.
    """
    RunnerClass = pipeBase.ButlerInitializedTaskRunner
    ConfigClass = RunTransformConfig

    # Subclasses should provide appropriate definitions for the attributes named below.
    # Properties can be used if apprpriate.
    #
    # Standard CmdLineTask attributes:
    _DefaultName = None

    # Boolean; True if the measurement operation was forced, otherwise False.
    wasForced = None

    # Butler dataset type of the source type to be transformed ("src", "forced_src", etc):
    sourceType = None

    # Butler dataset type of the calibration exposure to use when transforming ("calexp", etc):
    calexpType = None

    @property
    def inputSchemaType(self):
        """
        The Butler dataset type for the schema of the input source catalog.

        By default, we append `_schema` to the input source type. Subclasses may customise
        if required.
        """
        return self.sourceType + "_schema"

    @property
    def outputType(self):
        """
        The Butler dataset type for the schema of the output catalog.

        By default, we prepend `transformed_` to the input source type. Subclasses may
        customise if required.
        """
        return 'transformed_' + self.sourceType

    @property
    def measurementConfig(self):
        """
        The configuration of the measurement operation used to generate the input catalog.

        By default we look for `measurement` under the root configuration of the
        generating task. Subclasses may customise this (e.g. to `calibration.measurement`)
        if required.
        """
        return self.butler.get(self.config.inputConfigType).measurement.value

    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, config=kwargs['config'], log=kwargs['log'])
        if self.wasForced:
            pluginRegistry = measBase.forcedMeasurement.ForcedPlugin.registry
        else:
            pluginRegistry = measBase.sfm.SingleFramePlugin.registry

        self.butler = kwargs['butler']

        self.makeSubtask('transform', pluginRegistry=pluginRegistry,
                         measConfig=self.measurementConfig,
                         inputSchema=self.butler.get(self.inputSchemaType).schema,
                         outputType=self.outputType)

    @pipeBase.timeMethod
    def run(self, dataRef):
        """!Transform the source catalog referred to by dataRef.

        The result is both returned and written as dataset type "transformed" + the input
        source type to the provided dataRef.

        @param[in] dataRef  Data reference for source catalog & calibrated exposure.

        @returns A BaseCatalog containing the transformed measurements.
        """
        inputCat = dataRef.get(self.sourceType)
        wcs = dataRef.get(self.calexpType).getWcs()
        calib = dataRef.get(self.calexpType).getCalib()
        outputCat = self.transform.run(inputCat, wcs, calib)
        dataRef.put(outputCat, self.outputType)
        return outputCat


class SrcTransformTask(RunTransformTaskBase):
    """
    Specialization of RunTransformTaskBase for use with 'src' measurements from e.g. processCcd.
    """
    _DefaultName = "transformSrcMeasurement"
    wasForced = False
    sourceType = 'src'
    calexpType = 'calexp'


class ForcedSrcTransformTask(RunTransformTaskBase):
    """
    Specialization of RunTransformTaskBase for use with 'forced_src' measurements from e.g. forcedPhotCcd.
    """
    _DefaultName = "transformForcedSrcMeasurement"
    wasForced = True
    sourceType = 'forced_src'
    calexpType = 'calexp'


class CoaddSrcTransformTask(RunTransformTaskBase):
    """
    Specialization of RunTransformTaskBase for use with measurements from processCoadd.
    """
    _DefaultName = "transformCoaddSrcMeasurement"
    wasForced = False

    @property
    def coaddName(self):
        return self.self.butler.get(self.config.inputConfigType).coaddName

    @property
    def sourceType(self):
        return self.coaddName + "_src"

    @property
    def calexpType(self):
        return self.coaddName + "_calexp"

    def _getConfigName(self):
        return "%s_transformCoaddSrcMeasurement_config" % (self.coaddName,)

    def _getMetaDataName(self):
        return "%s_transformCoaddSrcMeasurement_metadata" % (self.coaddName,)
