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
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.utils.timer import timeMethod


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
        default=('id', 'coord_ra', 'coord_dec')
    )

## \addtogroup LSST_task_documentation
## \{
## \page TransformTask
## \ref TransformTask_ "TransformTask"
## \copybrief TransformTask
## \}


class TransformTask(pipeBase.Task):
    r"""!
    \anchor TransformTask_

    \brief Transform a SourceCatalog containing raw measurements to calibrated form.

    \section pipe_tasks_transform_Contents Contents

     - \ref pipe_tasks_transform_purpose
     - \ref pipe_tasks_transform_initialize
     - \ref pipe_tasks_transform_invoke

    \section pipe_tasks_transform_purpose Description

    Given a set of measurement algorithms with their associated configuration,
    the table of source measurements they have produced, and information about
    an associated WCS and calibration, transform the raw measurement output to
    a calibrated form.

    Transformations are defined on a per-measurement-plugin basis. In
    addition, a configurable set of fields may be simply copied from the input
    to the output catalog.

    This task operates on an input SourceCatalog and returns a BaseCatalog
    containing the transformed results. It requires that the caller supply
    information on the configuration of the measurement task which produced
    the input data as well as the world coordinate system and calibration
    under which the transformation will take place. It provides no
    functionality for reading or writing data from a Butler: rather,
    per-dataset-type command line tasks are provided to obtain the appropriate
    information from a Butler (or elsewhere) and then delegate to this task.

    \section pipe_tasks_transform_initialize Task initialization

    \copydoc \_\_init\_\_

    \section pipe_tasks_transform_invoke Task invocation

    \copydoc run
    """
    ConfigClass = TransformConfig
    _DefaultName = "transform"

    def __init__(self, measConfig, inputSchema, outputDataset, *args, **kwargs):
        """!Initialize TransformTask.

        @param[in] measConfig      Configuration for the measurement task which
                                   produced the measurments being transformed.
        @param[in] inputSchema     The schema of the input catalog.
        @param[in] outputDataset   The butler dataset type of the output catalog.
        @param[in] *args           Passed through to pipeBase.Task.__init__()
        @param[in] *kwargs         Passed through to pipeBase.Task.__init__()
        """
        pipeBase.Task.__init__(self, *args, **kwargs)

        # This task can be used to generate multiple different output dataset types. We
        # need to be able to specify the output type together with its schema.
        self.outputDataset = outputDataset

        # Define a mapper and add the basic fields to be copied.
        self.mapper = afwTable.SchemaMapper(inputSchema)
        for field in self.config.copyFields:
            self.mapper.addMapping(inputSchema.find(field).key)

        # Build a list of all transforms that will be applied to the input. We
        # will iterate over this in run().
        self.transforms = []
        for name in measConfig.plugins.names:
            config = measConfig.plugins.get(name)
            transformClass = measConfig.plugins.registry.get(name).PluginClass.getTransformClass()
            self.transforms.append(transformClass(config, name, self.mapper))

    def getSchemaCatalogs(self):
        """!Return a dict containing an empty catalog representative of this task's output."""
        transformedSrc = afwTable.BaseCatalog(self.mapper.getOutputSchema())
        return {self.outputDataset: transformedSrc}

    @timeMethod
    def run(self, inputCat, wcs, photoCalib):
        """!Transform raw source measurements to calibrated quantities.

        @param[in] inputCat  SourceCatalog of sources to transform.
        @param[in] wcs       The world coordinate system under which transformations will take place.
        @param[in] photoCalib     The calibration under which transformations will take place.

        @return A BaseCatalog containing the transformed measurements.
        """
        outputCat = afwTable.BaseCatalog(self.mapper.getOutputSchema())
        outputCat.extend(inputCat, mapper=self.mapper)

        # Transforms may use a ColumnView on the input and output catalogs,
        # which requires that the data be contiguous in memory.
        inputCat = makeContiguous(inputCat)
        outputCat = makeContiguous(outputCat)

        for transform in self.transforms:
            transform(inputCat, outputCat, wcs, photoCalib)
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
    r"""!
    \anchor RunTransformTaskBase_

    \brief Command line interface for TransformTask.

    \section pipe_tasks_transform_Contents Contents

     - \ref pipe_tasks_runtransform_purpose
     - \ref pipe_tasks_runtransform_invoke

    \section pipe_tasks_runtransform_purpose Description

    Provides a command-line task which can be used to run TransformTask.

    - Loads a plugin registry based on configuration;
    - Loads configuration for the measurement task which was applied from a repository;
    - Loads the SourceCatalog input schema from a repository;
    - For each input dataRef, reads the SourceCatalog, WCS and calibration from the
      repository and executes TransformTask.

    This is not a fully-fledged command line task: it requires specialization to a particular
    source type by defining the variables indicated below.

    \section pipe_tasks_runtransform_invoke Task invocation

    \copydoc run
    """
    RunnerClass = pipeBase.ButlerInitializedTaskRunner
    ConfigClass = RunTransformConfig

    # Subclasses should provide definitions for the attributes named below.
    # Properties can be used if appropriate.
    #
    # Standard CmdLineTask attributes:
    _DefaultName = None

    # Butler dataset type of the source type to be transformed ("src", "forced_src", etc):
    sourceType = None

    # Butler dataset type of the calibration exposure to use when transforming ("calexp", etc):
    calexpType = None

    @property
    def inputSchemaType(self):
        """!
        The Butler dataset type for the schema of the input source catalog.

        By default, we append `_schema` to the input source type. Subclasses may customize
        if required.
        """
        return self.sourceType + "_schema"

    @property
    def outputDataset(self):
        """!
        The Butler dataset type for the schema of the output catalog.

        By default, we prepend `transformed_` to the input source type. Subclasses may
        customize if required.
        """
        return 'transformed_' + self.sourceType

    @property
    def measurementConfig(self):
        """!
        The configuration of the measurement operation used to generate the input catalog.

        By default we look for `measurement` under the root configuration of the
        generating task. Subclasses may customize this (e.g. to `calibrate.measurement`)
        if required.
        """
        return self.butler.get(self.config.inputConfigType).measurement.value

    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, config=kwargs['config'], log=kwargs['log'])
        self.butler = kwargs['butler']
        self.makeSubtask('transform', measConfig=self.measurementConfig,
                         inputSchema=self.butler.get(self.inputSchemaType).schema,
                         outputDataset=self.outputDataset)

    @timeMethod
    def runDataRef(self, dataRef):
        """!Transform the source catalog referred to by dataRef.

        The result is both returned and written as dataset type "transformed_" + the input
        source dataset type to the provided dataRef.

        @param[in] dataRef  Data reference for source catalog & calibrated exposure.

        @returns A BaseCatalog containing the transformed measurements.
        """
        inputCat = dataRef.get(self.sourceType)
        wcs = dataRef.get(self.calexpType).getWcs()
        photoCalib = dataRef.get(self.calexpType).getPhotoCalib()
        outputCat = self.transform.run(inputCat, wcs, photoCalib)
        dataRef.put(outputCat, self.outputDataset)
        return outputCat


## \addtogroup LSST_task_documentation
## \{
## \page SrcTransformTask
## \ref SrcTransformTask_ "SrcTransformTask"
## \copybrief SrcTransformTask
## \}

class SrcTransformTask(RunTransformTaskBase):
    """!
    \anchor SrcTransformTask_

    \brief Transform ``src`` measuremenents to calibrated form.

    This is a specialization of \ref RunTransformTaskBase_ "RunTransformTaskBase" which
    operates on ``src`` measurements. Refer to the parent documentation for details.
    """
    _DefaultName = "transformSrcMeasurement"
    sourceType = 'src'
    calexpType = 'calexp'

    @property
    def measurementConfig(self):
        return self.butler.get(self.config.inputConfigType).calibrate.measurement.value


## \addtogroup LSST_task_documentation
## \{
## \page ForcedSrcTransformTask
## \ref ForcedSrcTransformTask_ "ForcedSrcTransformTask"
## \copybrief ForcedSrcTransformTask
## \}

class ForcedSrcTransformTask(RunTransformTaskBase):
    """!
    \anchor ForcedSrcTransformTask_

    \brief Transform ``forced_src`` measuremenents to calibrated form.

    This is a specialization of \ref RunTransformTaskBase_ "RunTransformTaskBase" which
    operates on ``forced_src`` measurements. Refer to the parent documentation for details.
    """
    _DefaultName = "transformForcedSrcMeasurement"
    sourceType = 'forced_src'
    calexpType = 'calexp'


## \addtogroup LSST_task_documentation
## \{
## \page CoaddSrcTransformTask
## \ref CoaddSrcTransformTask_ "CoaddSrcTransformTask"
## \copybrief CoaddSrcTransformTask
## \}

class CoaddSrcTransformTask(RunTransformTaskBase):
    """!
    \anchor CoaddSrcTransformTask_

    \brief Transform measuremenents made on coadds to calibrated form.

    This is a specialization of \ref RunTransformTaskBase_ "RunTransformTaskBase"  which
    operates on measurements made on coadds. Refer to the parent documentation for details.
    """
    _DefaultName = "transformCoaddSrcMeasurement"

    @property
    def coaddName(self):
        return self.butler.get(self.config.inputConfigType).coaddName

    @property
    def sourceType(self):
        return self.coaddName + "Coadd_meas"

    @property
    def calexpType(self):
        return self.coaddName + "Coadd_calexp"

    def _getConfigName(self):
        return "%s_transformCoaddSrcMeasurement_config" % (self.coaddName,)

    def _getMetaDataName(self):
        return "%s_transformCoaddSrcMeasurement_metadata" % (self.coaddName,)
