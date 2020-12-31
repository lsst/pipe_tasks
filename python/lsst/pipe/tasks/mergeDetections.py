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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#

from .multiBandUtils import (CullPeaksConfig, MergeSourcesRunner, _makeMakeIdFactory, makeMergeArgumentParser,
                             getInputSchema, readCatalog)


import lsst.afw.detection as afwDetect
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable

from lsst.meas.algorithms import SkyObjectsTask
from lsst.skymap import BaseSkyMap
from lsst.pex.config import Config, Field, ListField, ConfigurableField, ConfigField
from lsst.pipe.base import (CmdLineTask, PipelineTask, PipelineTaskConfig, Struct,
                            PipelineTaskConnections)
import lsst.pipe.base.connectionTypes as cT
from lsst.pipe.tasks.coaddBase import getSkyInfo


class MergeDetectionsConnections(PipelineTaskConnections,
                                 dimensions=("tract", "patch", "skymap"),
                                 defaultTemplates={"inputCoaddName": 'deep', "outputCoaddName": "deep"}):
    schema = cT.InitInput(
        doc="Schema of the input detection catalog",
        name="{inputCoaddName}Coadd_det_schema",
        storageClass="SourceCatalog"
    )

    outputSchema = cT.InitOutput(
        doc="Schema of the merged detection catalog",
        name="{outputCoaddName}Coadd_mergeDet_schema",
        storageClass="SourceCatalog"
    )

    outputPeakSchema = cT.InitOutput(
        doc="Output schema of the Footprint peak catalog",
        name="{outputCoaddName}Coadd_peak_schema",
        storageClass="PeakCatalog"
    )

    catalogs = cT.Input(
        doc="Detection Catalogs to be merged",
        name="{inputCoaddName}Coadd_det",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "skymap", "band"),
        multiple=True
    )

    skyMap = cT.Input(
        doc="SkyMap to be used in merging",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )

    outputCatalog = cT.Output(
        doc="Merged Detection catalog",
        name="{outputCoaddName}Coadd_mergeDet",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "skymap"),
    )


class MergeDetectionsConfig(PipelineTaskConfig, pipelineConnections=MergeDetectionsConnections):
    """!
    @anchor MergeDetectionsConfig_

    @brief Configuration parameters for the MergeDetectionsTask.
    """
    minNewPeak = Field(dtype=float, default=1,
                       doc="Minimum distance from closest peak to create a new one (in arcsec).")

    maxSamePeak = Field(dtype=float, default=0.3,
                        doc="When adding new catalogs to the merge, all peaks less than this distance "
                        " (in arcsec) to an existing peak will be flagged as detected in that catalog.")
    cullPeaks = ConfigField(dtype=CullPeaksConfig, doc="Configuration for how to cull peaks.")

    skyFilterName = Field(dtype=str, default="sky",
                          doc="Name of `filter' used to label sky objects (e.g. flag merge_peak_sky is set)\n"
                          "(N.b. should be in MergeMeasurementsConfig.pseudoFilterList)")
    skyObjects = ConfigurableField(target=SkyObjectsTask, doc="Generate sky objects")
    priorityList = ListField(dtype=str, default=[],
                             doc="Priority-ordered list of filter bands for the merge.")
    coaddName = Field(dtype=str, default="deep", doc="Name of coadd")

    def setDefaults(self):
        Config.setDefaults(self)
        self.skyObjects.avoidMask = ["DETECTED"]  # Nothing else is available in our custom mask

    def validate(self):
        super().validate()
        if len(self.priorityList) == 0:
            raise RuntimeError("No priority list provided")


class MergeDetectionsTask(PipelineTask, CmdLineTask):
    r"""!
    @anchor MergeDetectionsTask_

    @brief Merge coadd detections from multiple bands.

    @section pipe_tasks_multiBand_Contents Contents

      - @ref pipe_tasks_multiBand_MergeDetectionsTask_Purpose
      - @ref pipe_tasks_multiBand_MergeDetectionsTask_Init
      - @ref pipe_tasks_multiBand_MergeDetectionsTask_Run
      - @ref pipe_tasks_multiBand_MergeDetectionsTask_Config
      - @ref pipe_tasks_multiBand_MergeDetectionsTask_Debug
      - @ref pipe_tasks_multiband_MergeDetectionsTask_Example

    @section pipe_tasks_multiBand_MergeDetectionsTask_Purpose	Description

    Command-line task that merges sources detected in coadds of exposures obtained with different filters.

    To perform photometry consistently across coadds in multiple filter bands, we create a master catalog of
    sources from all bands by merging the sources (peaks & footprints) detected in each coadd, while keeping
    track of which band each source originates in.

    The catalog merge is performed by @ref getMergedSourceCatalog. Spurious peaks detected around bright
    objects are culled as described in @ref CullPeaksConfig_.

      @par Inputs:
        deepCoadd_det{tract,patch,filter}: SourceCatalog (only parent Footprints)
      @par Outputs:
        deepCoadd_mergeDet{tract,patch}: SourceCatalog (only parent Footprints)
      @par Data Unit:
        tract, patch

    @section pipe_tasks_multiBand_MergeDetectionsTask_Init       Task initialisation

    @copydoc \_\_init\_\_

    @section pipe_tasks_multiBand_MergeDetectionsTask_Run       Invoking the Task

    @copydoc run

    @section pipe_tasks_multiBand_MergeDetectionsTask_Config       Configuration parameters

    See @ref MergeDetectionsConfig_

    @section pipe_tasks_multiBand_MergeDetectionsTask_Debug		Debug variables

    The @link lsst.pipe.base.cmdLineTask.CmdLineTask command line task@endlink interface supports a flag @c -d
    to import @b debug.py from your @c PYTHONPATH; see @ref baseDebug for more about @b debug.py files.

    MergeDetectionsTask has no debug variables.

    @section pipe_tasks_multiband_MergeDetectionsTask_Example	A complete example of using MergeDetectionsTask

    MergeDetectionsTask is meant to be run after detecting sources in coadds generated for the chosen subset
    of the available bands.
    The purpose of the task is to merge sources (peaks & footprints) detected in the coadds generated from the
    chosen subset of filters.
    Subsequent tasks in the multi-band processing procedure will deblend the generated master list of sources
    and, eventually, perform forced photometry.
    Command-line usage of MergeDetectionsTask expects data references for all the coadds to be processed.
    A list of the available optional arguments can be obtained by calling mergeCoaddDetections.py with the
    `--help` command line argument:
    @code
    mergeCoaddDetections.py --help
    @endcode

    To demonstrate usage of the DetectCoaddSourcesTask in the larger context of multi-band processing, we
    will process HSC data in the [ci_hsc](https://github.com/lsst/ci_hsc) package. Assuming one has finished
    step 5 at @ref pipeTasks_multiBand, one may merge the catalogs of sources from each coadd as follows:
    @code
    mergeCoaddDetections.py $CI_HSC_DIR/DATA --id patch=5,4 tract=0 filter=HSC-I^HSC-R
    @endcode
    This will merge the HSC-I & -R band parent source catalogs and write the results to
    `$CI_HSC_DIR/DATA/deepCoadd-results/merged/0/5,4/mergeDet-0-5,4.fits`.

    The next step in the multi-band processing procedure is
    @ref MeasureMergedCoaddSourcesTask_ "MeasureMergedCoaddSourcesTask"
    """
    ConfigClass = MergeDetectionsConfig
    RunnerClass = MergeSourcesRunner
    _DefaultName = "mergeCoaddDetections"
    inputDataset = "det"
    outputDataset = "mergeDet"
    makeIdFactory = _makeMakeIdFactory("MergedCoaddId")

    @classmethod
    def _makeArgumentParser(cls):
        return makeMergeArgumentParser(cls._DefaultName, cls.inputDataset)

    def getInputSchema(self, butler=None, schema=None):
        return getInputSchema(self, butler, schema)

    def __init__(self, butler=None, schema=None, initInputs=None, **kwargs):
        # Make PipelineTask-only wording less transitional after cmdlineTask is removed
        """!
        @brief Initialize the merge detections task.

        A @ref FootprintMergeList_ "FootprintMergeList" will be used to
        merge the source catalogs.

        @param[in] schema     the schema of the detection catalogs used as input to this one
        @param[in] butler     a butler used to read the input schema from disk, if schema is None
        @param[in] initInputs This a PipelineTask-only argument that holds all inputs passed in
                              through the PipelineTask middleware
        @param[in] **kwargs   keyword arguments to be passed to CmdLineTask.__init__

        The task will set its own self.schema attribute to the schema of the output merged catalog.
        """
        super().__init__(**kwargs)
        if initInputs is not None:
            schema = initInputs['schema'].schema

        self.makeSubtask("skyObjects")
        self.schema = self.getInputSchema(butler=butler, schema=schema)

        filterNames = list(self.config.priorityList)
        filterNames.append(self.config.skyFilterName)
        self.merged = afwDetect.FootprintMergeList(self.schema, filterNames)
        self.outputSchema = afwTable.SourceCatalog(self.schema)
        self.outputPeakSchema = afwDetect.PeakCatalog(self.merged.getPeakSchema())

    def runDataRef(self, patchRefList):
        catalogs = dict(readCatalog(self, patchRef) for patchRef in patchRefList)
        skyInfo = getSkyInfo(coaddName=self.config.coaddName, patchRef=patchRefList[0])
        idFactory = self.makeIdFactory(patchRefList[0])
        skySeed = patchRefList[0].get(self.config.coaddName + "MergedCoaddId")
        mergeCatalogStruct = self.run(catalogs, skyInfo, idFactory, skySeed)
        self.write(patchRefList[0], mergeCatalogStruct.outputCatalog)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        packedId, maxBits = butlerQC.quantum.dataId.pack("tract_patch", returnMaxBits=True)
        inputs["skySeed"] = packedId
        inputs["idFactory"] = afwTable.IdFactory.makeSource(packedId, 64 - maxBits)
        catalogDict = {ref.dataId['band']: cat for ref, cat in zip(inputRefs.catalogs,
                       inputs['catalogs'])}
        inputs['catalogs'] = catalogDict
        skyMap = inputs.pop('skyMap')
        # Can use the first dataId to find the tract and patch being worked on
        tractNumber = inputRefs.catalogs[0].dataId['tract']
        tractInfo = skyMap[tractNumber]
        patchInfo = tractInfo.getPatchInfo(inputRefs.catalogs[0].dataId['patch'])
        skyInfo = Struct(
            skyMap=skyMap,
            tractInfo=tractInfo,
            patchInfo=patchInfo,
            wcs=tractInfo.getWcs(),
            bbox=patchInfo.getOuterBBox()
        )
        inputs['skyInfo'] = skyInfo

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, catalogs, skyInfo, idFactory, skySeed):
        r"""!
        @brief Merge multiple catalogs.

        After ordering the catalogs and filters in priority order,
        @ref getMergedSourceCatalog of the @ref FootprintMergeList_ "FootprintMergeList" created by
        @ref \_\_init\_\_ is used to perform the actual merging. Finally, @ref cullPeaks is used to remove
        garbage peaks detected around bright objects.

        @param[in]  catalogs
        @param[in]  patchRef
        @param[out] mergedList
        """

        # Convert distance to tract coordinate
        tractWcs = skyInfo.wcs
        peakDistance = self.config.minNewPeak / tractWcs.getPixelScale().asArcseconds()
        samePeakDistance = self.config.maxSamePeak / tractWcs.getPixelScale().asArcseconds()

        # Put catalogs, filters in priority order
        orderedCatalogs = [catalogs[band] for band in self.config.priorityList if band in catalogs.keys()]
        orderedBands = [band for band in self.config.priorityList if band in catalogs.keys()]

        mergedList = self.merged.getMergedSourceCatalog(orderedCatalogs, orderedBands, peakDistance,
                                                        self.schema, idFactory,
                                                        samePeakDistance)

        #
        # Add extra sources that correspond to blank sky
        #
        skySourceFootprints = self.getSkySourceFootprints(mergedList, skyInfo, skySeed)
        if skySourceFootprints:
            key = mergedList.schema.find("merge_footprint_%s" % self.config.skyFilterName).key
            for foot in skySourceFootprints:
                s = mergedList.addNew()
                s.setFootprint(foot)
                s.set(key, True)

        # Sort Peaks from brightest to faintest
        for record in mergedList:
            record.getFootprint().sortPeaks()
        self.log.info("Merged to %d sources" % len(mergedList))
        # Attempt to remove garbage peaks
        self.cullPeaks(mergedList)
        return Struct(outputCatalog=mergedList)

    def cullPeaks(self, catalog):
        """!
        @brief Attempt to remove garbage peaks (mostly on the outskirts of large blends).

        @param[in] catalog Source catalog
        """
        keys = [item.key for item in self.merged.getPeakSchema().extract("merge_peak_*").values()]
        assert len(keys) > 0, "Error finding flags that associate peaks with their detection bands."
        totalPeaks = 0
        culledPeaks = 0
        for parentSource in catalog:
            # Make a list copy so we can clear the attached PeakCatalog and append the ones we're keeping
            # to it (which is easier than deleting as we iterate).
            keptPeaks = parentSource.getFootprint().getPeaks()
            oldPeaks = list(keptPeaks)
            keptPeaks.clear()
            familySize = len(oldPeaks)
            totalPeaks += familySize
            for rank, peak in enumerate(oldPeaks):
                if ((rank < self.config.cullPeaks.rankSufficient)
                    or (sum([peak.get(k) for k in keys]) >= self.config.cullPeaks.nBandsSufficient)
                    or (rank < self.config.cullPeaks.rankConsidered
                        and rank < self.config.cullPeaks.rankNormalizedConsidered * familySize)):
                    keptPeaks.append(peak)
                else:
                    culledPeaks += 1
        self.log.info("Culled %d of %d peaks" % (culledPeaks, totalPeaks))

    def getSchemaCatalogs(self):
        """!
        Return a dict of empty catalogs for each catalog dataset produced by this task.

        @param[out] dictionary of empty catalogs
        """
        mergeDet = afwTable.SourceCatalog(self.schema)
        peak = afwDetect.PeakCatalog(self.merged.getPeakSchema())
        return {self.config.coaddName + "Coadd_mergeDet": mergeDet,
                self.config.coaddName + "Coadd_peak": peak}

    def getSkySourceFootprints(self, mergedList, skyInfo, seed):
        """!
        @brief Return a list of Footprints of sky objects which don't overlap with anything in mergedList

        @param mergedList  The merged Footprints from all the input bands
        @param skyInfo     A description of the patch
        @param seed        Seed for the random number generator
        """
        mask = afwImage.Mask(skyInfo.patchInfo.getOuterBBox())
        detected = mask.getPlaneBitMask("DETECTED")
        for s in mergedList:
            s.getFootprint().spans.setMask(mask, detected)

        footprints = self.skyObjects.run(mask, seed)
        if not footprints:
            return footprints

        # Need to convert the peak catalog's schema so we can set the "merge_peak_<skyFilterName>" flags
        schema = self.merged.getPeakSchema()
        mergeKey = schema.find("merge_peak_%s" % self.config.skyFilterName).key
        converted = []
        for oldFoot in footprints:
            assert len(oldFoot.getPeaks()) == 1, "Should be a single peak only"
            peak = oldFoot.getPeaks()[0]
            newFoot = afwDetect.Footprint(oldFoot.spans, schema)
            newFoot.addPeak(peak.getFx(), peak.getFy(), peak.getPeakValue())
            newFoot.getPeaks()[0].set(mergeKey, True)
            converted.append(newFoot)

        return converted

    def write(self, patchRef, catalog):
        """!
        @brief Write the output.

        @param[in]  patchRef   data reference for patch
        @param[in]  catalog    catalog

        We write as the dataset provided by the 'outputDataset'
        class variable.
        """
        patchRef.put(catalog, self.config.coaddName + "Coadd_" + self.outputDataset)
        # since the filter isn't actually part of the data ID for the dataset we're saving,
        # it's confusing to see it in the log message, even if the butler simply ignores it.
        mergeDataId = patchRef.dataId.copy()
        del mergeDataId["filter"]
        self.log.info("Wrote merged catalog: %s" % (mergeDataId,))

    def writeMetadata(self, dataRefList):
        """!
        @brief No metadata to write, and not sure how to write it for a list of dataRefs.
        """
        pass
