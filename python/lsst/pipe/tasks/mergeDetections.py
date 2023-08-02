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

__all__ = ["MergeDetectionsConfig", "MergeDetectionsTask"]

import numpy as np
from numpy.lib.recfunctions import rec_join

from .multiBandUtils import CullPeaksConfig

import lsst.afw.detection as afwDetect
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable

from lsst.meas.algorithms import SkyObjectsTask
from lsst.skymap import BaseSkyMap
from lsst.pex.config import Config, Field, ListField, ConfigurableField, ConfigField
from lsst.pipe.base import (PipelineTask, PipelineTaskConfig, Struct,
                            PipelineTaskConnections)
import lsst.pipe.base.connectionTypes as cT
from lsst.meas.base import SkyMapIdGeneratorConfig


def matchCatalogsExact(catalog1, catalog2, patch1=None, patch2=None):
    """Match two catalogs derived from the same mergeDet catalog.

    When testing downstream features, like deblending methods/parameters
    and measurement algorithms/parameters, it is useful to to compare
    the same sources in two catalogs. In most cases this must be done
    by matching on either RA/DEC or XY positions, which occassionally
    will mismatch one source with another.

    For a more robust solution, as long as the downstream catalog is
    derived from the same mergeDet catalog, exact source matching
    can be done via the unique ``(parent, deblend_peakID)``
    combination. So this function performs this exact matching for
    all sources both catalogs.

    Parameters
    ----------
    catalog1, catalog2 : `lsst.afw.table.SourceCatalog`
        The two catalogs to merge
    patch1, patch2 : `array` of `int`
        Patch for each row, converted into an integer.

    Returns
    -------
    result : `list` of `lsst.afw.table.SourceMatch`
        List of matches for each source (using an inner join).
    """
    # Only match the individual sources, the parents will
    # already be matched by the mergeDet catalog
    sidx1 = catalog1["parent"] != 0
    sidx2 = catalog2["parent"] != 0

    # Create the keys used to merge the catalogs
    parents1 = np.array(catalog1["parent"][sidx1])
    peaks1 = np.array(catalog1["deblend_peakId"][sidx1])
    index1 = np.arange(len(catalog1))[sidx1]
    parents2 = np.array(catalog2["parent"][sidx2])
    peaks2 = np.array(catalog2["deblend_peakId"][sidx2])
    index2 = np.arange(len(catalog2))[sidx2]

    if patch1 is not None:
        if patch2 is None:
            msg = ("If the catalogs are from different patches then patch1 and patch2 must be specified"
                   ", got {} and {}").format(patch1, patch2)
            raise ValueError(msg)
        patch1 = patch1[sidx1]
        patch2 = patch2[sidx2]

        key1 = np.rec.array((parents1, peaks1, patch1, index1),
                            dtype=[('parent', np.int64), ('peakId', np.int32),
                                   ("patch", patch1.dtype), ("index", np.int32)])
        key2 = np.rec.array((parents2, peaks2, patch2, index2),
                            dtype=[('parent', np.int64), ('peakId', np.int32),
                                   ("patch", patch2.dtype), ("index", np.int32)])
        matchColumns = ("parent", "peakId", "patch")
    else:
        key1 = np.rec.array((parents1, peaks1, index1),
                            dtype=[('parent', np.int64), ('peakId', np.int32), ("index", np.int32)])
        key2 = np.rec.array((parents2, peaks2, index2),
                            dtype=[('parent', np.int64), ('peakId', np.int32), ("index", np.int32)])
        matchColumns = ("parent", "peakId")
    # Match the two keys.
    # This line performs an inner join on the structured
    # arrays `key1` and `key2`, which stores their indices
    # as columns in a structured array.
    matched = rec_join(matchColumns, key1, key2, jointype="inner")

    # Create the full index for both catalogs
    indices1 = matched["index1"]
    indices2 = matched["index2"]

    # Re-index the resulting catalogs
    matches = [
        afwTable.SourceMatch(catalog1[int(i1)], catalog2[int(i2)], 0.0)
        for i1, i2 in zip(indices1, indices2)
    ]

    return matches


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
    """Configuration parameters for the MergeDetectionsTask.
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
    idGenerator = SkyMapIdGeneratorConfig.make_field()

    def setDefaults(self):
        Config.setDefaults(self)
        self.skyObjects.avoidMask = ["DETECTED"]  # Nothing else is available in our custom mask

    def validate(self):
        super().validate()
        if len(self.priorityList) == 0:
            raise RuntimeError("No priority list provided")


class MergeDetectionsTask(PipelineTask):
    """Merge sources detected in coadds of exposures obtained with different filters.

    Merge sources detected in coadds of exposures obtained with different
    filters. To perform photometry consistently across coadds in multiple
    filter bands, we create a master catalog of sources from all bands by
    merging the sources (peaks & footprints) detected in each coadd, while
    keeping track of which band each source originates in. The catalog merge
    is performed by
    `~lsst.afw.detection.FootprintMergeList.getMergedSourceCatalog`. Spurious
    peaks detected around bright objects are culled as described in
    `~lsst.pipe.tasks.multiBandUtils.CullPeaksConfig`.

    MergeDetectionsTask is meant to be run after detecting sources in coadds
    generated for the chosen subset of the available bands. The purpose of the
    task is to merge sources (peaks & footprints) detected in the coadds
    generated from the chosen subset of filters. Subsequent tasks in the
    multi-band processing procedure will deblend the generated master list of
    sources and, eventually, perform forced photometry.

    Parameters
    ----------
    schema : `lsst.afw.table.Schema`, optional
        The schema of the detection catalogs used as input to this task.
    initInputs : `dict`, optional
        Dictionary that can contain a key ``schema`` containing the
        input schema. If present will override the value of ``schema``.
    **kwargs
        Additional keyword arguments.
    """
    ConfigClass = MergeDetectionsConfig
    _DefaultName = "mergeCoaddDetections"

    def __init__(self, schema=None, initInputs=None, **kwargs):
        super().__init__(**kwargs)

        if initInputs is not None:
            schema = initInputs['schema'].schema

        if schema is None:
            raise ValueError("No input schema or initInputs['schema'] provided.")

        self.schema = schema

        self.makeSubtask("skyObjects")

        filterNames = list(self.config.priorityList)
        filterNames.append(self.config.skyFilterName)
        self.merged = afwDetect.FootprintMergeList(self.schema, filterNames)
        self.outputSchema = afwTable.SourceCatalog(self.schema)
        self.outputPeakSchema = afwDetect.PeakCatalog(self.merged.getPeakSchema())

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        idGenerator = self.config.idGenerator.apply(butlerQC.quantum.dataId)
        inputs["skySeed"] = idGenerator.catalog_id
        inputs["idFactory"] = idGenerator.make_table_id_factory()
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
        """Merge multiple catalogs.

        After ordering the catalogs and filters in priority order,
        ``getMergedSourceCatalog`` of the
        `~lsst.afw.detection.FootprintMergeList` created by ``__init__`` is
        used to perform the actual merging. Finally, `cullPeaks` is used to
        remove garbage peaks detected around bright objects.

        Parameters
        ----------
        catalogs : `lsst.afw.table.SourceCatalog`
            Catalogs to be merged.
        mergedList : `lsst.afw.table.SourceCatalog`
            Merged catalogs.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``outputCatalog``
                Merged catalogs (`lsst.afw.table.SourceCatalog`).
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
        self.log.info("Merged to %d sources", len(mergedList))
        # Attempt to remove garbage peaks
        self.cullPeaks(mergedList)
        return Struct(outputCatalog=mergedList)

    def cullPeaks(self, catalog):
        """Attempt to remove garbage peaks (mostly on the outskirts of large blends).

        Parameters
        ----------
        catalog : `lsst.afw.table.SourceCatalog`
            Source catalog.
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
        self.log.info("Culled %d of %d peaks", culledPeaks, totalPeaks)

    def getSkySourceFootprints(self, mergedList, skyInfo, seed):
        """Return a list of Footprints of sky objects which don't overlap with anything in mergedList.

        Parameters
        ----------
        mergedList : `lsst.afw.table.SourceCatalog`
            The merged Footprints from all the input bands.
        skyInfo : `lsst.pipe.base.Struct`
            A description of the patch.
        seed : `int`
            Seed for the random number generator.
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
