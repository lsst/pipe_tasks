#
# LSST Data Management System
# Copyright 2008-2017 AURA/LSST.
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
from __future__ import absolute_import, division, print_function
from builtins import range

import numpy as np

from lsst.pipe.supertask import SuperTask, Quantum
from lsst.pex.config import (Config, Field, ListField, ConfigField)
from lsst.afw.table import IdFactory, SourceCatalog
from lsst.afw.detection import FootprintMergeList, PeakCatalog, Footprint
from lsst.afw.geom import SpanSet
from lsst.afw.image import MaskU
from lsst.obs.base import repodb
from .multiBand import CullPeaksConfig, getShortFilterName


def makeMergedIdFactory(self, dataset, butler):
    """Return an IdFactory for SourceCatalogs that includes the Coadd ID.

    The actual parameters used in the IdFactory are provided by
    the butler (through the provided data reference.
    """
    expBits = butler.get("deepMergedCoaddId_bits")
    expId = int(butler.get("deepMergedCoaddId", dataset.getDataId()))
    return IdFactory.makeSource(expId, 64 - expBits)


def getInputSchema(self, DatasetClass, butler=None, schema=None):
    """Obtain the input schema either directly or froma  butler reference.

    Parameters
    ----------
    butler : `daf.persistence.Butler`
        butler reference to obtain the input schema from
    schema : `afw.table.Schema`
        the input schema
    """
    if schema is None:
        assert butler is not None, "Neither butler nor schema specified"
        schema = butler.get(DatasetClass.name + "_schema").schema
    return schema


class MergeDetectionsConfig(Config):
    priorityList = ListField(
        dtype=str, default=[],
        doc="Priority-ordered list of bands for the merge.")
    minNewPeak = Field(
        dtype=float, default=1,
        doc="Minimum distance from closest peak to create a new one (arcsec).")
    maxSamePeak = Field(
        dtype=float, default=0.3,
        doc=("When adding new catalogs to the merge, all peaks less than this "
             "distance (in arcsec) to an existing peak will be flagged as "
             "detected in that catalog.")
    )
    cullPeaks = ConfigField(
        dtype=CullPeaksConfig,
        doc="Configuration for how to cull peaks."
    )
    skyFilterName = Field(
        dtype=str, default="sky",
        doc=("Name of `filter' used to label sky objects (e.g. flag "
             "merge_peak_sky is set)\n"
             "(N.b. should be in MergeMeasurementsConfig.pseudoFilterList)")
    )
    skySourceRadius = Field(
        dtype=float, default=8,
        doc="Radius, in pixels, of sky objects"
    )
    skyGrowDetectedFootprints = Field(
        dtype=int, default=0,
        doc=("Number of pixels to grow the detected footprint mask "
             "when adding sky objects")
    )
    nSkySourcesPerPatch = Field(
        dtype=int, default=100,
        doc=("Try to add this many sky objects to the mergeDet list, which will\n"
             "then be measured along with the detected objects in sourceMeasurementTask")
    )
    nTrialSkySourcesPerPatch = Field(
        dtype=int, default=None, optional=True,
        doc=("Maximum number of trial sky object positions\n"
             "(default: nSkySourcesPerPatch*nTrialSkySourcesPerPatchMultiplier)")
    )
    nTrialSkySourcesPerPatchMultiplier = Field(
        dtype=int, default=5,
        doc=("Set nTrialSkySourcesPerPatch to\n"
             "    nSkySourcesPerPatch*nTrialSkySourcesPerPatchMultiplier\n"
             "if nTrialSkySourcesPerPatch is None")
    )
    inputCatalog = Field(
        dtype=str, default="deepCoadd_det",
        doc="Name of per-band Footprint-only catalog dataset used as input."
    )
    outputCatalog = Field(
        dtype=str, default="deepCoadd_mergeDet",
        doc="Name of cross-band Footprint-only catalog dataset used as output."
    )

    def validate(self):
        Config.validate(self)
        if len(self.priorityList) == 0:
            raise RuntimeError("No priority list provided")


class MergeDetectionsTask(SuperTask):
    ConfigClass = MergeDetectionsConfig
    _DefaultName = "mergeCoaddDetections"

    def __init__(self, butler=None, schema=None, **kwargs):
        SuperTask.__init__(self, butler=butler, **kwargs)
        self.InputCatalog = repodb.Dataset.subclass(
            self.config.inputCatalog,
            patch=repodb.PatchUnit,
            tract=repodb.TractUnit,
            filter=repodb.FilterUnit
        )
        self.OutputCatalog = repodb.Dataset.subclass(
            self.config.outputCatalog,
            patch=repodb.PatchUnit,
            tract=repodb.TractUnit
        )

        self.schema = getInputSchema(
            self.InputCatalog,
            butler=butler, schema=schema
        )
        filterNames = [getShortFilterName(name)
                       for name in self.config.priorityList]
        if self.config.nSkySourcesPerPatch > 0:
            filterNames += [self.config.skyFilterName]
        self.merged = FootprintMergeList(self.schema, filterNames)

    def defineQuanta(self, repoGraph, butler):
        """Return a set of discrete work packages that can be run independently
        via runQuantum.
        """
        quanta = {}
        byTractAndPatch = {}
        for inputCatalog in repoGraph.datasets[self.InputCatalog]:
            byTractAndPatch.setdefault(
                (inputCatalog.tract, inputCatalog.patch),
                set()
            ).add(inputCatalog)
        for (tract, patch), inputs in byTractAndPatch.items():
            outputCatalog = repoGraph.addDataset(
                self.OutputCatalog,
                tract=tract,
                patch=patch
            )
            quanta.append(
                Quantum(
                    inputs={self.InputCatalog: inputs},
                    outputs={self.OutputCatalog: set([outputCatalog])}
                )
            )
        return quanta

    def runQuantum(self, quantum, butler):
        """Merge coadd detection catalogs using a Butler for inputs and outputs.
        """
        inputCatalogs = {dataset.filter.name: dataset.get(butler)
                         for dataset in quantum.inputs[self.InputCatalog]}
        outputCatalogDataset, = quantum.outputs[self.OutputCatalog]
        idFactory = makeMergedIdFactory(outputCatalogDataset, butler)
        skyMap = butler.get("deepCoadd_skyMap")
        tractInfo = skyMap.findTract(outputCatalogDataset.tract.number)
        tractWcs = tractInfo.getWcs()
        patchBBox = tractInfo[outputCatalogDataset.patch.x,
                              outputCatalogDataset.patch.y].getOuterBBox()
        merged = self.run(inputCatalogs, idFactory, tractWcs, patchBBox)
        outputCatalogDataset.put(butler, merged)
        return merged

    def run(self, catalogs, idFactory, tractWcs, patchBBox):
        # Convert distance to tract coordinate
        peakDistance = (self.config.minNewPeak /
                        tractWcs.pixelScale().asArcseconds())
        samePeakDistance = (self.config.maxSamePeak /
                            tractWcs.pixelScale().asArcseconds())

        # Put catalogs, filters in priority order
        orderedCatalogs = [catalogs[band] for band in self.config.priorityList
                           if band in catalogs.keys()]
        orderedBands = [getShortFilterName(band)
                        for band in self.config.priorityList
                        if band in catalogs.keys()]

        mergedList = self.merged.getMergedSourceCatalog(
            orderedCatalogs, orderedBands, peakDistance,
            self.schema, idFactory, samePeakDistance
        )

        #
        # Add extra sources that correspond to blank sky
        #
        skySourceFootprints = self.getSkySourceFootprints(
            mergedList, patchBBox,
            self.config.skyGrowDetectedFootprints
        )
        if skySourceFootprints:
            key = mergedList.schema.find("merge_footprint_%s" %
                                         self.config.skyFilterName).key

            for foot in skySourceFootprints:
                s = mergedList.addNew()
                s.setFootprint(foot)
                s.set(key, True)

            self.log.info("Added %d sky sources (%.0f%% of requested)",
                          len(skySourceFootprints),
                          (100*len(skySourceFootprints) /
                           float(self.config.nSkySourcesPerPatch)))

        # Sort Peaks from brightest to faintest
        for record in mergedList:
            record.getFootprint().sortPeaks()
        self.log.info("Merged to %d sources" % len(mergedList))
        # Attempt to remove garbage peaks
        self.cullPeaks(mergedList)
        return mergedList

    def cullPeaks(self, catalog):
        """!
        \Attempt to remove garbage peaks (mostly on the outskirts of large blends).

        \param[in] catalog Source catalog
        """
        keys = [item.key for item in
                self.merged.getPeakSchema().extract("merge.peak.*").values()]
        totalPeaks = 0
        culledPeaks = 0
        for parentSource in catalog:
            # Make a list copy so we can clear the attached PeakCatalog and
            # append the ones we're keeping to it (which is easier than
            # deleting as we iterate).
            keptPeaks = parentSource.getFootprint().getPeaks()
            oldPeaks = list(keptPeaks)
            keptPeaks.clear()
            familySize = len(oldPeaks)
            totalPeaks += familySize
            for rank, peak in enumerate(oldPeaks):
                if ((rank < self.config.cullPeaks.rankSufficient) or
                    (self.config.cullPeaks.nBandsSufficient > 1 and
                     sum([peak.get(k) for k in keys]) >= self.config.cullPeaks.nBandsSufficient) or
                    (rank < self.config.cullPeaks.rankConsidered and
                     rank < self.config.cullPeaks.rankNormalizedConsidered * familySize)):
                    keptPeaks.append(peak)
                else:
                    culledPeaks += 1
        self.log.info("Culled %d of %d peaks" % (culledPeaks, totalPeaks))

    def getSchemaCatalogs(self):
        """!
        Return a dict of empty catalogs for each catalog dataset produced by this task.

        \param[out] dictionary of empty catalogs
        """
        mergeDet = SourceCatalog(self.schema)
        peak = PeakCatalog(self.merged.getPeakSchema())
        return {self.config.coaddName + "Coadd_mergeDet": mergeDet,
                self.config.coaddName + "Coadd_peak": peak}

    def getSkySourceFootprints(self, mergedList, patchBBox,
                               growDetectedFootprints=0):

        if self.config.nSkySourcesPerPatch <= 0:
            return []

        skySourceRadius = self.config.skySourceRadius
        nSkySourcesPerPatch = self.config.nSkySourcesPerPatch
        nTrialSkySourcesPerPatch = self.config.nTrialSkySourcesPerPatch
        if nTrialSkySourcesPerPatch is None:
            nTrialSkySourcesPerPatch = (
                nSkySourcesPerPatch *
                self.config.nTrialSkySourcesPerPatchMultiplier
            )
        #
        # We are going to find circular Footprints that don't intersect any
        # pre-existing Footprints, and the easiest way to do this is to
        # generate a Mask containing all the detected pixels (as merged by
        # this task).
        #
        mask = MaskU(patchBBox)
        detectedMask = mask.getPlaneBitMask("DETECTED")
        for s in mergedList:
            foot = s.getFootprint()
            if growDetectedFootprints > 0:
                foot.dilate(growDetectedFootprints)
            foot.spans.setMask(mask, detectedMask)

        xmin, ymin = patchBBox.getMin()
        xmax, ymax = patchBBox.getMax()
        # Avoid objects partially off the image
        xmin += skySourceRadius + 1
        ymin += skySourceRadius + 1
        xmax -= skySourceRadius + 1
        ymax -= skySourceRadius + 1

        skySourceFootprints = []
        maskToSpanSet = SpanSet.fromMask(mask)
        for i in range(nTrialSkySourcesPerPatch):
            if len(skySourceFootprints) == nSkySourcesPerPatch:
                break

            x = int(np.random.uniform(xmin, xmax))
            y = int(np.random.uniform(ymin, ymax))
            spans = SpanSet.fromShape(int(skySourceRadius), offset=(x, y))
            foot = Footprint(spans, patchBBox)
            foot.setPeakSchema(self.merged.getPeakSchema())

            if not foot.spans.overlaps(maskToSpanSet):
                foot.addPeak(x, y, 0)
                foot.getPeaks()[0].set(
                    "merge_peak_%s" % self.config.skyFilterName,
                    True
                )
                skySourceFootprints.append(foot)

        return skySourceFootprints

    def getDatasetClasses(self):
        return dict(
            inputs={self.InputCatalog.name: self.InputCatalog},
            outputs={self.OutputCatalog.name: self.OutputCatalog}
        )
