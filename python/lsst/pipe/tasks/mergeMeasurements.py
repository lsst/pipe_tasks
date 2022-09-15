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

__all__ = ["MergeMeasurementsConfig", "MergeMeasurementsTask"]

import numpy
import warnings

import lsst.afw.table as afwTable
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

from lsst.pipe.base import PipelineTaskConnections, PipelineTaskConfig
import lsst.pipe.base.connectionTypes as cT


class MergeMeasurementsConnections(PipelineTaskConnections,
                                   dimensions=("skymap", "tract", "patch"),
                                   defaultTemplates={"inputCoaddName": "deep",
                                                     "outputCoaddName": "deep"}):
    inputSchema = cT.InitInput(
        doc="Schema for the output merged measurement catalog.",
        name="{inputCoaddName}Coadd_meas_schema",
        storageClass="SourceCatalog",
    )
    outputSchema = cT.InitOutput(
        doc="Schema for the output merged measurement catalog.",
        name="{outputCoaddName}Coadd_ref_schema",
        storageClass="SourceCatalog",
    )
    catalogs = cT.Input(
        doc="Input catalogs to merge.",
        name="{inputCoaddName}Coadd_meas",
        multiple=True,
        storageClass="SourceCatalog",
        dimensions=["band", "skymap", "tract", "patch"],
    )
    mergedCatalog = cT.Output(
        doc="Output merged catalog.",
        name="{outputCoaddName}Coadd_ref",
        storageClass="SourceCatalog",
        dimensions=["skymap", "tract", "patch"],
    )


class MergeMeasurementsConfig(PipelineTaskConfig, pipelineConnections=MergeMeasurementsConnections):
    """Configuration parameters for the MergeMeasurementsTask.
    """
    pseudoFilterList = pexConfig.ListField(
        dtype=str,
        default=["sky"],
        doc="Names of filters which may have no associated detection\n"
        "(N.b. should include MergeDetectionsConfig.skyFilterName)"
    )
    snName = pexConfig.Field(
        dtype=str,
        default="base_PsfFlux",
        doc="Name of flux measurement for calculating the S/N when choosing the reference band."
    )
    minSN = pexConfig.Field(
        dtype=float,
        default=10.,
        doc="If the S/N from the priority band is below this value (and the S/N "
        "is larger than minSNDiff compared to the priority band), use the band with "
        "the largest S/N as the reference band."
    )
    minSNDiff = pexConfig.Field(
        dtype=float,
        default=3.,
        doc="If the difference in S/N between another band and the priority band is larger "
        "than this value (and the S/N in the priority band is less than minSN) "
        "use the band with the largest S/N as the reference band"
    )
    flags = pexConfig.ListField(
        dtype=str,
        doc="Require that these flags, if available, are not set",
        default=["base_PixelFlags_flag_interpolatedCenter", "base_PsfFlux_flag",
                 "ext_photometryKron_KronFlux_flag", "modelfit_CModel_flag", ]
    )
    priorityList = pexConfig.ListField(
        dtype=str,
        default=[],
        doc="Priority-ordered list of filter bands for the merge."
    )
    coaddName = pexConfig.Field(
        dtype=str,
        default="deep",
        doc="Name of coadd"
    )

    def validate(self):
        super().validate()
        if len(self.priorityList) == 0:
            raise RuntimeError("No priority list provided")


class MergeMeasurementsTask(pipeBase.PipelineTask):
    """Merge measurements from multiple bands.

    Combines consistent (i.e. with the same peaks and footprints) catalogs of
    sources from multiple filter bands to construct a unified catalog that is
    suitable for driving forced photometry. Every source is required to have
    centroid, shape and flux measurements in each band.

    MergeMeasurementsTask is meant to be run after deblending & measuring
    sources in every band. The purpose of the task is to generate a catalog of
    sources suitable for driving forced photometry in coadds and individual
    exposures.

    Parameters
    ----------
    butler : `None`, optional
        Compatibility parameter. Should always be `None`.
    schema : `lsst.afw.table.Schema`, optional
        The schema of the detection catalogs used as input to this task.
    initInputs : `dict`, optional
        Dictionary that can contain a key ``inputSchema`` containing the
        input schema. If present will override the value of ``schema``.
    **kwargs
        Additional keyword arguments.
    """

    _DefaultName = "mergeCoaddMeasurements"
    ConfigClass = MergeMeasurementsConfig

    inputDataset = "meas"
    outputDataset = "ref"

    def __init__(self, butler=None, schema=None, initInputs=None, **kwargs):
        super().__init__(**kwargs)

        if butler is not None:
            warnings.warn("The 'butler' parameter is no longer used and can be safely removed.",
                          category=FutureWarning, stacklevel=2)
            butler = None

        if initInputs is not None:
            schema = initInputs['inputSchema'].schema

        if schema is None:
            raise ValueError("No input schema or initInputs['inputSchema'] provided.")

        inputSchema = schema

        self.schemaMapper = afwTable.SchemaMapper(inputSchema, True)
        self.schemaMapper.addMinimalSchema(inputSchema, True)
        self.instFluxKey = inputSchema.find(self.config.snName + "_instFlux").getKey()
        self.instFluxErrKey = inputSchema.find(self.config.snName + "_instFluxErr").getKey()
        self.fluxFlagKey = inputSchema.find(self.config.snName + "_flag").getKey()

        self.flagKeys = {}
        for band in self.config.priorityList:
            outputKey = self.schemaMapper.editOutputSchema().addField(
                "merge_measurement_%s" % band,
                type="Flag",
                doc="Flag field set if the measurements here are from the %s filter" % band
            )
            peakKey = inputSchema.find("merge_peak_%s" % band).key
            footprintKey = inputSchema.find("merge_footprint_%s" % band).key
            self.flagKeys[band] = pipeBase.Struct(peak=peakKey, footprint=footprintKey, output=outputKey)
        self.schema = self.schemaMapper.getOutputSchema()

        self.pseudoFilterKeys = []
        for filt in self.config.pseudoFilterList:
            try:
                self.pseudoFilterKeys.append(self.schema.find("merge_peak_%s" % filt).getKey())
            except Exception as e:
                self.log.warning("merge_peak is not set for pseudo-filter %s: %s", filt, e)

        self.badFlags = {}
        for flag in self.config.flags:
            try:
                self.badFlags[flag] = self.schema.find(flag).getKey()
            except KeyError as exc:
                self.log.warning("Can't find flag %s in schema: %s", flag, exc)
        self.outputSchema = afwTable.SourceCatalog(self.schema)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        dataIds = (ref.dataId for ref in inputRefs.catalogs)
        catalogDict = {dataId['band']: cat for dataId, cat in zip(dataIds, inputs['catalogs'])}
        inputs['catalogs'] = catalogDict
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, catalogs):
        """Merge measurement catalogs to create a single reference catalog for forced photometry.

        Parameters
        ----------
        catalogs : `lsst.afw.table.SourceCatalog`
            Catalogs to be merged.

        Raises
        ------
        ValueError
            Raised if no catalog records were found;
            if there is no valid reference for the input record ID;
            or if there is a mismatch between catalog sizes.

        Notes
        -----
        For parent sources, we choose the first band in config.priorityList for which the
        merge_footprint flag for that band is is True.

        For child sources, the logic is the same, except that we use the merge_peak flags.
        """
        # Put catalogs, filters in priority order
        orderedCatalogs = [catalogs[band] for band in self.config.priorityList if band in catalogs.keys()]
        orderedKeys = [self.flagKeys[band] for band in self.config.priorityList if band in catalogs.keys()]

        mergedCatalog = afwTable.SourceCatalog(self.schema)
        mergedCatalog.reserve(len(orderedCatalogs[0]))

        idKey = orderedCatalogs[0].table.getIdKey()
        for catalog in orderedCatalogs[1:]:
            if numpy.any(orderedCatalogs[0].get(idKey) != catalog.get(idKey)):
                raise ValueError("Error in inputs to MergeCoaddMeasurements: source IDs do not match")

        # This first zip iterates over all the catalogs simultaneously, yielding a sequence of one
        # record for each band, in priority order.
        for orderedRecords in zip(*orderedCatalogs):

            maxSNRecord = None
            maxSNFlagKeys = None
            maxSN = 0.
            priorityRecord = None
            priorityFlagKeys = None
            prioritySN = 0.
            hasPseudoFilter = False

            # Now we iterate over those record-band pairs, keeping track of the priority and the
            # largest S/N band.
            for inputRecord, flagKeys in zip(orderedRecords, orderedKeys):
                parent = (inputRecord.getParent() == 0 and inputRecord.get(flagKeys.footprint))
                child = (inputRecord.getParent() != 0 and inputRecord.get(flagKeys.peak))

                if not (parent or child):
                    for pseudoFilterKey in self.pseudoFilterKeys:
                        if inputRecord.get(pseudoFilterKey):
                            hasPseudoFilter = True
                            priorityRecord = inputRecord
                            priorityFlagKeys = flagKeys
                            break
                    if hasPseudoFilter:
                        break

                isBad = any(inputRecord.get(flag) for flag in self.badFlags)
                if isBad or inputRecord.get(self.fluxFlagKey) or inputRecord.get(self.instFluxErrKey) == 0:
                    sn = 0.
                else:
                    sn = inputRecord.get(self.instFluxKey)/inputRecord.get(self.instFluxErrKey)
                if numpy.isnan(sn) or sn < 0.:
                    sn = 0.
                if (parent or child) and priorityRecord is None:
                    priorityRecord = inputRecord
                    priorityFlagKeys = flagKeys
                    prioritySN = sn
                if sn > maxSN:
                    maxSNRecord = inputRecord
                    maxSNFlagKeys = flagKeys
                    maxSN = sn

            # If the priority band has a low S/N we would like to choose the band with the highest S/N as
            # the reference band instead.  However, we only want to choose the highest S/N band if it is
            # significantly better than the priority band.  Therefore, to choose a band other than the
            # priority, we require that the priority S/N is below the minimum threshold and that the
            # difference between the priority and highest S/N is larger than the difference threshold.
            #
            # For pseudo code objects we always choose the first band in the priority list.
            bestRecord = None
            bestFlagKeys = None
            if hasPseudoFilter:
                bestRecord = priorityRecord
                bestFlagKeys = priorityFlagKeys
            elif (prioritySN < self.config.minSN and (maxSN - prioritySN) > self.config.minSNDiff
                  and maxSNRecord is not None):
                bestRecord = maxSNRecord
                bestFlagKeys = maxSNFlagKeys
            elif priorityRecord is not None:
                bestRecord = priorityRecord
                bestFlagKeys = priorityFlagKeys

            if bestRecord is not None and bestFlagKeys is not None:
                outputRecord = mergedCatalog.addNew()
                outputRecord.assign(bestRecord, self.schemaMapper)
                outputRecord.set(bestFlagKeys.output, True)
            else:  # if we didn't find any records
                raise ValueError("Error in inputs to MergeCoaddMeasurements: no valid reference for %s" %
                                 inputRecord.getId())

        # more checking for sane inputs, since zip silently iterates over the smallest sequence
        for inputCatalog in orderedCatalogs:
            if len(mergedCatalog) != len(inputCatalog):
                raise ValueError("Mismatch between catalog sizes: %s != %s" %
                                 (len(mergedCatalog), len(orderedCatalogs)))

        return pipeBase.Struct(
            mergedCatalog=mergedCatalog
        )
