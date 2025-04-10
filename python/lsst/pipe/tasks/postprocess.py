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

__all__ = ["WriteObjectTableConfig", "WriteObjectTableTask",
           "WriteSourceTableConfig", "WriteSourceTableTask",
           "WriteRecalibratedSourceTableConfig", "WriteRecalibratedSourceTableTask",
           "PostprocessAnalysis",
           "TransformCatalogBaseConfig", "TransformCatalogBaseTask",
           "TransformObjectCatalogConfig", "TransformObjectCatalogTask",
           "ConsolidateObjectTableConfig", "ConsolidateObjectTableTask",
           "TransformSourceTableConfig", "TransformSourceTableTask",
           "ConsolidateVisitSummaryConfig", "ConsolidateVisitSummaryTask",
           "ConsolidateSourceTableConfig", "ConsolidateSourceTableTask",
           "MakeCcdVisitTableConfig", "MakeCcdVisitTableTask",
           "MakeVisitTableConfig", "MakeVisitTableTask",
           "WriteForcedSourceTableConfig", "WriteForcedSourceTableTask",
           "TransformForcedSourceTableConfig", "TransformForcedSourceTableTask",
           "ConsolidateTractConfig", "ConsolidateTractTask"]

from collections import defaultdict
import dataclasses
import functools
import logging
import numbers
import os

import numpy as np
import pandas as pd
import astropy.table
from astro_metadata_translator.headers import merge_headers

import lsst.geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.daf.base as dafBase
from lsst.daf.butler.formatters.parquet import pandas_to_astropy
from lsst.pipe.base import NoWorkFound, connectionTypes
import lsst.afw.table as afwTable
from lsst.afw.image import ExposureSummaryStats, ExposureF
from lsst.meas.base import SingleFrameMeasurementTask, DetectorVisitIdGeneratorConfig
from lsst.obs.base.utils import strip_provenance_from_fits_header

from .functors import CompositeFunctor, Column

log = logging.getLogger(__name__)


def flattenFilters(df, noDupCols=["coord_ra", "coord_dec"], camelCase=False, inputBands=None):
    """Flattens a dataframe with multilevel column index.
    """
    newDf = pd.DataFrame()
    # band is the level 0 index
    dfBands = df.columns.unique(level=0).values
    for band in dfBands:
        subdf = df[band]
        columnFormat = "{0}{1}" if camelCase else "{0}_{1}"
        newColumns = {c: columnFormat.format(band, c)
                      for c in subdf.columns if c not in noDupCols}
        cols = list(newColumns.keys())
        newDf = pd.concat([newDf, subdf[cols].rename(columns=newColumns)], axis=1)

    # Band must be present in the input and output or else column is all NaN:
    presentBands = dfBands if inputBands is None else list(set(inputBands).intersection(dfBands))
    # Get the unexploded columns from any present band's partition
    noDupDf = df[presentBands[0]][noDupCols]
    newDf = pd.concat([noDupDf, newDf], axis=1)
    return newDf


class TableVStack:
    """A helper class for stacking astropy tables without having them all in
    memory at once.

    Parameters
    ----------
    capacity : `int`
        Full size of the final table.

    Notes
    -----
    Unlike `astropy.table.vstack`, this class requires all tables to have the
    exact same columns (it's slightly more strict than even the
    ``join_type="exact"`` argument to `astropy.table.vstack`).
    """

    def __init__(self, capacity):
        self.index = 0
        self.capacity = capacity
        self.result = None

    @classmethod
    def from_handles(cls, handles):
        """Construct from an iterable of
        `lsst.daf.butler.DeferredDatasetHandle`.

        Parameters
        ----------
        handles : `~collections.abc.Iterable` [ \
                `lsst.daf.butler.DeferredDatasetHandle` ]
            Iterable of handles.   Must have a storage class that supports the
            "rowcount" component, which is all that will be fetched.

        Returns
        -------
        vstack : `TableVStack`
            An instance of this class, initialized with capacity equal to the
            sum of the rowcounts of all the given table handles.
        """
        capacity = sum(handle.get(component="rowcount") for handle in handles)
        return cls(capacity=capacity)

    def extend(self, table):
        """Add a single table to the stack.

        Parameters
        ----------
        table : `astropy.table.Table`
            An astropy table instance.
        """
        if self.result is None:
            self.result = astropy.table.Table()
            for name in table.colnames:
                column = table[name]
                column_cls = type(column)
                self.result[name] = column_cls.info.new_like([column], self.capacity, name=name)
                self.result[name][:len(table)] = column
            self.index = len(table)
            self.result.meta = table.meta.copy()
        else:
            next_index = self.index + len(table)
            if set(self.result.colnames) != set(table.colnames):
                raise TypeError(
                    "Inconsistent columns in concatentation: "
                    f"{set(self.result.colnames).symmetric_difference(table.colnames)}"
                )
            for name in table.colnames:
                out_col = self.result[name]
                in_col = table[name]
                if out_col.dtype != in_col.dtype:
                    raise TypeError(f"Type mismatch on column {name!r}: {out_col.dtype} != {in_col.dtype}.")
                self.result[name][self.index:next_index] = table[name]
            self.index = next_index
            # Butler provenance should be stripped on merge. It will be
            # added by butler on write. No attempt is made here to combine
            # provenance from multiple input tables.
            self.result.meta = merge_headers([self.result.meta, table.meta], mode="drop")
            strip_provenance_from_fits_header(self.result.meta)

    @classmethod
    def vstack_handles(cls, handles):
        """Vertically stack tables represented by deferred dataset handles.

        Parameters
        ----------
        handles : `~collections.abc.Iterable` [ \
                `lsst.daf.butler.DeferredDatasetHandle` ]
            Iterable of handles.   Must have the "ArrowAstropy" storage class
            and identical columns.

        Returns
        -------
        table : `astropy.table.Table`
            Concatenated table with the same columns as each input table and
            the rows of all of them.
        """
        handles = tuple(handles)  # guard against single-pass iterators
        # Ensure that zero length catalogs are not included
        rowcount = tuple(handle.get(component="rowcount") for handle in handles)
        handles = tuple(handle for handle, count in zip(handles, rowcount) if count > 0)

        vstack = cls.from_handles(handles)
        for handle in handles:
            vstack.extend(handle.get())
        return vstack.result


class WriteObjectTableConnections(pipeBase.PipelineTaskConnections,
                                  defaultTemplates={"coaddName": "deep"},
                                  dimensions=("tract", "patch", "skymap")):
    inputCatalogMeas = connectionTypes.Input(
        doc="Catalog of source measurements on the deepCoadd.",
        dimensions=("tract", "patch", "band", "skymap"),
        storageClass="SourceCatalog",
        name="{coaddName}Coadd_meas",
        multiple=True
    )
    inputCatalogForcedSrc = connectionTypes.Input(
        doc="Catalog of forced measurements (shape and position parameters held fixed) on the deepCoadd.",
        dimensions=("tract", "patch", "band", "skymap"),
        storageClass="SourceCatalog",
        name="{coaddName}Coadd_forced_src",
        multiple=True
    )
    inputCatalogPsfsMultiprofit = connectionTypes.Input(
        doc="Catalog of Gaussian mixture model fit parameters for the PSF model at each object centroid.",
        dimensions=("tract", "patch", "band", "skymap"),
        storageClass="ArrowAstropy",
        name="{coaddName}Coadd_psfs_multiprofit",
        multiple=True,
    )
    outputCatalog = connectionTypes.Output(
        doc="A vertical concatenation of the deepCoadd_{ref|meas|forced_src} catalogs, "
            "stored as a DataFrame with a multi-level column index per-patch.",
        dimensions=("tract", "patch", "skymap"),
        storageClass="DataFrame",
        name="{coaddName}Coadd_obj"
    )


class WriteObjectTableConfig(pipeBase.PipelineTaskConfig,
                             pipelineConnections=WriteObjectTableConnections):
    coaddName = pexConfig.Field(
        dtype=str,
        default="deep",
        doc="Name of coadd"
    )


class WriteObjectTableTask(pipeBase.PipelineTask):
    """Write filter-merged object tables as a DataFrame in parquet format.
    """
    _DefaultName = "writeObjectTable"
    ConfigClass = WriteObjectTableConfig

    # Tag of output dataset written by `MergeSourcesTask.write`
    outputDataset = "obj"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        catalogs = defaultdict(dict)
        for dataset, connection in (
            ("meas", "inputCatalogMeas"),
            ("forced_src", "inputCatalogForcedSrc"),
            ("psfs_multiprofit", "inputCatalogPsfsMultiprofit"),
        ):
            for ref, cat in zip(getattr(inputRefs, connection), inputs[connection]):
                catalogs[ref.dataId["band"]][dataset] = cat

        dataId = butlerQC.quantum.dataId
        df = self.run(catalogs=catalogs, tract=dataId["tract"], patch=dataId["patch"])
        outputs = pipeBase.Struct(outputCatalog=df)
        butlerQC.put(outputs, outputRefs)

    def run(self, catalogs, tract, patch):
        """Merge multiple catalogs.

        Parameters
        ----------
        catalogs : `dict`
            Mapping from filter names to dict of catalogs.
        tract : int
            tractId to use for the tractId column.
        patch : str
            patchId to use for the patchId column.

        Returns
        -------
        catalog : `pandas.DataFrame`
            Merged dataframe.

        Raises
        ------
        ValueError
            Raised if any of the catalogs is of an unsupported type.
        """
        dfs = []
        for filt, tableDict in catalogs.items():
            for dataset, table in tableDict.items():
                # Convert afwTable to pandas DataFrame if needed
                if isinstance(table, pd.DataFrame):
                    df = table
                elif isinstance(table, afwTable.SourceCatalog):
                    df = table.asAstropy().to_pandas()
                elif isinstance(table, astropy.table.Table):
                    df = table.to_pandas()
                else:
                    raise ValueError(f"{dataset=} has unsupported {type(table)=}")
                df.set_index("id", drop=True, inplace=True)

                # Sort columns by name, to ensure matching schema among patches
                df = df.reindex(sorted(df.columns), axis=1)
                df = df.assign(tractId=tract, patchId=patch)

                # Make columns a 3-level MultiIndex
                df.columns = pd.MultiIndex.from_tuples([(dataset, filt, c) for c in df.columns],
                                                       names=("dataset", "band", "column"))
                dfs.append(df)

        # We do this dance and not `pd.concat(dfs)` because the pandas
        # concatenation uses infinite memory.
        catalog = functools.reduce(lambda d1, d2: d1.join(d2), dfs)
        return catalog


class WriteSourceTableConnections(pipeBase.PipelineTaskConnections,
                                  defaultTemplates={"catalogType": ""},
                                  dimensions=("instrument", "visit", "detector")):

    catalog = connectionTypes.Input(
        doc="Input full-depth catalog of sources produced by CalibrateTask",
        name="{catalogType}src",
        storageClass="SourceCatalog",
        dimensions=("instrument", "visit", "detector")
    )
    outputCatalog = connectionTypes.Output(
        doc="Catalog of sources, `src` in Astropy/Parquet format.  Columns are unchanged.",
        name="{catalogType}source",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit", "detector")
    )


class WriteSourceTableConfig(pipeBase.PipelineTaskConfig,
                             pipelineConnections=WriteSourceTableConnections):
    pass


class WriteSourceTableTask(pipeBase.PipelineTask):
    """Write source table to DataFrame Parquet format.
    """
    _DefaultName = "writeSourceTable"
    ConfigClass = WriteSourceTableConfig

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        inputs["visit"] = butlerQC.quantum.dataId["visit"]
        inputs["detector"] = butlerQC.quantum.dataId["detector"]
        result = self.run(**inputs)
        outputs = pipeBase.Struct(outputCatalog=result.table)
        butlerQC.put(outputs, outputRefs)

    def run(self, catalog, visit, detector, **kwargs):
        """Convert `src` catalog to an Astropy table.

        Parameters
        ----------
        catalog: `afwTable.SourceCatalog`
            catalog to be converted
        visit, detector: `int`
            Visit and detector ids to be added as columns.
        **kwargs
            Additional keyword arguments are ignored as a convenience for
            subclasses that pass the same arguments to several different
            methods.

        Returns
        -------
        result : `~lsst.pipe.base.Struct`
            ``table``
                `astropy.table.Table` version of the input catalog
        """
        self.log.info("Generating DataFrame from src catalog visit,detector=%i,%i", visit, detector)
        tbl = catalog.asAstropy()
        tbl["visit"] = visit
        # int16 instead of uint8 because databases don't like unsigned bytes.
        tbl["detector"] = np.int16(detector)

        return pipeBase.Struct(table=tbl)


class WriteRecalibratedSourceTableConnections(WriteSourceTableConnections,
                                              defaultTemplates={"catalogType": ""},
                                              dimensions=("instrument", "visit", "detector", "skymap")):
    visitSummary = connectionTypes.Input(
        doc="Input visit-summary catalog with updated calibration objects.",
        name="finalVisitSummary",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit",),
    )

    def __init__(self, config):
        # We don't want the input catalog here to be an initial existence
        # constraint in QG generation, because that can unfortunately limit the
        # set of data IDs of inputs to other tasks, even those that run earlier
        # (e.g. updateVisitSummary), when the input 'src' catalog is not
        # produced.  It's safer to just use 'visitSummary' existence as an
        # initial constraint, and then let the graph prune out the detectors
        # that don't have a 'src' for this task only.
        self.catalog = dataclasses.replace(self.catalog, deferGraphConstraint=True)


class WriteRecalibratedSourceTableConfig(WriteSourceTableConfig,
                                         pipelineConnections=WriteRecalibratedSourceTableConnections):

    doReevaluatePhotoCalib = pexConfig.Field(
        dtype=bool,
        default=True,
        doc=("Add or replace local photoCalib columns"),
    )
    doReevaluateSkyWcs = pexConfig.Field(
        dtype=bool,
        default=True,
        doc=("Add or replace local WCS columns and update the coord columns, coord_ra and coord_dec"),
    )


class WriteRecalibratedSourceTableTask(WriteSourceTableTask):
    """Write source table to DataFrame Parquet format.
    """
    _DefaultName = "writeRecalibratedSourceTable"
    ConfigClass = WriteRecalibratedSourceTableConfig

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        inputs["visit"] = butlerQC.quantum.dataId["visit"]
        inputs["detector"] = butlerQC.quantum.dataId["detector"]

        if self.config.doReevaluatePhotoCalib or self.config.doReevaluateSkyWcs:
            exposure = ExposureF()
            inputs["exposure"] = self.prepareCalibratedExposure(
                exposure=exposure,
                visitSummary=inputs["visitSummary"],
                detectorId=butlerQC.quantum.dataId["detector"]
            )
            inputs["catalog"] = self.addCalibColumns(**inputs)

        result = self.run(**inputs)
        outputs = pipeBase.Struct(outputCatalog=result.table)
        butlerQC.put(outputs, outputRefs)

    def prepareCalibratedExposure(self, exposure, detectorId, visitSummary=None):
        """Prepare a calibrated exposure and apply external calibrations
        if so configured.

        Parameters
        ----------
        exposure : `lsst.afw.image.exposure.Exposure`
            Input exposure to adjust calibrations. May be an empty Exposure.
        detectorId : `int`
            Detector ID associated with the exposure.
        visitSummary : `lsst.afw.table.ExposureCatalog`, optional
            Exposure catalog with all calibration objects.  WCS and PhotoCalib
            are always applied if ``visitSummary`` is provided and those
            components are not `None`.

        Returns
        -------
        exposure : `lsst.afw.image.exposure.Exposure`
            Exposure with adjusted calibrations.
        """
        if visitSummary is not None:
            row = visitSummary.find(detectorId)
            if row is None:
                raise pipeBase.NoWorkFound(f"Visit summary for detector {detectorId} is missing.")
            if (photoCalib := row.getPhotoCalib()) is None:
                self.log.warning("Detector id %s has None for photoCalib in visit summary; "
                                 "skipping reevaluation of photoCalib.", detectorId)
                exposure.setPhotoCalib(None)
            else:
                exposure.setPhotoCalib(photoCalib)
            if (skyWcs := row.getWcs()) is None:
                self.log.warning("Detector id %s has None for skyWcs in visit summary; "
                                 "skipping reevaluation of skyWcs.", detectorId)
                exposure.setWcs(None)
            else:
                exposure.setWcs(skyWcs)

        return exposure

    def addCalibColumns(self, catalog, exposure, **kwargs):
        """Add replace columns with calibs evaluated at each centroid

        Add or replace 'base_LocalWcs' and 'base_LocalPhotoCalib' columns in
        a source catalog, by rerunning the plugins.

        Parameters
        ----------
        catalog : `lsst.afw.table.SourceCatalog`
            catalog to which calib columns will be added
        exposure : `lsst.afw.image.exposure.Exposure`
            Exposure with attached PhotoCalibs and SkyWcs attributes to be
            reevaluated at local centroids. Pixels are not required.
        **kwargs
            Additional keyword arguments are ignored to facilitate passing the
            same arguments to several methods.

        Returns
        -------
        newCat:  `lsst.afw.table.SourceCatalog`
            Source Catalog with requested local calib columns
        """
        measureConfig = SingleFrameMeasurementTask.ConfigClass()
        measureConfig.doReplaceWithNoise = False

        # Clear all slots, because we aren't running the relevant plugins.
        for slot in measureConfig.slots:
            setattr(measureConfig.slots, slot, None)

        measureConfig.plugins.names = []
        if self.config.doReevaluateSkyWcs:
            measureConfig.plugins.names.add("base_LocalWcs")
            self.log.info("Re-evaluating base_LocalWcs plugin")
        if self.config.doReevaluatePhotoCalib:
            measureConfig.plugins.names.add("base_LocalPhotoCalib")
            self.log.info("Re-evaluating base_LocalPhotoCalib plugin")
        pluginsNotToCopy = tuple(measureConfig.plugins.names)

        # Create a new schema and catalog
        # Copy all columns from original except for the ones to reevaluate
        aliasMap = catalog.schema.getAliasMap()
        mapper = afwTable.SchemaMapper(catalog.schema)
        for item in catalog.schema:
            if not item.field.getName().startswith(pluginsNotToCopy):
                mapper.addMapping(item.key)

        schema = mapper.getOutputSchema()
        measurement = SingleFrameMeasurementTask(config=measureConfig, schema=schema)
        schema.setAliasMap(aliasMap)
        newCat = afwTable.SourceCatalog(schema)
        newCat.extend(catalog, mapper=mapper)

        # Fluxes in sourceCatalogs are in counts, so there are no fluxes to
        # update here. LocalPhotoCalibs are applied during transform tasks.
        # Update coord_ra/coord_dec, which are expected to be positions on the
        # sky and are used as such in sdm tables without transform
        if self.config.doReevaluateSkyWcs and exposure.wcs is not None:
            afwTable.updateSourceCoords(exposure.wcs, newCat)
            wcsPlugin = measurement.plugins["base_LocalWcs"]
        else:
            wcsPlugin = None

        if self.config.doReevaluatePhotoCalib and exposure.getPhotoCalib() is not None:
            pcPlugin = measurement.plugins["base_LocalPhotoCalib"]
        else:
            pcPlugin = None

        for row in newCat:
            if wcsPlugin is not None:
                wcsPlugin.measure(row, exposure)
            if pcPlugin is not None:
                pcPlugin.measure(row, exposure)

        return newCat


class PostprocessAnalysis(object):
    """Calculate columns from DataFrames or handles storing DataFrames.

    This object manages and organizes an arbitrary set of computations
    on a catalog.  The catalog is defined by a
    `DeferredDatasetHandle` or `InMemoryDatasetHandle` object
    (or list thereof), such as a ``deepCoadd_obj`` dataset, and the
    computations are defined by a collection of
    `~lsst.pipe.tasks.functors.Functor` objects (or, equivalently, a
    ``CompositeFunctor``).

    After the object is initialized, accessing the ``.df`` attribute (which
    holds the `pandas.DataFrame` containing the results of the calculations)
    triggers computation of said dataframe.

    One of the conveniences of using this object is the ability to define a
    desired common filter for all functors.  This enables the same functor
    collection to be passed to several different `PostprocessAnalysis` objects
    without having to change the original functor collection, since the ``filt``
    keyword argument of this object triggers an overwrite of the ``filt``
    property for all functors in the collection.

    This object also allows a list of refFlags to be passed, and defines a set
    of default refFlags that are always included even if not requested.

    If a list of DataFrames or Handles is passed, rather than a single one,
    then the calculations will be mapped over all the input catalogs.  In
    principle, it should be straightforward to parallelize this activity, but
    initial tests have failed (see TODO in code comments).

    Parameters
    ----------
    handles : `~lsst.daf.butler.DeferredDatasetHandle` or
              `~lsst.pipe.base.InMemoryDatasetHandle` or
              list of these.
        Source catalog(s) for computation.
    functors : `list`, `dict`, or `~lsst.pipe.tasks.functors.CompositeFunctor`
        Computations to do (functors that act on ``handles``).
        If a dict, the output
        DataFrame will have columns keyed accordingly.
        If a list, the column keys will come from the
        ``.shortname`` attribute of each functor.

    filt : `str`, optional
        Filter in which to calculate.  If provided,
        this will overwrite any existing ``.filt`` attribute
        of the provided functors.

    flags : `list`, optional
        List of flags (per-band) to include in output table.
        Taken from the ``meas`` dataset if applied to a multilevel Object Table.

    refFlags : `list`, optional
        List of refFlags (only reference band) to include in output table.

    forcedFlags : `list`, optional
        List of flags (per-band) to include in output table.
        Taken from the ``forced_src`` dataset if applied to a
        multilevel Object Table. Intended for flags from measurement plugins
        only run during multi-band forced-photometry.
    """
    _defaultRefFlags = []
    _defaultFuncs = ()

    def __init__(self, handles, functors, filt=None, flags=None, refFlags=None, forcedFlags=None):
        self.handles = handles
        self.functors = functors

        self.filt = filt
        self.flags = list(flags) if flags is not None else []
        self.forcedFlags = list(forcedFlags) if forcedFlags is not None else []
        self.refFlags = list(self._defaultRefFlags)
        if refFlags is not None:
            self.refFlags += list(refFlags)

        self._df = None

    @property
    def defaultFuncs(self):
        funcs = dict(self._defaultFuncs)
        return funcs

    @property
    def func(self):
        additionalFuncs = self.defaultFuncs
        additionalFuncs.update({flag: Column(flag, dataset="forced_src") for flag in self.forcedFlags})
        additionalFuncs.update({flag: Column(flag, dataset="ref") for flag in self.refFlags})
        additionalFuncs.update({flag: Column(flag, dataset="meas") for flag in self.flags})

        if isinstance(self.functors, CompositeFunctor):
            func = self.functors
        else:
            func = CompositeFunctor(self.functors)

        func.funcDict.update(additionalFuncs)
        func.filt = self.filt

        return func

    @property
    def noDupCols(self):
        return [name for name, func in self.func.funcDict.items() if func.noDup]

    @property
    def df(self):
        if self._df is None:
            self.compute()
        return self._df

    def compute(self, dropna=False, pool=None):
        # map over multiple handles
        if type(self.handles) in (list, tuple):
            if pool is None:
                dflist = [self.func(handle, dropna=dropna) for handle in self.handles]
            else:
                # TODO: Figure out why this doesn't work (pyarrow pickling
                # issues?)
                dflist = pool.map(functools.partial(self.func, dropna=dropna), self.handles)
            self._df = pd.concat(dflist)
        else:
            self._df = self.func(self.handles, dropna=dropna)

        return self._df


class TransformCatalogBaseConnections(pipeBase.PipelineTaskConnections,
                                      dimensions=()):
    """Expected Connections for subclasses of TransformCatalogBaseTask.

    Must be subclassed.
    """
    inputCatalog = connectionTypes.Input(
        name="",
        storageClass="DataFrame",
    )
    outputCatalog = connectionTypes.Output(
        name="",
        storageClass="ArrowAstropy",
    )


class TransformCatalogBaseConfig(pipeBase.PipelineTaskConfig,
                                 pipelineConnections=TransformCatalogBaseConnections):
    functorFile = pexConfig.Field(
        dtype=str,
        doc="Path to YAML file specifying Science Data Model functors to use "
            "when copying columns and computing calibrated values.",
        default=None,
        optional=True
    )
    primaryKey = pexConfig.Field(
        dtype=str,
        doc="Name of column to be set as the DataFrame index. If None, the index"
            "will be named `id`",
        default=None,
        optional=True
    )
    columnsFromDataId = pexConfig.ListField(
        dtype=str,
        default=None,
        optional=True,
        doc="Columns to extract from the dataId",
    )


class TransformCatalogBaseTask(pipeBase.PipelineTask):
    """Base class for transforming/standardizing a catalog by applying functors
    that convert units and apply calibrations.

    The purpose of this task is to perform a set of computations on an input
    ``DeferredDatasetHandle`` or ``InMemoryDatasetHandle`` that holds a
    ``DataFrame`` dataset (such as ``deepCoadd_obj``), and write the results to
    a new dataset (which needs to be declared in an ``outputDataset``
    attribute).

    The calculations to be performed are defined in a YAML file that specifies
    a set of functors to be computed, provided as a ``--functorFile`` config
    parameter. An example of such a YAML file is the following:

        funcs:
            sourceId:
                functor: Index
            x:
                functor: Column
                args: slot_Centroid_x
            y:
                functor: Column
                args: slot_Centroid_y
            psfFlux:
                functor: LocalNanojansky
                args:
                    - slot_PsfFlux_instFlux
                    - slot_PsfFlux_instFluxErr
                    - base_LocalPhotoCalib
                    - base_LocalPhotoCalibErr
            psfFluxErr:
                functor: LocalNanojanskyErr
                args:
                    - slot_PsfFlux_instFlux
                    - slot_PsfFlux_instFluxErr
                    - base_LocalPhotoCalib
                    - base_LocalPhotoCalibErr
        flags:
            - detect_isPrimary

    The names for each entry under "func" will become the names of columns in
    the output dataset.  All the functors referenced are defined in
    `~lsst.pipe.tasks.functors`.  Positional arguments to be passed to each
    functor are in the `args` list, and any additional entries for each column
    other than "functor" or "args" (e.g., ``'filt'``, ``'dataset'``) are
    treated as keyword arguments to be passed to the functor initialization.

    The "flags" entry is the default shortcut for `Column` functors.
    All columns listed under "flags" will be copied to the output table
    untransformed. They can be of any datatype.
    In the special case of transforming a multi-level oject table with
    band and dataset indices (deepCoadd_obj), these will be taked from the
    ``meas`` dataset and exploded out per band.

    There are two special shortcuts that only apply when transforming
    multi-level Object (deepCoadd_obj) tables:
     -  The "refFlags" entry is shortcut for `Column` functor
        taken from the ``ref`` dataset if transforming an ObjectTable.
     -  The "forcedFlags" entry is shortcut for `Column` functors.
        taken from the ``forced_src`` dataset if transforming an ObjectTable.
        These are expanded out per band.


    This task uses the `lsst.pipe.tasks.postprocess.PostprocessAnalysis` object
    to organize and excecute the calculations.
    """
    @property
    def _DefaultName(self):
        raise NotImplementedError("Subclass must define the \"_DefaultName\" attribute.")

    @property
    def outputDataset(self):
        raise NotImplementedError("Subclass must define the \"outputDataset\" attribute.")

    @property
    def inputDataset(self):
        raise NotImplementedError("Subclass must define \"inputDataset\" attribute.")

    @property
    def ConfigClass(self):
        raise NotImplementedError("Subclass must define \"ConfigClass\" attribute.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.functorFile:
            self.log.info("Loading tranform functor definitions from %s",
                          self.config.functorFile)
            self.funcs = CompositeFunctor.from_file(self.config.functorFile)
            self.funcs.update(dict(PostprocessAnalysis._defaultFuncs))
        else:
            self.funcs = None

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        if self.funcs is None:
            raise ValueError("config.functorFile is None. "
                             "Must be a valid path to yaml in order to run Task as a PipelineTask.")
        result = self.run(handle=inputs["inputCatalog"], funcs=self.funcs,
                          dataId=dict(outputRefs.outputCatalog.dataId.mapping))
        butlerQC.put(result, outputRefs)

    def run(self, handle, funcs=None, dataId=None, band=None):
        """Do postprocessing calculations

        Takes a ``DeferredDatasetHandle`` or ``InMemoryDatasetHandle`` or
        ``DataFrame`` object and dataId,
        returns a dataframe with results of postprocessing calculations.

        Parameters
        ----------
        handles : `~lsst.daf.butler.DeferredDatasetHandle` or
                  `~lsst.pipe.base.InMemoryDatasetHandle` or
                  `~pandas.DataFrame`, or list of these.
            DataFrames from which calculations are done.
        funcs : `~lsst.pipe.tasks.functors.Functor`
            Functors to apply to the table's columns
        dataId : dict, optional
            Used to add a `patchId` column to the output dataframe.
        band : `str`, optional
            Filter band that is being processed.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Result struct, with a single ``outputCatalog`` attribute holding
            the transformed catalog.
        """
        self.log.info("Transforming/standardizing the source table dataId: %s", dataId)

        df = self.transform(band, handle, funcs, dataId).df
        self.log.info("Made a table of %d columns and %d rows", len(df.columns), len(df))
        result = pipeBase.Struct(outputCatalog=pandas_to_astropy(df))
        return result

    def getFunctors(self):
        return self.funcs

    def getAnalysis(self, handles, funcs=None, band=None):
        if funcs is None:
            funcs = self.funcs
        analysis = PostprocessAnalysis(handles, funcs, filt=band)
        return analysis

    def transform(self, band, handles, funcs, dataId):
        analysis = self.getAnalysis(handles, funcs=funcs, band=band)
        df = analysis.df
        if dataId and self.config.columnsFromDataId:
            for key in self.config.columnsFromDataId:
                if key in dataId:
                    if key == "detector":
                        # int16 instead of uint8 because databases don't like unsigned bytes.
                        df[key] = np.int16(dataId[key])
                    else:
                        df[key] = dataId[key]
                else:
                    raise ValueError(f"'{key}' in config.columnsFromDataId not found in dataId: {dataId}")

        if self.config.primaryKey:
            if df.index.name != self.config.primaryKey and self.config.primaryKey in df:
                df.reset_index(inplace=True, drop=True)
                df.set_index(self.config.primaryKey, inplace=True)

        return pipeBase.Struct(
            df=df,
            analysis=analysis
        )


class TransformObjectCatalogConnections(pipeBase.PipelineTaskConnections,
                                        defaultTemplates={"coaddName": "deep"},
                                        dimensions=("tract", "patch", "skymap")):
    inputCatalog = connectionTypes.Input(
        doc="The vertical concatenation of the {coaddName}_{meas|forced_src|psfs_multiprofit} catalogs, "
            "stored as a DataFrame with a multi-level column index per-patch.",
        dimensions=("tract", "patch", "skymap"),
        storageClass="DataFrame",
        name="{coaddName}Coadd_obj",
        deferLoad=True,
    )
    inputCatalogRef = connectionTypes.Input(
        doc="Catalog marking the primary detection (which band provides a good shape and position)"
            "for each detection in deepCoadd_mergeDet.",
        dimensions=("tract", "patch", "skymap"),
        storageClass="SourceCatalog",
        name="{coaddName}Coadd_ref",
        deferLoad=True,
    )
    inputCatalogSersicMultiprofit = connectionTypes.Input(
        doc="Catalog of source measurements on the deepCoadd.",
        dimensions=("tract", "patch", "skymap"),
        storageClass="ArrowAstropy",
        name="{coaddName}Coadd_Sersic_multiprofit",
        deferLoad=True,
    )
    inputCatalogEpoch = connectionTypes.Input(
        doc="Catalog of mean epochs for each object per band.",
        dimensions=("tract", "patch", "skymap"),
        storageClass="ArrowAstropy",
        name="object_epoch",
        deferLoad=True,
    )
    outputCatalog = connectionTypes.Output(
        doc="Per-Patch Object Table of columns transformed from the deepCoadd_obj table per the standard "
            "data model.",
        dimensions=("tract", "patch", "skymap"),
        storageClass="ArrowAstropy",
        name="objectTable"
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if config.multilevelOutput:
            self.outputCatalog = dataclasses.replace(self.outputCatalog, storageClass="DataFrame")


class TransformObjectCatalogConfig(TransformCatalogBaseConfig,
                                   pipelineConnections=TransformObjectCatalogConnections):
    coaddName = pexConfig.Field(
        dtype=str,
        default="deep",
        doc="Name of coadd"
    )
    outputBands = pexConfig.ListField(
        dtype=str,
        default=None,
        optional=True,
        doc=("These bands and only these bands will appear in the output,"
             " NaN-filled if the input does not include them."
             " If None, then use all bands found in the input.")
    )
    camelCase = pexConfig.Field(
        dtype=bool,
        default=False,
        doc=("Write per-band columns names with camelCase, else underscore "
             "For example: gPsFlux instead of g_PsFlux.")
    )
    multilevelOutput = pexConfig.Field(
        dtype=bool,
        default=False,
        doc=("Whether results dataframe should have a multilevel column index (True) or be flat "
             "and name-munged (False).  If True, the output storage class will be "
             "set to DataFrame, since astropy tables do not support multi-level indexing."),
        deprecated="Support for multi-level outputs is deprecated and will be removed after v29.",
    )
    goodFlags = pexConfig.ListField(
        dtype=str,
        default=[],
        doc=("List of 'good' flags that should be set False when populating empty tables. "
             "All other flags are considered to be 'bad' flags and will be set to True.")
    )
    floatFillValue = pexConfig.Field(
        dtype=float,
        default=np.nan,
        doc="Fill value for float fields when populating empty tables."
    )
    integerFillValue = pexConfig.Field(
        dtype=int,
        default=-1,
        doc="Fill value for integer fields when populating empty tables."
    )

    def setDefaults(self):
        super().setDefaults()
        self.functorFile = os.path.join("$PIPE_TASKS_DIR", "schemas", "Object.yaml")
        self.primaryKey = "objectId"
        self.columnsFromDataId = ["tract", "patch"]
        self.goodFlags = ["calib_astrometry_used",
                          "calib_photometry_reserved",
                          "calib_photometry_used",
                          "calib_psf_candidate",
                          "calib_psf_reserved",
                          "calib_psf_used"]


class TransformObjectCatalogTask(TransformCatalogBaseTask):
    """Produce a flattened Object Table to match the format specified in
    sdm_schemas.

    Do the same set of postprocessing calculations on all bands.

    This is identical to `TransformCatalogBaseTask`, except for that it does
    the specified functor calculations for all filters present in the
    input `deepCoadd_obj` table.  Any specific ``"filt"`` keywords specified
    by the YAML file will be superceded.
    """
    _DefaultName = "transformObjectCatalog"
    ConfigClass = TransformObjectCatalogConfig

    datasets_multiband = ("epoch", "ref", "Sersic_multiprofit")

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        if self.funcs is None:
            raise ValueError("config.functorFile is None. "
                             "Must be a valid path to yaml in order to run Task as a PipelineTask.")
        result = self.run(handle=inputs["inputCatalog"], funcs=self.funcs,
                          dataId=dict(outputRefs.outputCatalog.dataId.mapping),
                          handle_epoch=inputs["inputCatalogEpoch"],
                          handle_ref=inputs["inputCatalogRef"],
                          handle_Sersic_multiprofit=inputs["inputCatalogSersicMultiprofit"],
                          )
        butlerQC.put(result, outputRefs)

    def run(self, handle, funcs=None, dataId=None, band=None, **kwargs):
        # NOTE: band kwarg is ignored here.
        # TODO: Document and improve funcs argument usage in DM-48895
        # self.getAnalysis only supports list, dict and CompositeFunctor
        if isinstance(funcs, CompositeFunctor):
            funcDict_in = funcs.funcDict
        elif isinstance(funcs, dict):
            funcDict_in = funcs
        elif isinstance(funcs, list):
            funcDict_in = {idx: v for idx, v in enumerate(funcs)}
        else:
            raise TypeError(f"Unsupported {type(funcs)=}")

        handles_multi = {}
        funcDicts_multiband = {}
        for dataset in self.datasets_multiband:
            if (handle_multi := kwargs.get(f"handle_{dataset}")) is None:
                raise RuntimeError(f"Missing required handle_{dataset} kwarg")
            handles_multi[dataset] = handle_multi
            funcDicts_multiband[dataset] = {}

        dfDict = {}
        analysisDict = {}
        templateDf = pd.DataFrame()

        columns = handle.get(component="columns")
        inputBands = columns.unique(level=1).values

        outputBands = self.config.outputBands if self.config.outputBands else inputBands

        # Split up funcs for per-band and multiband tables
        funcDict_band = {}

        for name, func in funcDict_in.items():
            if func.dataset in funcDicts_multiband:
                # This is something like a MultibandColumn
                if band := getattr(func, "band_to_check", None):
                    if band not in outputBands:
                        continue
                # This is something like a ReferenceBand that has configurable bands
                elif hasattr(func, "bands"):
                    # TODO: Determine if this can be avoided DM-48895
                    # This will work fine if the init doesn't manipulate bands
                    # If it does, then one would need to make a new functor
                    # Determining the (kw)args is tricky in that case
                    func.bands = tuple(inputBands)

            funcDict = funcDicts_multiband.get(func.dataset, funcDict_band)
            funcDict[name] = func

        funcs_band = CompositeFunctor(funcDict_band)

        # Perform transform for data of filters that exist in the handle dataframe.
        for inputBand in inputBands:
            if inputBand not in outputBands:
                self.log.info("Ignoring %s band data in the input", inputBand)
                continue
            self.log.info("Transforming the catalog of band %s", inputBand)
            result = self.transform(inputBand, handle, funcs_band, dataId)
            dfDict[inputBand] = result.df
            analysisDict[inputBand] = result.analysis
            if templateDf.empty:
                templateDf = result.df

        # Put filler values in columns of other wanted bands
        for filt in outputBands:
            if filt not in dfDict:
                self.log.info("Adding empty columns for band %s", filt)
                dfTemp = templateDf.copy()
                for col in dfTemp.columns:
                    testValue = dfTemp[col].values[0]
                    if isinstance(testValue, (np.bool_, pd.BooleanDtype)):
                        # Boolean flag type, check if it is a "good" flag
                        if col in self.config.goodFlags:
                            fillValue = False
                        else:
                            fillValue = True
                    elif isinstance(testValue, numbers.Integral):
                        # Checking numbers.Integral catches all flavors
                        # of python, numpy, pandas, etc. integers.
                        # We must ensure this is not an unsigned integer.
                        if isinstance(testValue, np.unsignedinteger):
                            raise ValueError("Parquet tables may not have unsigned integer columns.")
                        else:
                            fillValue = self.config.integerFillValue
                    else:
                        fillValue = self.config.floatFillValue
                    dfTemp[col].values[:] = fillValue
                dfDict[filt] = dfTemp

        # This makes a multilevel column index, with band as first level
        df = pd.concat(dfDict, axis=1, names=["band", "column"])
        name_index = df.index.name

        # TODO: Remove in DM-48895
        if not self.config.multilevelOutput:
            noDupCols = list(set.union(*[set(v.noDupCols) for v in analysisDict.values()]))
            if self.config.primaryKey in noDupCols:
                noDupCols.remove(self.config.primaryKey)
            if dataId and self.config.columnsFromDataId:
                noDupCols += self.config.columnsFromDataId
            df = flattenFilters(df, noDupCols=noDupCols, camelCase=self.config.camelCase,
                                inputBands=inputBands)

        # Apply per-dataset functors to each multiband dataset in turn
        for dataset, funcDict in funcDicts_multiband.items():
            handle_multiband = handles_multi[dataset]
            df_dataset = handle_multiband.get()
            if isinstance(df_dataset, astropy.table.Table):
                # Allow astropy table inputs to already have the output index
                if name_index not in df_dataset.colnames:
                    if self.config.primaryKey in df_dataset.colnames:
                        name_index_ap = self.config.primaryKey
                    else:
                        raise RuntimeError(
                            f"Neither of {name_index=} nor {self.config.primaryKey=} appear in"
                            f" {df_dataset.colnames=} for {dataset=}"
                        )
                else:
                    name_index_ap = name_index
                df_dataset = df_dataset.to_pandas().set_index(name_index_ap, drop=False)
            elif isinstance(df_dataset, afwTable.SourceCatalog):
                df_dataset = df_dataset.asAstropy().to_pandas().set_index(name_index, drop=False)
            # TODO: should funcDict have noDup funcs removed?
            # noDup was intended for per-band tables.
            result = self.transform(
                None,
                pipeBase.InMemoryDatasetHandle(df_dataset, storageClass="DataFrame"),
                CompositeFunctor(funcDict),
                dataId,
            )
            result.df.index.name = name_index
            # Drop columns from dataId if present (patch, tract)
            if self.config.columnsFromDataId:
                columns_drop = [column for column in self.config.columnsFromDataId if column in result.df]
                if columns_drop:
                    result.df.drop(columns_drop, axis=1, inplace=True)
            # Make the same multi-index for the multiband table if needed
            # This might end up making copies, one of several reasons to avoid
            # using multilevel indexes, or DataFrames at all
            to_concat = pd.concat(
                {band: result.df for band in self.config.outputBands}, axis=1, names=["band", "column"]
            ) if self.config.multilevelOutput else result.df
            df = pd.concat([df, to_concat], axis=1)
            analysisDict[dataset] = result.analysis
            del result

        df.index.name = self.config.primaryKey

        if not self.config.multilevelOutput:
            tbl = pandas_to_astropy(df)
        else:
            tbl = df

        self.log.info("Made a table of %d columns and %d rows", len(tbl.columns), len(tbl))

        return pipeBase.Struct(outputCatalog=tbl)


class ConsolidateObjectTableConnections(pipeBase.PipelineTaskConnections,
                                        dimensions=("tract", "skymap")):
    inputCatalogs = connectionTypes.Input(
        doc="Per-Patch objectTables conforming to the standard data model.",
        name="objectTable",
        storageClass="ArrowAstropy",
        dimensions=("tract", "patch", "skymap"),
        multiple=True,
        deferLoad=True,
    )
    outputCatalog = connectionTypes.Output(
        doc="Pre-tract horizontal concatenation of the input objectTables",
        name="objectTable_tract",
        storageClass="ArrowAstropy",
        dimensions=("tract", "skymap"),
    )


class ConsolidateObjectTableConfig(pipeBase.PipelineTaskConfig,
                                   pipelineConnections=ConsolidateObjectTableConnections):
    coaddName = pexConfig.Field(
        dtype=str,
        default="deep",
        doc="Name of coadd"
    )


class ConsolidateObjectTableTask(pipeBase.PipelineTask):
    """Write patch-merged source tables to a tract-level DataFrame Parquet file.

    Concatenates `objectTable` list into a per-visit `objectTable_tract`.
    """
    _DefaultName = "consolidateObjectTable"
    ConfigClass = ConsolidateObjectTableConfig

    inputDataset = "objectTable"
    outputDataset = "objectTable_tract"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        self.log.info("Concatenating %s per-patch Object Tables",
                      len(inputs["inputCatalogs"]))
        table = TableVStack.vstack_handles(inputs["inputCatalogs"])
        butlerQC.put(pipeBase.Struct(outputCatalog=table), outputRefs)


class TransformSourceTableConnections(pipeBase.PipelineTaskConnections,
                                      defaultTemplates={"catalogType": ""},
                                      dimensions=("instrument", "visit", "detector")):

    inputCatalog = connectionTypes.Input(
        doc="Wide input catalog of sources produced by WriteSourceTableTask",
        name="{catalogType}source",
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "detector"),
        deferLoad=True
    )
    outputCatalog = connectionTypes.Output(
        doc="Narrower, per-detector Source Table transformed and converted per a "
            "specified set of functors",
        name="{catalogType}sourceTable",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit", "detector")
    )


class TransformSourceTableConfig(TransformCatalogBaseConfig,
                                 pipelineConnections=TransformSourceTableConnections):

    def setDefaults(self):
        super().setDefaults()
        self.functorFile = os.path.join("$PIPE_TASKS_DIR", "schemas", "Source.yaml")
        self.primaryKey = "sourceId"
        self.columnsFromDataId = ["visit", "detector", "band", "physical_filter"]


class TransformSourceTableTask(TransformCatalogBaseTask):
    """Transform/standardize a source catalog
    """
    _DefaultName = "transformSourceTable"
    ConfigClass = TransformSourceTableConfig


class ConsolidateVisitSummaryConnections(pipeBase.PipelineTaskConnections,
                                         dimensions=("instrument", "visit",),
                                         defaultTemplates={"calexpType": ""}):
    calexp = connectionTypes.Input(
        doc="Processed exposures used for metadata",
        name="calexp",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
        deferLoad=True,
        multiple=True,
    )
    visitSummary = connectionTypes.Output(
        doc=("Per-visit consolidated exposure metadata.  These catalogs use "
             "detector id for the id and are sorted for fast lookups of a "
             "detector."),
        name="visitSummary",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit"),
    )
    visitSummarySchema = connectionTypes.InitOutput(
        doc="Schema of the visitSummary catalog",
        name="visitSummary_schema",
        storageClass="ExposureCatalog",
    )


class ConsolidateVisitSummaryConfig(pipeBase.PipelineTaskConfig,
                                    pipelineConnections=ConsolidateVisitSummaryConnections):
    """Config for ConsolidateVisitSummaryTask"""
    pass


class ConsolidateVisitSummaryTask(pipeBase.PipelineTask):
    """Task to consolidate per-detector visit metadata.

    This task aggregates the following metadata from all the detectors in a
    single visit into an exposure catalog:
    - The visitInfo.
    - The wcs.
    - The photoCalib.
    - The physical_filter and band (if available).
    - The psf size, shape, and effective area at the center of the detector.
    - The corners of the bounding box in right ascension/declination.

    Other quantities such as Detector, Psf, ApCorrMap, and TransmissionCurve
    are not persisted here because of storage concerns, and because of their
    limited utility as summary statistics.

    Tests for this task are performed in ci_hsc_gen3.
    """
    _DefaultName = "consolidateVisitSummary"
    ConfigClass = ConsolidateVisitSummaryConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.schema = afwTable.ExposureTable.makeMinimalSchema()
        self.schema.addField("visit", type="L", doc="Visit number")
        self.schema.addField("physical_filter", type="String", size=32, doc="Physical filter")
        self.schema.addField("band", type="String", size=32, doc="Name of band")
        ExposureSummaryStats.update_schema(self.schema)
        self.visitSummarySchema = afwTable.ExposureCatalog(self.schema)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        dataRefs = butlerQC.get(inputRefs.calexp)
        visit = dataRefs[0].dataId["visit"]

        self.log.debug("Concatenating metadata from %d per-detector calexps (visit %d)",
                       len(dataRefs), visit)

        expCatalog = self._combineExposureMetadata(visit, dataRefs)

        butlerQC.put(expCatalog, outputRefs.visitSummary)

    def _combineExposureMetadata(self, visit, dataRefs):
        """Make a combined exposure catalog from a list of dataRefs.
        These dataRefs must point to exposures with wcs, summaryStats,
        and other visit metadata.

        Parameters
        ----------
        visit : `int`
            Visit identification number.
        dataRefs : `list` of `lsst.daf.butler.DeferredDatasetHandle`
            List of dataRefs in visit.

        Returns
        -------
        visitSummary : `lsst.afw.table.ExposureCatalog`
            Exposure catalog with per-detector summary information.
        """
        cat = afwTable.ExposureCatalog(self.schema)
        cat.resize(len(dataRefs))

        cat["visit"] = visit

        for i, dataRef in enumerate(dataRefs):
            visitInfo = dataRef.get(component="visitInfo")
            filterLabel = dataRef.get(component="filter")
            summaryStats = dataRef.get(component="summaryStats")
            detector = dataRef.get(component="detector")
            wcs = dataRef.get(component="wcs")
            photoCalib = dataRef.get(component="photoCalib")
            detector = dataRef.get(component="detector")
            bbox = dataRef.get(component="bbox")
            validPolygon = dataRef.get(component="validPolygon")

            rec = cat[i]
            rec.setBBox(bbox)
            rec.setVisitInfo(visitInfo)
            rec.setWcs(wcs)
            rec.setPhotoCalib(photoCalib)
            rec.setValidPolygon(validPolygon)

            rec["physical_filter"] = filterLabel.physicalLabel if filterLabel.hasPhysicalLabel() else ""
            rec["band"] = filterLabel.bandLabel if filterLabel.hasBandLabel() else ""
            rec.setId(detector.getId())
            summaryStats.update_record(rec)

        if not cat:
            raise pipeBase.NoWorkFound(
                "No detectors had sufficient information to make a visit summary row."
            )

        metadata = dafBase.PropertyList()
        metadata.add("COMMENT", "Catalog id is detector id, sorted.")
        # We are looping over existing datarefs, so the following is true
        metadata.add("COMMENT", "Only detectors with data have entries.")
        cat.setMetadata(metadata)

        cat.sort()
        return cat


class ConsolidateSourceTableConnections(pipeBase.PipelineTaskConnections,
                                        defaultTemplates={"catalogType": ""},
                                        dimensions=("instrument", "visit")):
    inputCatalogs = connectionTypes.Input(
        doc="Input per-detector Source Tables",
        name="{catalogType}sourceTable",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit", "detector"),
        multiple=True,
        deferLoad=True,
    )
    outputCatalog = connectionTypes.Output(
        doc="Per-visit concatenation of Source Table",
        name="{catalogType}sourceTable_visit",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit")
    )


class ConsolidateSourceTableConfig(pipeBase.PipelineTaskConfig,
                                   pipelineConnections=ConsolidateSourceTableConnections):
    pass


class ConsolidateSourceTableTask(pipeBase.PipelineTask):
    """Concatenate `sourceTable` list into a per-visit `sourceTable_visit`
    """
    _DefaultName = "consolidateSourceTable"
    ConfigClass = ConsolidateSourceTableConfig

    inputDataset = "sourceTable"
    outputDataset = "sourceTable_visit"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        from .makeWarp import reorderRefs

        detectorOrder = [ref.dataId["detector"] for ref in inputRefs.inputCatalogs]
        detectorOrder.sort()
        inputRefs = reorderRefs(inputRefs, detectorOrder, dataIdKey="detector")
        inputs = butlerQC.get(inputRefs)
        self.log.info("Concatenating %s per-detector Source Tables",
                      len(inputs["inputCatalogs"]))
        table = TableVStack.vstack_handles(inputs["inputCatalogs"])
        butlerQC.put(pipeBase.Struct(outputCatalog=table), outputRefs)


class MakeCcdVisitTableConnections(pipeBase.PipelineTaskConnections,
                                   dimensions=("instrument",),
                                   defaultTemplates={"calexpType": ""}):
    visitSummaryRefs = connectionTypes.Input(
        doc="Data references for per-visit consolidated exposure metadata",
        name="finalVisitSummary",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit"),
        multiple=True,
        deferLoad=True,
    )
    outputCatalog = connectionTypes.Output(
        doc="CCD and Visit metadata table",
        name="ccdVisitTable",
        storageClass="ArrowAstropy",
        dimensions=("instrument",)
    )


class MakeCcdVisitTableConfig(pipeBase.PipelineTaskConfig,
                              pipelineConnections=MakeCcdVisitTableConnections):
    idGenerator = DetectorVisitIdGeneratorConfig.make_field()


class MakeCcdVisitTableTask(pipeBase.PipelineTask):
    """Produce a `ccdVisitTable` from the visit summary exposure catalogs.
    """
    _DefaultName = "makeCcdVisitTable"
    ConfigClass = MakeCcdVisitTableConfig

    def run(self, visitSummaryRefs):
        """Make a table of ccd information from the visit summary catalogs.

        Parameters
        ----------
        visitSummaryRefs : `list` of `lsst.daf.butler.DeferredDatasetHandle`
            List of DeferredDatasetHandles pointing to exposure catalogs with
            per-detector summary information.

        Returns
        -------
        result : `~lsst.pipe.base.Struct`
           Results struct with attribute:

           ``outputCatalog``
               Catalog of ccd and visit information.
        """
        ccdEntries = []
        for visitSummaryRef in visitSummaryRefs:
            visitSummary = visitSummaryRef.get()
            if not visitSummary:
                continue
            visitInfo = visitSummary[0].getVisitInfo()

            # Strip provenance to prevent merge confusion.
            strip_provenance_from_fits_header(visitSummary.metadata)

            ccdEntry = {}
            summaryTable = visitSummary.asAstropy()
            selectColumns = ["id", "visit", "physical_filter", "band", "ra", "dec",
                             "pixelScale", "zenithDistance",
                             "expTime", "zeroPoint", "psfSigma", "skyBg", "skyNoise",
                             "astromOffsetMean", "astromOffsetStd", "nPsfStar",
                             "psfStarDeltaE1Median", "psfStarDeltaE2Median",
                             "psfStarDeltaE1Scatter", "psfStarDeltaE2Scatter",
                             "psfStarDeltaSizeMedian", "psfStarDeltaSizeScatter",
                             "psfStarScaledDeltaSizeScatter", "psfTraceRadiusDelta",
                             "psfApFluxDelta", "psfApCorrSigmaScaledDelta",
                             "maxDistToNearestPsf",
                             "effTime", "effTimePsfSigmaScale",
                             "effTimeSkyBgScale", "effTimeZeroPointScale",
                             "magLim"]
            ccdEntry = summaryTable[selectColumns]
            # 'visit' is the human readable visit number.
            # 'visitId' is the key to the visitId table. They are the same.
            # Technically you should join to get the visit from the visit
            # table.
            ccdEntry.rename_column("visit", "visitId")
            ccdEntry.rename_column("id", "detectorId")

            # RFC-924: Temporarily keep a duplicate "decl" entry for backwards
            # compatibility. To be removed after September 2023.
            ccdEntry["decl"] = ccdEntry["dec"]

            ccdEntry["ccdVisitId"] = [
                self.config.idGenerator.apply(
                    visitSummaryRef.dataId,
                    detector=detector_id,
                    is_exposure=False,
                ).catalog_id  # The "catalog ID" here is the ccdVisit ID
                              # because it's usually the ID for a whole catalog
                              # with a {visit, detector}, and that's the main
                              # use case for IdGenerator.  This usage for a
                              # summary table is rare.
                for detector_id in summaryTable["id"]
            ]
            ccdEntry["detector"] = summaryTable["id"]
            ccdEntry["seeing"] = (
                visitSummary["psfSigma"] * visitSummary["pixelScale"] * np.sqrt(8 * np.log(2))
            )
            ccdEntry["skyRotation"] = visitInfo.getBoresightRotAngle().asDegrees()
            ccdEntry["expMidpt"] = np.datetime64(visitInfo.getDate().toPython(), "ns")
            ccdEntry["expMidptMJD"] = visitInfo.getDate().get(dafBase.DateTime.MJD)
            expTime = visitInfo.getExposureTime()
            ccdEntry["obsStart"] = (
                ccdEntry["expMidpt"] - 0.5 * np.timedelta64(int(expTime * 1E9), "ns")
            )
            expTime_days = expTime / (60*60*24)
            ccdEntry["obsStartMJD"] = ccdEntry["expMidptMJD"] - 0.5 * expTime_days
            ccdEntry["darkTime"] = visitInfo.getDarkTime()
            ccdEntry["xSize"] = summaryTable["bbox_max_x"] - summaryTable["bbox_min_x"]
            ccdEntry["ySize"] = summaryTable["bbox_max_y"] - summaryTable["bbox_min_y"]
            ccdEntry["llcra"] = summaryTable["raCorners"][:, 0]
            ccdEntry["llcdec"] = summaryTable["decCorners"][:, 0]
            ccdEntry["ulcra"] = summaryTable["raCorners"][:, 1]
            ccdEntry["ulcdec"] = summaryTable["decCorners"][:, 1]
            ccdEntry["urcra"] = summaryTable["raCorners"][:, 2]
            ccdEntry["urcdec"] = summaryTable["decCorners"][:, 2]
            ccdEntry["lrcra"] = summaryTable["raCorners"][:, 3]
            ccdEntry["lrcdec"] = summaryTable["decCorners"][:, 3]
            # TODO: DM-30618, Add raftName, nExposures, ccdTemp, binX, binY,
            # and flags, and decide if WCS, and llcx, llcy, ulcx, ulcy, etc.
            # values are actually wanted.
            ccdEntries.append(ccdEntry)

        outputCatalog = astropy.table.vstack(ccdEntries, join_type="exact")
        return pipeBase.Struct(outputCatalog=outputCatalog)


class MakeVisitTableConnections(pipeBase.PipelineTaskConnections,
                                dimensions=("instrument",),
                                defaultTemplates={"calexpType": ""}):
    visitSummaries = connectionTypes.Input(
        doc="Per-visit consolidated exposure metadata",
        name="finalVisitSummary",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit",),
        multiple=True,
        deferLoad=True,
    )
    outputCatalog = connectionTypes.Output(
        doc="Visit metadata table",
        name="visitTable",
        storageClass="ArrowAstropy",
        dimensions=("instrument",)
    )


class MakeVisitTableConfig(pipeBase.PipelineTaskConfig,
                           pipelineConnections=MakeVisitTableConnections):
    pass


class MakeVisitTableTask(pipeBase.PipelineTask):
    """Produce a `visitTable` from the visit summary exposure catalogs.
    """
    _DefaultName = "makeVisitTable"
    ConfigClass = MakeVisitTableConfig

    def run(self, visitSummaries):
        """Make a table of visit information from the visit summary catalogs.

        Parameters
        ----------
        visitSummaries : `list` of `lsst.afw.table.ExposureCatalog`
            List of exposure catalogs with per-detector summary information.
        Returns
        -------
        result : `~lsst.pipe.base.Struct`
            Results struct with attribute:

            ``outputCatalog``
                 Catalog of visit information.
        """
        visitEntries = []
        for visitSummary in visitSummaries:
            visitSummary = visitSummary.get()
            if not visitSummary:
                continue
            visitRow = visitSummary[0]
            visitInfo = visitRow.getVisitInfo()

            visitEntry = {}
            visitEntry["visitId"] = visitRow["visit"]
            visitEntry["visit"] = visitRow["visit"]
            visitEntry["physical_filter"] = visitRow["physical_filter"]
            visitEntry["band"] = visitRow["band"]
            raDec = visitInfo.getBoresightRaDec()
            visitEntry["ra"] = raDec.getRa().asDegrees()
            visitEntry["dec"] = raDec.getDec().asDegrees()

            # RFC-924: Temporarily keep a duplicate "decl" entry for backwards
            # compatibility. To be removed after September 2023.
            visitEntry["decl"] = visitEntry["dec"]

            visitEntry["skyRotation"] = visitInfo.getBoresightRotAngle().asDegrees()
            azAlt = visitInfo.getBoresightAzAlt()
            visitEntry["azimuth"] = azAlt.getLongitude().asDegrees()
            visitEntry["altitude"] = azAlt.getLatitude().asDegrees()
            visitEntry["zenithDistance"] = 90 - azAlt.getLatitude().asDegrees()
            visitEntry["airmass"] = visitInfo.getBoresightAirmass()
            expTime = visitInfo.getExposureTime()
            visitEntry["expTime"] = expTime
            visitEntry["expMidpt"] = np.datetime64(visitInfo.getDate().toPython(), "ns")
            visitEntry["expMidptMJD"] = visitInfo.getDate().get(dafBase.DateTime.MJD)
            visitEntry["obsStart"] = visitEntry["expMidpt"] - 0.5 * np.timedelta64(int(expTime * 1E9), "ns")
            expTime_days = expTime / (60*60*24)
            visitEntry["obsStartMJD"] = visitEntry["expMidptMJD"] - 0.5 * expTime_days
            visitEntries.append(visitEntry)

            # TODO: DM-30623, Add programId, exposureType, cameraTemp,
            # mirror1Temp, mirror2Temp, mirror3Temp, domeTemp, externalTemp,
            # dimmSeeing, pwvGPS, pwvMW, flags, nExposures.

        outputCatalog = astropy.table.Table(rows=visitEntries)
        return pipeBase.Struct(outputCatalog=outputCatalog)


class WriteForcedSourceTableConnections(pipeBase.PipelineTaskConnections,
                                        dimensions=("instrument", "visit", "detector", "skymap", "tract")):

    inputCatalog = connectionTypes.Input(
        doc="Primary per-detector, single-epoch forced-photometry catalog. "
            "By default, it is the output of ForcedPhotCcdTask on calexps",
        name="forced_src",
        storageClass="SourceCatalog",
        dimensions=("instrument", "visit", "detector", "skymap", "tract")
    )
    inputCatalogDiff = connectionTypes.Input(
        doc="Secondary multi-epoch, per-detector, forced photometry catalog. "
            "By default, it is the output of ForcedPhotCcdTask run on image differences.",
        name="forced_diff",
        storageClass="SourceCatalog",
        dimensions=("instrument", "visit", "detector", "skymap", "tract")
    )
    outputCatalog = connectionTypes.Output(
        doc="InputCatalogs horizonatally joined on `objectId` in DataFrame parquet format",
        name="mergedForcedSource",
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "detector", "skymap", "tract")
    )


class WriteForcedSourceTableConfig(pipeBase.PipelineTaskConfig,
                                   pipelineConnections=WriteForcedSourceTableConnections):
    key = lsst.pex.config.Field(
        doc="Column on which to join the two input tables on and make the primary key of the output",
        dtype=str,
        default="objectId",
    )


class WriteForcedSourceTableTask(pipeBase.PipelineTask):
    """Merge and convert per-detector forced source catalogs to DataFrame Parquet format.

    Because the predecessor ForcedPhotCcdTask operates per-detector,
    per-tract, (i.e., it has tract in its dimensions), detectors
    on the tract boundary may have multiple forced source catalogs.

    The successor task TransformForcedSourceTable runs per-patch
    and temporally-aggregates overlapping mergedForcedSource catalogs from all
    available multiple epochs.
    """
    _DefaultName = "writeForcedSourceTable"
    ConfigClass = WriteForcedSourceTableConfig

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        inputs["visit"] = butlerQC.quantum.dataId["visit"]
        inputs["detector"] = butlerQC.quantum.dataId["detector"]
        inputs["band"] = butlerQC.quantum.dataId["band"]
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, inputCatalog, inputCatalogDiff, visit, detector, band=None):
        dfs = []
        for table, dataset, in zip((inputCatalog, inputCatalogDiff), ("calexp", "diff")):
            df = table.asAstropy().to_pandas().set_index(self.config.key, drop=False)
            df = df.reindex(sorted(df.columns), axis=1)
            df["visit"] = visit
            # int16 instead of uint8 because databases don't like unsigned bytes.
            df["detector"] = np.int16(detector)
            df["band"] = band if band else pd.NA
            df.columns = pd.MultiIndex.from_tuples([(dataset, c) for c in df.columns],
                                                   names=("dataset", "column"))

            dfs.append(df)

        outputCatalog = functools.reduce(lambda d1, d2: d1.join(d2), dfs)
        return pipeBase.Struct(outputCatalog=outputCatalog)


class TransformForcedSourceTableConnections(pipeBase.PipelineTaskConnections,
                                            dimensions=("instrument", "skymap", "patch", "tract")):

    inputCatalogs = connectionTypes.Input(
        doc="DataFrames of merged ForcedSources produced by WriteForcedSourceTableTask",
        name="mergedForcedSource",
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "detector", "skymap", "tract"),
        multiple=True,
        deferLoad=True
    )
    referenceCatalog = connectionTypes.Input(
        doc="Reference catalog which was used to seed the forcedPhot. Columns "
            "objectId, detect_isPrimary, detect_isTractInner, detect_isPatchInner "
            "are expected.",
        name="objectTable",
        storageClass="DataFrame",
        dimensions=("tract", "patch", "skymap"),
        deferLoad=True
    )
    outputCatalog = connectionTypes.Output(
        doc="Narrower, temporally-aggregated, per-patch ForcedSource Table transformed and converted per a "
            "specified set of functors",
        name="forcedSourceTable",
        storageClass="ArrowAstropy",
        dimensions=("tract", "patch", "skymap")
    )


class TransformForcedSourceTableConfig(TransformCatalogBaseConfig,
                                       pipelineConnections=TransformForcedSourceTableConnections):
    referenceColumns = pexConfig.ListField(
        dtype=str,
        default=["detect_isPrimary", "detect_isTractInner", "detect_isPatchInner"],
        optional=True,
        doc="Columns to pull from reference catalog",
    )
    keyRef = lsst.pex.config.Field(
        doc="Column on which to join the two input tables on and make the primary key of the output",
        dtype=str,
        default="objectId",
    )
    key = lsst.pex.config.Field(
        doc="Rename the output DataFrame index to this name",
        dtype=str,
        default="forcedSourceId",
    )

    def setDefaults(self):
        super().setDefaults()
        self.functorFile = os.path.join("$PIPE_TASKS_DIR", "schemas", "ForcedSource.yaml")
        self.columnsFromDataId = ["tract", "patch"]


class TransformForcedSourceTableTask(TransformCatalogBaseTask):
    """Transform/standardize a ForcedSource catalog

    Transforms each wide, per-detector forcedSource DataFrame per the
    specification file (per-camera defaults found in ForcedSource.yaml).
    All epochs that overlap the patch are aggregated into one per-patch
    narrow-DataFrame file.

    No de-duplication of rows is performed. Duplicate resolutions flags are
    pulled in from the referenceCatalog: `detect_isPrimary`,
    `detect_isTractInner`,`detect_isPatchInner`, so that user may de-duplicate
    for analysis or compare duplicates for QA.

    The resulting table includes multiple bands. Epochs (MJDs) and other useful
    per-visit rows can be retreived by joining with the CcdVisitTable on
    ccdVisitId.
    """
    _DefaultName = "transformForcedSourceTable"
    ConfigClass = TransformForcedSourceTableConfig

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        if self.funcs is None:
            raise ValueError("config.functorFile is None. "
                             "Must be a valid path to yaml in order to run Task as a PipelineTask.")
        outputs = self.run(inputs["inputCatalogs"], inputs["referenceCatalog"], funcs=self.funcs,
                           dataId=dict(outputRefs.outputCatalog.dataId.mapping))

        butlerQC.put(outputs, outputRefs)

    def run(self, inputCatalogs, referenceCatalog, funcs=None, dataId=None, band=None):
        dfs = []
        refColumns = list(self.config.referenceColumns)
        refColumns.append(self.config.keyRef)
        ref = referenceCatalog.get(parameters={"columns": refColumns})
        if ref.index.name != self.config.keyRef:
            # If the DataFrame we loaded was originally written as some other
            # Parquet type, it probably doesn't have the index set.  If it was
            # written as a DataFrame, the index should already be set and
            # trying to set it again would be an error, since it doens't exist
            # as a regular column anymore.
            ref.set_index(self.config.keyRef, inplace=True)
        self.log.info("Aggregating %s input catalogs" % (len(inputCatalogs)))
        for handle in inputCatalogs:
            result = self.transform(None, handle, funcs, dataId)
            # Filter for only rows that were detected on (overlap) the patch
            dfs.append(result.df.join(ref, how="inner"))

        outputCatalog = pd.concat(dfs)

        if outputCatalog.empty:
            raise NoWorkFound(f"No forced photometry rows for {dataId}.")

        # Now that we are done joining on config.keyRef
        # Change index to config.key by
        outputCatalog.index.rename(self.config.keyRef, inplace=True)
        # Add config.keyRef to the column list
        outputCatalog.reset_index(inplace=True)
        # Set the forcedSourceId to the index. This is specified in the
        # ForcedSource.yaml
        outputCatalog.set_index("forcedSourceId", inplace=True, verify_integrity=True)
        # Rename it to the config.key
        outputCatalog.index.rename(self.config.key, inplace=True)

        self.log.info("Made a table of %d columns and %d rows",
                      len(outputCatalog.columns), len(outputCatalog))
        return pipeBase.Struct(outputCatalog=pandas_to_astropy(outputCatalog))


class ConsolidateTractConnections(pipeBase.PipelineTaskConnections,
                                  defaultTemplates={"catalogType": ""},
                                  dimensions=("instrument", "tract")):
    inputCatalogs = connectionTypes.Input(
        doc="Input per-patch DataFrame Tables to be concatenated",
        name="{catalogType}ForcedSourceTable",
        storageClass="DataFrame",
        dimensions=("tract", "patch", "skymap"),
        multiple=True,
    )

    outputCatalog = connectionTypes.Output(
        doc="Output per-tract concatenation of DataFrame Tables",
        name="{catalogType}ForcedSourceTable_tract",
        storageClass="DataFrame",
        dimensions=("tract", "skymap"),
    )


class ConsolidateTractConfig(pipeBase.PipelineTaskConfig,
                             pipelineConnections=ConsolidateTractConnections):
    pass


class ConsolidateTractTask(pipeBase.PipelineTask):
    """Concatenate any per-patch, dataframe list into a single
    per-tract DataFrame.
    """
    _DefaultName = "ConsolidateTract"
    ConfigClass = ConsolidateTractConfig

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        # Not checking at least one inputCatalog exists because that'd be an
        # empty QG.
        self.log.info("Concatenating %s per-patch %s Tables",
                      len(inputs["inputCatalogs"]),
                      inputRefs.inputCatalogs[0].datasetType.name)
        df = pd.concat(inputs["inputCatalogs"])
        butlerQC.put(pipeBase.Struct(outputCatalog=df), outputRefs)
