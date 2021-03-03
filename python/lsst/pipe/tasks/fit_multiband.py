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

__all__ = ["CatalogExposure", "MultibandFitSubTask", "MultibandFitConfig", "MultibandFitTask"]

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.daf.butler as dafButler
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from typing import Dict, Iterable, List, Optional, Set


@dataclass(frozen=True)
class CatalogExposure:
    @property
    def band(self) -> str:
        return self.dataId['band']

    @property
    def calib(self) -> Optional[afwImage.PhotoCalib]:
        return None if self.exposure is None else self.exposure.getPhotoCalib()

    catalog: Optional[afwTable.SourceCatalog]
    exposure: Optional[afwImage.Exposure]
    dataId: dafButler.DataCoordinate
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if 'band' not in self.dataId:
            raise ValueError(f'dataId={self.dataId} must have a band')


multibandFitBaseTemplates = {
    "name_input_coadd": "deep",
    "name_output_coadd": "deep",
    "name_output_cat": "fit",
}


class MultibandFitConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("tract", "patch", "skymap"),
    defaultTemplates=multibandFitBaseTemplates,
):
    cat_ref = cT.Input(
        doc="Reference multiband source catalog",
        name="{name_input_coadd}Coadd_ref",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "skymap"),
    )
    cats_meas = cT.Input(
        doc="Deblended single-band source catalogs",
        name="{name_input_coadd}Coadd_meas",
        storageClass="SourceCatalog",
        multiple=True,
        dimensions=("tract", "patch", "band", "skymap"),
    )
    coadds = cT.Input(
        doc="Exposures on which to run fits",
        name="{name_input_coadd}Coadd_calexp",
        storageClass="ExposureF",
        multiple=True,
        dimensions=("tract", "patch", "band", "skymap"),
    )
    cat_output = cT.Output(
        doc="Measurement multi-band catalog",
        name="{name_output_coadd}Coadd_{name_output_cat}",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "skymap"),
    )
    cat_ref_schema = cT.InitInput(
        doc="Schema associated with a ref source catalog",
        storageClass="SourceCatalog",
        name="{name_input_coadd}Coadd_ref_schema",
    )
    cat_output_schema = cT.InitOutput(
        doc="Output of the schema used in deblending task",
        name="{name_output_coadd}Coadd_{name_output_cat}_schema",
        storageClass="SourceCatalog"
    )

    def adjustQuantum(self, datasetRefMap):
        """Validates the `lsst.daf.butler.DatasetRef` bands against the
        subtask's list of bands to fit and drops unnecessary bands.

        Parameters
        ----------
        datasetRefMap : `NamedKeyDict`
            Mapping from dataset type to a `set` of
            `lsst.daf.butler.DatasetRef` objects

        Returns
        -------
        datasetRefMap : `NamedKeyDict`
            Modified mapping of input with possibly adjusted
            `lsst.daf.butler.DatasetRef` objects.

        Raises
        ------
        ValueError
            Raised if any of the per-band datasets have an inconsistent band
            set, or if the band set to fit is not a subset of the data bands.

        """
        datasetRefMap = super().adjustQuantum(datasetRefMap)
        # Check which bands are going to be fit
        bands_fit, bands_read_only = self.config.get_band_sets()
        bands_needed = bands_fit.union(bands_read_only)

        bands_data = None
        bands_extra = set()

        for type_d, ref_d in datasetRefMap.items():
            # Datasets without bands in their dimensions should be fine
            if 'band' in type_d.dimensions:
                bands_set = {dref.dataId['band'] for dref in ref_d}
                if bands_data is None:
                    bands_data = bands_set
                    if bands_needed != bands_data:
                        if not bands_needed.issubset(bands_data):
                            raise ValueError(
                                f'Datarefs={ref_d} have data with bands in the set={bands_set},'
                                f'which is not a subset of the required bands={bands_needed} defined by '
                                f'{self.config.__class__}.fit_multiband='
                                f'{self.config.fit_multiband._value.__class__}\'s attributes'
                                f' bands_fit={bands_fit} and bands_read_only()={bands_read_only}.'
                                f' Add the required bands={bands_needed.difference(bands_data)}.'
                            )
                        else:
                            bands_extra = bands_data.difference(bands_needed)
                elif bands_set != bands_data:
                    raise ValueError(
                        f'Datarefs={ref_d} have data with bands in the set={bands_set}'
                        f' which differs from the previous={bands_data}); bandsets must be identical.'
                    )
                if bands_extra:
                    for dref in ref_d:
                        if dref.dataId['band'] in bands_extra:
                            ref_d.remove(dref)
        return datasetRefMap


class MultibandFitSubConfig(pexConfig.Config):
    """Config class for the MultibandFitTask to define methods returning
    values that depend on multiple config settings.

    """
    def bands_read_only(self) -> Set:
        """Return the set of bands that the Task needs to read (e.g. for
        defining priors) but not necessarily fit.

        Returns
        -------
        The set of such bands.
        """
        return set()


class MultibandFitSubTask(pipeBase.Task, ABC):
    """An abstract interface for subtasks of MultibandFitTask to perform
    multiband fitting of deblended sources.

    Parameters
    ----------
    schema : `lsst.afw.table.Schema`
        The input schema for the reference source catalog, used to initialize
        the output schema.
    **kwargs
        Additional arguments to be passed to the `lsst.pipe.base.Task`
        constructor.
    """
    ConfigClass = MultibandFitSubConfig

    def __init__(self, schema: afwTable.Schema, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def run(
        self, catexps: Iterable[CatalogExposure], cat_ref: afwTable.SourceCatalog
    ) -> pipeBase.Struct:
        """Fit sources from a reference catalog using data from multiple
        exposures in the same patch.

        Parameters
        ----------
        catexps : `typing.List [CatalogExposure]`
            A list of catalog-exposure pairs in a given band.
        cat_ref : `lsst.afw.table.SourceCatalog`
            A reference source catalog to fit.

        Returns
        -------
        retStruct : `lsst.pipe.base.Struct`
            A struct with a cat_output attribute containing the output
            measurement catalog.

        Notes
        -----
        Subclasses may have further requirements on the input parameters,
        including:
            - Passing only one catexp per band;
            - Catalogs containing HeavyFootprints with deblended images;
            - Fitting only a subset of the sources.
        If any requirements are not met, the subtask should fail as soon as
        possible.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def schema(self) -> afwTable.Schema:
        raise NotImplementedError()


class MultibandFitConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=MultibandFitConnections,
):
    """Configuration class for the MultibandFitTask, containing a
    configurable subtask that does all fitting.
    """
    fit_multiband = pexConfig.ConfigurableField(
        target=MultibandFitSubTask,
        doc="Task to fit sources using multiple bands",
    )

    def get_band_sets(self):
        """Get the set of bands required by the fit_multiband subtask.

        Returns
        -------
        bands_fit : `set`
            The set of bands that the subtask will fit.
        bands_read_only : `set`
            The set of bands that the subtask will only read data (measurement catalog and exposure) for.
        """
        try:
            bands_fit = self.fit_multiband.bands_fit
        except AttributeError:
            raise RuntimeError(f'{__class__}.fit_multiband must have bands_fit attribute') from None
        bands_read_only = self.fit_multiband.bands_read_only()
        return set(bands_fit), set(bands_read_only)


class MultibandFitTask(pipeBase.PipelineTask):
    ConfigClass = MultibandFitConfig
    _DefaultName = "multibandFit"

    def __init__(self, initInputs, **kwargs):
        super().__init__(initInputs=initInputs, **kwargs)
        self.makeSubtask("fit_multiband", schema=initInputs["cat_ref_schema"].schema)
        self.cat_output_schema = afwTable.SourceCatalog(self.fit_multiband.schema)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        input_refs_objs = [(inputRefs.cats_meas, inputs['cats_meas']), (inputRefs.coadds, inputs['coadds'])]
        cats, exps = [
            {dRef.dataId: obj for dRef, obj in zip(refs, objs)}
            for refs, objs in input_refs_objs
        ]
        dataIds = set(cats).union(set(exps))
        catexps = [
            CatalogExposure(catalog=cats.get(dataId), exposure=exps.get(dataId), dataId=dataId)
            for dataId in dataIds
        ]
        outputs = self.run(catexps=catexps, cat_ref=inputs['cat_ref'])
        butlerQC.put(outputs, outputRefs)
        # Validate the output catalog's schema and raise if inconsistent (after output to allow debugging)
        if outputs.cat_output.schema != self.cat_output_schema.schema:
            raise RuntimeError(f'{__class__}.config.fit_multiband.run schema != initOutput schema:'
                               f' {outputs.cat_output.schema} vs {self.cat_output_schema.schema}')

    def run(self, catexps: List[CatalogExposure], cat_ref: afwTable.SourceCatalog) -> pipeBase.Struct:
        """Fit sources from a reference catalog using data from multiple
        exposures in the same region (patch).

        Parameters
        ----------
        catexps : `typing.List [CatalogExposure]`
            A list of catalog-exposure pairs in a given band.
        cat_ref : `lsst.afw.table.SourceCatalog`
            A reference source catalog to fit.

        Returns
        -------
        retStruct : `lsst.pipe.base.Struct`
            A struct with a cat_output attribute containing the output
            measurement catalog.

        Notes
        -----
        Subtasks may have further requirements; see `MultibandFitSubTask.run`.
        """
        cat_output = self.fit_multiband.run(catexps, cat_ref).output
        retStruct = pipeBase.Struct(cat_output=cat_output)
        return retStruct
