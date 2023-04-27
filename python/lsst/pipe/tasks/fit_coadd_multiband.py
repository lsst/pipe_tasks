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

__all__ = [
    "CoaddMultibandFitConfig", "CoaddMultibandFitSubConfig", "CoaddMultibandFitSubTask",
    "CoaddMultibandFitTask",
]

from .fit_multiband import CatalogExposure, CatalogExposureConfig

import lsst.afw.table as afwTable
from lsst.meas.base import SkyMapIdGeneratorConfig
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT

import astropy
from abc import ABC, abstractmethod
from pydantic import Field
from pydantic.dataclasses import dataclass
from typing import Iterable

CoaddMultibandFitBaseTemplates = {
    "name_coadd": "deep",
    "name_method": "multiprofit",
}


@dataclass(frozen=True, kw_only=True, config=CatalogExposureConfig)
class CatalogExposureInputs(CatalogExposure):
    table_psf_fits: astropy.table.Table = Field(title="A table of PSF fit parameters for each source")

    def get_catalog(self):
        return self.catalog


class CoaddMultibandFitConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("tract", "patch", "skymap"),
    defaultTemplates=CoaddMultibandFitBaseTemplates,
):
    cat_ref = cT.Input(
        doc="Reference multiband source catalog",
        name="{name_coadd}Coadd_ref",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "skymap"),
    )
    cats_meas = cT.Input(
        doc="Deblended single-band source catalogs",
        name="{name_coadd}Coadd_meas",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "band", "skymap"),
        multiple=True,
    )
    coadds = cT.Input(
        doc="Exposures on which to run fits",
        name="{name_coadd}Coadd_calexp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "band", "skymap"),
        multiple=True,
    )
    models_psf = cT.Input(
        doc="Input PSF model parameter catalog",
        # Consider allowing independent psf fit method
        name="{name_coadd}Coadd_psfs_{name_method}",
        storageClass="ArrowAstropy",
        dimensions=("tract", "patch", "band", "skymap"),
        multiple=True,
    )
    models_scarlet = pipeBase.connectionTypes.Input(
        doc="Multiband scarlet models produced by the deblender",
        name="{name_coadd}Coadd_scarletModelData",
        storageClass="ScarletModelData",
        dimensions=("tract", "patch", "skymap"),
    )
    cat_output = cT.Output(
        doc="Output source model fit parameter catalog",
        name="{name_coadd}Coadd_objects_{name_method}",
        storageClass="ArrowTable",
        dimensions=("tract", "patch", "skymap"),
    )

    def adjustQuantum(self, inputs, outputs, label, data_id):
        """Validates the `lsst.daf.butler.DatasetRef` bands against the
        subtask's list of bands to fit and drops unnecessary bands.

        Parameters
        ----------
        inputs : `dict`
            Dictionary whose keys are an input (regular or prerequisite)
            connection name and whose values are a tuple of the connection
            instance and a collection of associated `DatasetRef` objects.
            The exact type of the nested collections is unspecified; it can be
            assumed to be multi-pass iterable and support `len` and ``in``, but
            it should not be mutated in place.  In contrast, the outer
            dictionaries are guaranteed to be temporary copies that are true
            `dict` instances, and hence may be modified and even returned; this
            is especially useful for delegating to `super` (see notes below).
        outputs : `Mapping`
            Mapping of output datasets, with the same structure as ``inputs``.
        label : `str`
            Label for this task in the pipeline (should be used in all
            diagnostic messages).
        data_id : `lsst.daf.butler.DataCoordinate`
            Data ID for this quantum in the pipeline (should be used in all
            diagnostic messages).

        Returns
        -------
        adjusted_inputs : `Mapping`
            Mapping of the same form as ``inputs`` with updated containers of
            input `DatasetRef` objects.  All inputs involving the 'band'
            dimension are adjusted to put them in consistent order and remove
            unneeded bands.
        adjusted_outputs : `Mapping`
            Mapping of updated output datasets; always empty for this task.

        Raises
        ------
        lsst.pipe.base.NoWorkFound
            Raised if there are not enough of the right bands to run the task
            on this quantum.
        """
        # Check which bands are going to be fit
        bands_fit, bands_read_only = self.config.get_band_sets()
        bands_needed = bands_fit.union(bands_read_only)

        adjusted_inputs = {}
        for connection_name, (connection, dataset_refs) in inputs.items():
            # Datasets without bands in their dimensions should be fine
            if 'band' in connection.dimensions:
                datasets_by_band = {dref.dataId['band']: dref for dref in dataset_refs}
                if not bands_needed.issubset(datasets_by_band.keys()):
                    raise pipeBase.NoWorkFound(
                        f'DatasetRefs={dataset_refs} have data with bands in the'
                        f' set={set(datasets_by_band.keys())},'
                        f' which is not a superset of the required bands={bands_needed} defined by'
                        f' {self.config.__class__}.fit_coadd_multiband='
                        f'{self.config.fit_coadd_multiband._value.__class__}\'s attributes'
                        f' bands_fit={bands_fit} and bands_read_only()={bands_read_only}.'
                        f' Add the required bands={bands_needed.difference(datasets_by_band.keys())}.'
                    )
                # Adjust all datasets with band dimensions to include just
                # the needed bands, in consistent order.
                adjusted_inputs[connection_name] = (
                    connection,
                    [datasets_by_band[band] for band in bands_needed]
                )

        # Delegate to super for more checks.
        inputs.update(adjusted_inputs)
        super().adjustQuantum(inputs, outputs, label, data_id)
        return adjusted_inputs, {}


class CoaddMultibandFitSubConfig(pexConfig.Config):
    """Configuration for implementing fitter subtasks.
    """
    @abstractmethod
    def bands_read_only(self) -> set:
        """Return the set of bands that the Task needs to read (e.g. for
        defining priors) but not necessarily fit.

        Returns
        -------
        The set of such bands.
        """


class CoaddMultibandFitSubTask(pipeBase.Task, ABC):
    """Subtask interface for multiband fitting of deblended sources.

    Parameters
    ----------
    **kwargs
        Additional arguments to be passed to the `lsst.pipe.base.Task`
        constructor.
    """
    ConfigClass = CoaddMultibandFitSubConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def run(
        self, catexps: Iterable[CatalogExposureInputs], cat_ref: afwTable.SourceCatalog
    ) -> pipeBase.Struct:
        """Fit models to deblended sources from multi-band inputs.

        Parameters
        ----------
        catexps : `typing.List [CatalogExposureInputs]`
            A list of catalog-exposure pairs with metadata in a given band.
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


class CoaddMultibandFitConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=CoaddMultibandFitConnections,
):
    """Configure a CoaddMultibandFitTask, including a configurable fitting subtask.
    """
    fit_coadd_multiband = pexConfig.ConfigurableField(
        target=CoaddMultibandFitSubTask,
        doc="Task to fit sources using multiple bands",
    )
    idGenerator = SkyMapIdGeneratorConfig.make_field()

    def get_band_sets(self):
        """Get the set of bands required by the fit_coadd_multiband subtask.

        Returns
        -------
        bands_fit : `set`
            The set of bands that the subtask will fit.
        bands_read_only : `set`
            The set of bands that the subtask will only read data
            (measurement catalog and exposure) for.
        """
        try:
            bands_fit = self.fit_coadd_multiband.bands_fit
        except AttributeError:
            raise RuntimeError(f'{__class__}.fit_coadd_multiband must have bands_fit attribute') from None
        bands_read_only = self.fit_coadd_multiband.bands_read_only()
        return set(bands_fit), set(bands_read_only)


class CoaddMultibandFitTask(pipeBase.PipelineTask):
    """Fit deblended exposures in multiple bands simultaneously.

    It is generally assumed but not enforced (except optionally by the
    configurable `fit_coadd_multiband` subtask) that there is only one exposure
    per band, presumably a coadd.
    """
    ConfigClass = CoaddMultibandFitConfig
    _DefaultName = "CoaddMultibandFit"

    def __init__(self, initInputs, **kwargs):
        super().__init__(initInputs=initInputs, **kwargs)
        self.makeSubtask("fit_coadd_multiband")

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        id_tp = self.config.id_generator.apply(butlerQC.quantum.dataId).catalog_id
        # This is a roundabout way of ensuring all inputs get sorted and matched
        input_refs_objs = [(getattr(inputRefs, key), inputs[key])
                           for key in ("cats_meas", "coadds", "models_psf")]
        cats, exps, models_psf = [
            {dRef.dataId: obj for dRef, obj in zip(refs, objs)}
            for refs, objs in input_refs_objs
        ]
        dataIds = set(cats).union(set(exps))
        models_scarlet = inputs["models_scarlet"]
        catexps = [None]*len(dataIds)
        for idx, dataId in enumerate(dataIds):
            catalog = cats[dataId]
            exposure = exps[dataId]
            models_scarlet.updateCatalogFootprints(
                catalog=catalog,
                band=dataId['band'],
                psfModel=exposure.getPsf(),
                redistributeImage=exposure.image,
                removeScarletData=True,
                updateFluxColumns=False,
            )
            catexps[idx] = CatalogExposureInputs(
                catalog=catalog, exposure=exposure, table_psf_fits=models_psf[dataId],
                dataId=dataId, id_tract_patch=id_tp,
            )
        outputs = self.run(catexps=catexps, cat_ref=inputs['cat_ref'])
        butlerQC.put(outputs, outputRefs)

    def run(self, catexps: list[CatalogExposure], cat_ref: afwTable.SourceCatalog) -> pipeBase.Struct:
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
        Subtasks may have further requirements; see `CoaddMultibandFitSubTask.run`.
        """
        cat_output = self.fit_coadd_multiband.run(catalog_multi=cat_ref, catexps=catexps).output
        retStruct = pipeBase.Struct(cat_output=cat_output)
        return retStruct
