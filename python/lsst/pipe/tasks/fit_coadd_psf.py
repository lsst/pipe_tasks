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
    "CoaddPsfFitConfig", "CoaddPsfFitSubConfig", "CoaddPsfFitSubTask", "CoaddPsfFitTask",
]

from .fit_multiband import CatalogExposure, CatalogExposureConfig
from lsst.meas.base import SkyMapIdGeneratorConfig
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT

from abc import ABC, abstractmethod
from pydantic.dataclasses import dataclass


@dataclass(frozen=True, kw_only=True, config=CatalogExposureConfig)
class CatalogExposurePsf(CatalogExposure):
    def get_catalog(self):
        return self.catalog

    def get_psf_image(self, source):
        bbox = source.getFootprint().getBBox()
        center = bbox.getCenter()
        return self.exposure.getPsf().computeKernelImage(center).array


CoaddPsfFitBaseTemplates = {
    "name_coadd": "deep",
    "name_output_method": "multiprofit",
}


class CoaddPsfFitConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("tract", "patch", "band", "skymap"),
    defaultTemplates=CoaddPsfFitBaseTemplates,
):
    coadd = cT.Input(
        doc="Coadd image to fit a PSF model to",
        name="{name_coadd}Coadd_calexp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "band", "skymap"),
    )
    cat_meas = cT.Input(
        doc="Deblended single-band source catalog",
        name="{name_coadd}Coadd_meas",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "band", "skymap"),
    )
    cat_output = cT.Output(
        doc="Output PSF fit parameter catalog",
        name="{name_coadd}Coadd_psfs_{name_output_method}",
        storageClass="ArrowTable",
        dimensions=("tract", "patch", "band", "skymap"),
    )


class CoaddPsfFitSubConfig(pexConfig.Config):
    """Base config class for the CoaddPsfFitTask.

    Implementing classes may add any necessary attributes.
    """


class CoaddPsfFitSubTask(pipeBase.Task, ABC):
    """Interface for CoaddPsfFitTask subtasks to fit PSFs.

    Parameters
    ----------
    **kwargs
        Additional arguments to be passed to the `lsst.pipe.base.Task`
        constructor.
    """
    ConfigClass = CoaddPsfFitSubConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def run(
        self, catexp: CatalogExposurePsf
    ) -> pipeBase.Struct:
        """Fit PSF images at locations of sources in a single exposure.

        Parameters
        ----------
        catexp : `CatalogExposurePsf`
            An exposure to fit a model PSF at the position of all
            sources in the corresponding catalog.

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


class CoaddPsfFitConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=CoaddPsfFitConnections,
):
    """Configure a CoaddPsfFitTask, including a configurable fitting subtask.
    """
    fit_coadd_psf = pexConfig.ConfigurableField(
        target=CoaddPsfFitSubTask,
        doc="Task to fit PSF models for a single coadd",
    )
    idGenerator = SkyMapIdGeneratorConfig.make_field()


class CoaddPsfFitTask(pipeBase.PipelineTask):
    """Fit a PSF model at the location of sources in a coadd.

    This task is intended to fit only a single PSF model at the
    centroid of all of the sources in a single coadd exposure.
    Subtasks may choose to filter which sources they fit,
    and may output whatever columns they desire in addition to
    the minimum of 'id'.
    """
    ConfigClass = CoaddPsfFitConfig
    _DefaultName = "CoaddPsfFit"

    def __init__(self, initInputs, **kwargs):
        super().__init__(initInputs=initInputs, **kwargs)
        self.makeSubtask("fit_coadd_psf")

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        id_tp = self.config.idGenerator.apply(butlerQC.quantum.dataId).catalog_id
        dataId = inputRefs.cat_meas.dataId
        for dataRef in (inputRefs.coadd,):
            if dataRef.dataId != dataId:
                raise RuntimeError(f'{dataRef=}.dataId != {inputRefs.cat_meas.dataId=}')

        catexp = CatalogExposurePsf(
            catalog=inputs['cat_meas'], exposure=inputs['coadd'], dataId=dataId, id_tract_patch=id_tp,
        )
        outputs = self.run(catexp=catexp)
        butlerQC.put(outputs, outputRefs)

    def run(self, catexp: CatalogExposurePsf) -> pipeBase.Struct:
        """Fit a PSF model at the location of sources in a coadd.

        Parameters
        ----------
        catexp : `typing.List [CatalogExposurePsf]`
            A list of catalog-exposure pairs in a given band.

        Returns
        -------
        retStruct : `lsst.pipe.base.Struct`
            A struct with a cat_output attribute containing the output
            measurement catalog.

        Notes
        -----
        Subtasks may have further requirements; see `CoaddPsfFitSubTask.run`.
        """
        cat_output = self.fit_coadd_psf.run(catexp).output
        retStruct = pipeBase.Struct(cat_output=cat_output)
        return retStruct
