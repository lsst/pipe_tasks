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
import numpy as np

from lsstDebug import getDebugFrame
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.daf.base as dafBase
from lsst.afw.math import BackgroundList
from lsst.afw.table import SourceTable, SourceCatalog, IdFactory
from lsst.meas.algorithms import SubtractBackgroundTask, SourceDetectionTask, MeasureApCorrTask
from lsst.meas.algorithms.installGaussianPsf import InstallGaussianPsfTask
from lsst.meas.astrom import RefMatchTask, displayAstrometry
from lsst.meas.extensions.astrometryNet import LoadAstrometryNetObjectsTask
from lsst.obs.base import ExposureIdInfo
from lsst.meas.base import SingleFrameMeasurementTask, ApplyApCorrTask, CatalogCalculationTask
from lsst.meas.deblender import SourceDeblendTask
from .measurePsf import MeasurePsfTask
from .repair import RepairTask

__all__ = ["CharacterizeImageConfig", "CharacterizeImageTask"]


class CharacterizeImageConfig(pexConfig.Config):
    """Config for CharacterizeImageTask"""
    doMeasurePsf = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Measure PSF? If False then for all subsequent operations use either existing PSF "
            "model when present, or install simple PSF model when not (see installSimplePsf "
            "config options)"
    )
    doWrite = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Persist results?",
    )
    doWriteExposure = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Write icExp and icExpBackground in addition to icSrc? Ignored if doWrite False.",
    )
    psfIterations = pexConfig.RangeField(
        dtype=int,
        default=2,
        min=1,
        doc="Number of iterations of detect sources, measure sources, "
            "estimate PSF. If useSimplePsf is True then 2 should be plenty; "
            "otherwise more may be wanted.",
    )
    background = pexConfig.ConfigurableField(
        target=SubtractBackgroundTask,
        doc="Configuration for initial background estimation",
    )
    detection = pexConfig.ConfigurableField(
        target=SourceDetectionTask,
        doc="Detect sources"
    )
    doDeblend = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Run deblender input exposure"
    )
    deblend = pexConfig.ConfigurableField(
        target=SourceDeblendTask,
        doc="Split blended source into their components"
    )
    measurement = pexConfig.ConfigurableField(
        target=SingleFrameMeasurementTask,
        doc="Measure sources"
    )
    doApCorr = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Run subtasks to measure and apply aperture corrections"
    )
    measureApCorr = pexConfig.ConfigurableField(
        target=MeasureApCorrTask,
        doc="Subtask to measure aperture corrections"
    )
    applyApCorr = pexConfig.ConfigurableField(
        target=ApplyApCorrTask,
        doc="Subtask to apply aperture corrections"
    )
    # If doApCorr is False, and the exposure does not have apcorrections already applied, the
    # active plugins in catalogCalculation almost certainly should not contain the characterization plugin
    catalogCalculation = pexConfig.ConfigurableField(
        target=CatalogCalculationTask,
        doc="Subtask to run catalogCalculation plugins on catalog"
    )
    useSimplePsf = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Replace the existing PSF model with a simplified version that has the same sigma "
        "at the start of each PSF determination iteration? Doing so makes PSF determination "
        "converge more robustly and quickly.",
    )
    installSimplePsf = pexConfig.ConfigurableField(
        target=InstallGaussianPsfTask,
        doc="Install a simple PSF model",
    )
    refObjLoader = pexConfig.ConfigurableField(
        target=LoadAstrometryNetObjectsTask,
        doc="reference object loader",
    )
    ref_match = pexConfig.ConfigurableField(
        target=RefMatchTask,
        doc="Task to load and match reference objects. Only used if measurePsf can use matches. "
        "Warning: matching will only work well if the initial WCS is accurate enough "
        "to give good matches (roughly: good to 3 arcsec across the CCD).",
    )
    measurePsf = pexConfig.ConfigurableField(
        target=MeasurePsfTask,
        doc="Measure PSF",
    )
    repair = pexConfig.ConfigurableField(
        target=RepairTask,
        doc="Remove cosmic rays",
    )
    checkUnitsParseStrict = pexConfig.Field(
        doc="Strictness of Astropy unit compatibility check, can be 'raise', 'warn' or 'silent'",
        dtype=str,
        default="raise",
    )

    def setDefaults(self):
        pexConfig.Config.setDefaults(self)
        # just detect bright stars; includeThresholdMultipler=10 seems large,
        # but these are the values we have been using
        self.detection.thresholdValue = 5.0
        self.detection.includeThresholdMultiplier = 10.0
        # do not deblend, as it makes a mess
        self.doDeblend = False
        # measure and apply aperture correction; note: measuring and applying aperture
        # correction are disabled until the final measurement, after PSF is measured
        self.doApCorr = True
        # minimal set of measurements needed to determine PSF
        self.measurement.plugins.names = [
            "base_PixelFlags",
            "base_SdssCentroid",
            "base_SdssShape",
            "base_GaussianFlux",
            "base_PsfFlux",
            "base_CircularApertureFlux",
        ]

    def validate(self):
        if self.doApCorr and not self.measurePsf:
            raise RuntimeError("Must measure PSF to measure aperture correction, "
                               "because flags determined by PSF measurement are used to identify "
                               "sources used to measure aperture correction")


class CharacterizeImageTask(pipeBase.CmdLineTask):
    """Measure bright sources and use this to estimate background and PSF of
    an exposure.

    Given an exposure (typically, e.g., as output by IsrTask):
        (1) Iterate over the following config.psfIteration times, or once if
        config.doMeasurePsf is False:
            - detect and measure bright sources
            - do an initial repair of cosmic rays (no interpolation yet)
            - measure and subtract background
            - do an initial PSF measurement estimate
        (2) Update or set final PSF
        (3) Do final cosmic ray repair, including interpolation
        (4) Perform final measurement with final PSF, including measuring and
            applying aperture correction, if applicable

    Parameters
    ----------
    butler : `lsst.daf.persistence.Butler`, optional
        A butler object is passed to the refObjLoader constructor in case
        it is needed to load catalogs.
        May be None if a catalog-based star selector is not used, if the
        reference object loader constructor does not require a butler, or if a
        reference object loader is passed directly via the refObjLoader
        argument.
    refObjLoader : `lsst.meas.algorithms.LoadReferenceObjectsTask`, optional
        An instance of LoadReferenceObjectsTask that supplies an
        external reference catalog to a catalog-based star selector.
        May be None if a catalog star selector is not used or the loader
        can be constructed from the butler argument.
    schema : `lsst.afw.table.SourceTable`, optional
        Initial schema, or None.
    **kwargs
        Other keyword arguments for `lsst.pipe.base.CmdLineTask`

    Notes
    -----
    If you want this task to load inputs or persist outputs, then call
    the `runDataRef` method (a thin wrapper around the `run` method).

    If you already have the inputs loaded and do not want to persist the
    output, then it is more direct to call the `run` method.
    """
    ConfigClass = CharacterizeImageConfig
    _DefaultName = "characterizeImage"
    RunnerClass = pipeBase.ButlerInitializedTaskRunner

    def __init__(self, butler=None, refObjLoader=None, schema=None, **kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)
        if schema is None:
            schema = SourceTable.makeMinimalSchema()
        self.schema = schema
        self.makeSubtask("background")
        self.makeSubtask("installSimplePsf")
        self.makeSubtask("repair")
        self.makeSubtask("measurePsf", schema=self.schema)
        if self.config.doMeasurePsf and self.measurePsf.usesMatches:
            if not refObjLoader:
                self.makeSubtask('refObjLoader', butler=butler)
                refObjLoader = self.refObjLoader
            self.makeSubtask("ref_match", refObjLoader=refObjLoader)
        self.algMetadata = dafBase.PropertyList()
        self.makeSubtask('detection', schema=self.schema)
        if self.config.doDeblend:
            self.makeSubtask("deblend", schema=self.schema)
        self.makeSubtask('measurement', schema=self.schema, algMetadata=self.algMetadata)
        if self.config.doApCorr:
            self.makeSubtask('measureApCorr', schema=self.schema)
            self.makeSubtask('applyApCorr', schema=self.schema)
        self.makeSubtask('catalogCalculation', schema=self.schema)
        self._initialFrame = getDebugFrame(self._display, "frame") or 1
        self._frame = self._initialFrame
        self.schema.checkUnits(parse_strict=self.config.checkUnitsParseStrict)

    @pipeBase.timeMethod
    def runDataRef(self, dataRef, exposure=None, background=None, doUnpersist=True):
        """Characterize a science image and optionally persist the results.

        This unpacks an exposure and passes it to the characterize method
        to do the work.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Butler data reference for science exposure
        exposure : `lsst.afw.image.Exposure`, optional
            Exposure to characterize.
            If None, then use the butler to get the "postISRCCD" dataset type.
        background : `lsst.afw.math.BackgroundList`, optional
            Initial model of background already subtracted from exposure
            (an lsst.afw.math.BackgroundList). May be None if no background has
            been subtracted, which is typical for image characterization.
            A refined background model is output.
        doUnpersist : `bool`
            If True, the exposure is read from the repository
            and the exposure and background arguments must be None.
            If False, the exposure must be provided.
            True is intended for running as a command-line task, False for
            running as a subtask.

        Returns
        -------
        charRes : `lsst.pipe.base.Struct`
            Result struct from the final iteration of
            detectMeasureAndEstimatePsf with components:
            - ``exposure`` : characterized exposure; image is repaired by
                interpolating over cosmic rays, mask is updated accordingly,
                and the PSF model is set (an `lsst.afw.image.Exposure`)
            - ``sourceCat`` : catalog of detected sources
                (an `lsst.afw.table.SourceCatalog`)
            - ``background`` : model of background subtracted from exposure
                (an `lsst.afw.math.BackgroundList`)
            - ``psfCellSet`` : spatial cells of PSF candidates
                (an `lsst.afw.math.SpatialCellSet`)

        Raises
        ------
        RuntimeError
            Raised if doUnpersist True but exposure and background
            are not both None; also raised if doUnpersist False but
            exposure is not provided.
        """
        self._frame = self._initialFrame  # reset debug display frame
        self.log.info("Processing %s" % (dataRef.dataId))

        if doUnpersist:
            if exposure is not None or background is not None:
                raise RuntimeError("doUnpersist true; exposure and background must be None")
            exposure = dataRef.get("postISRCCD", immediate=True)
        elif exposure is None:
            raise RuntimeError("doUnpersist false; exposure must be provided")

        exposureIdInfo = dataRef.get("expIdInfo")

        charRes = self.run(
            exposure=exposure,
            exposureIdInfo=exposureIdInfo,
            background=background,
        )

        if self.config.doWrite:
            dataRef.put(charRes.sourceCat, "icSrc")
            if self.config.doWriteExposure:
                dataRef.put(charRes.exposure, "icExp")
                dataRef.put(charRes.background, "icExpBackground")

        return charRes

    @pipeBase.timeMethod
    def run(self, exposure, exposureIdInfo=None, background=None):
        """Characterize a science image.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure to characterize.
        exposureIdInfo : `lsst.obs.base.ExposureIdInfo`, optional
            ID info for exposure. If not provided, returned SourceCatalog IDs
            will not be globally unique.
        background : `lsst.afw.math.BackgroundList`, optional
            initial model of background already subtracted from exposure.
            May be None if no background has been subtracted, which is typical
            for image characterization.

        Returns
        -------
        dmeRes : `lsst.pipe.base.Struct`
            Result struct from the final iteration of
            detectMeasureAndEstimatePsf with components:
            - ``exposure`` : characterized exposure; image is repaired by
                interpolating over cosmic rays, mask is updated accordingly,
                and the PSF model is set (an `lsst.afw.image.Exposure`)
            - ``sourceCat`` : catalog of detected sources
                (an `lsst.afw.table.SourceCatalog`)
            - ``background`` : model of background subtracted from exposure
                (an `lsst.afw.math.BackgroundList`)
            - ``psfCellSet`` : spatial cells of PSF candidates
                (an `lsst.afw.math.SpatialCellSet`)
        """
        self._frame = self._initialFrame  # reset debug display frame

        if not self.config.doMeasurePsf and not exposure.hasPsf():
            self.log.warn("Source catalog detected and measured with placeholder or default PSF")
            self.installSimplePsf.run(exposure=exposure)

        if exposureIdInfo is None:
            exposureIdInfo = ExposureIdInfo()

        # subtract an initial estimate of background level
        background = self.background.run(exposure).background

        psfIterations = self.config.psfIterations if self.config.doMeasurePsf else 1
        for i in range(psfIterations):
            dmeRes = self.detectMeasureAndEstimatePsf(
                exposure=exposure,
                exposureIdInfo=exposureIdInfo,
                background=background,
            )

            psf = dmeRes.exposure.getPsf()
            psfSigma = psf.computeShape().getDeterminantRadius()
            psfDimensions = psf.computeImage().getDimensions()
            medBackground = np.median(dmeRes.background.getImage().getArray())
            self.log.info("iter %s; PSF sigma=%0.2f, dimensions=%s; median background=%0.2f" %
                          (i + 1, psfSigma, psfDimensions, medBackground))

        self.display("psf", exposure=dmeRes.exposure, sourceCat=dmeRes.sourceCat)

        # perform final repair with final PSF
        self.repair.run(exposure=dmeRes.exposure)
        self.display("repair", exposure=dmeRes.exposure, sourceCat=dmeRes.sourceCat)

        # perform final measurement with final PSF, including measuring and
        # applying aperture correction, if wanted
        self.measurement.run(measCat=dmeRes.sourceCat, exposure=dmeRes.exposure,
                             exposureId=exposureIdInfo.expId)
        if self.config.doApCorr:
            apCorrMap = self.measureApCorr.run(exposure=dmeRes.exposure, catalog=dmeRes.sourceCat).apCorrMap
            dmeRes.exposure.getInfo().setApCorrMap(apCorrMap)
            self.applyApCorr.run(catalog=dmeRes.sourceCat, apCorrMap=exposure.getInfo().getApCorrMap())
        self.catalogCalculation.run(dmeRes.sourceCat)

        self.display("measure", exposure=dmeRes.exposure, sourceCat=dmeRes.sourceCat)

        return dmeRes

    @pipeBase.timeMethod
    def detectMeasureAndEstimatePsf(self, exposure, exposureIdInfo, background):
        """Perform one iteration of detect, measure, and estimate PSF.

        The following changes are made to the input exposure:
            - update or set psf
            - update detection and cosmic ray mask planes
            - subtract background

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure to characterize.
        exposureIdInfo : `lsst.afw.math.BackgroundList`
            ID info for exposure.
        background : `lsst.afw.math.BackgroundList`
            Initial model of background already subtracted from exposure.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Result struct (from the final iteration of detect sources, measure
            sources, and estimate PSF) with components:
            - ``exposure`` : characterized exposure; image is repaired by
                interpolating over cosmic rays, mask is updated accordingly,
                and the PSF model is set (an `lsst.afw.image.Exposure`)
            - ``sourceCat`` : catalog of detected sources
                (an `lsst.afw.table.SourceCatalog`)
            - ``background`` : model of background subtracted from exposure
                (an `lsst.afw.math.BackgroundList`)
            - ``psfCellSet`` : spatial cells of PSF candidates
                (an `lsst.afw.math.SpatialCellSet`)

        Notes
        -----
        ``detectMeasureAndEstimatePsf`` performs the following operations:

        if the exposure doesn't have a PSF and/or the user wants to make and
        use a simple one:
        - install a simple PSF model (replacing the existing one, if relevant)
        - interpolate over cosmic rays with keepCRs=True
        - estimate background and subtract it from the exposure
        - detect, deblend and measure sources, and subtract a refined
          background model

        if ``config.doMeasurePsf``:
        - measure PSF
        """
        # install a simple PSF model, if needed or wanted
        if not exposure.hasPsf() or (self.config.doMeasurePsf and self.config.useSimplePsf):
            self.log.warn("Source catalog detected and measured with placeholder or default PSF")
            self.installSimplePsf.run(exposure=exposure)

        # run repair, but do not interpolate over cosmic rays
        # (do that later in `run` with the final PSF model)
        self.repair.run(exposure=exposure, keepCRs=True)
        self.display("repair_iter", exposure=exposure)

        if background is None:
            background = BackgroundList()

        sourceIdFactory = IdFactory.makeSource(exposureIdInfo.expId, exposureIdInfo.unusedBits)
        table = SourceTable.make(self.schema, sourceIdFactory)
        table.setMetadata(self.algMetadata)

        detRes = self.detection.run(table=table, exposure=exposure, doSmooth=True)
        sourceCat = detRes.sources
        if detRes.fpSets.background:
            for bg in detRes.fpSets.background:
                background.append(bg)

        if self.config.doDeblend:
            self.deblend.run(exposure=exposure, sources=sourceCat)

        self.measurement.run(measCat=sourceCat, exposure=exposure, exposureId=exposureIdInfo.expId)

        measPsfRes = pipeBase.Struct(cellSet=None)
        if self.config.doMeasurePsf:
            if self.measurePsf.usesMatches:
                matches = self.ref_match.loadAndMatch(exposure=exposure, sourceCat=sourceCat).matches
            else:
                matches = None
            measPsfRes = self.measurePsf.run(exposure=exposure, sources=sourceCat, matches=matches,
                                             expId=exposureIdInfo.expId)
        self.display("measure_iter", exposure=exposure, sourceCat=sourceCat)

        return pipeBase.Struct(
            exposure=exposure,
            sourceCat=sourceCat,
            background=background,
            psfCellSet=measPsfRes.cellSet,
        )

    def getSchemaCatalogs(self):
        """Return a dict of empty catalogs for each catalog dataset produced
        by this task.
        """
        sourceCat = SourceCatalog(self.schema)
        sourceCat.getTable().setMetadata(self.algMetadata)
        return {"icSrc": sourceCat}

    def display(self, itemName, exposure, sourceCat=None):
        """Display exposure and sources on next frame, if display of itemName
        has been requested.

        Parameters
        ----------
        itemName : `str`
            Name of item in debugInfo.
        exposure : `lsst.afw.image.Exposure`
            Exposure to display.
        sourceCat : `lsst.afw.table.SourceCatalog`, optional
            Source catalog to display.
        """
        val = getDebugFrame(self._display, itemName)
        if not val:
            return

        displayAstrometry(exposure=exposure, sourceCat=sourceCat, frame=self._frame, pause=False)
        self._frame += 1
