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

"""
Insert fake sources into calexps
"""

__all__ = ["ProcessCcdWithFakesConfig", "ProcessCcdWithFakesTask",
           "ProcessCcdWithVariableFakesConfig", "ProcessCcdWithVariableFakesTask"]

import numpy as np
import pandas as pd

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

from .insertFakes import InsertFakesTask
from lsst.afw.table import SourceTable
from lsst.obs.base import ExposureIdInfo
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections
import lsst.pipe.base.connectionTypes as cT
import lsst.afw.table as afwTable
from lsst.skymap import BaseSkyMap
from lsst.pipe.tasks.calibrate import CalibrateTask


class ProcessCcdWithFakesConnections(PipelineTaskConnections,
                                     dimensions=("instrument", "visit", "detector"),
                                     defaultTemplates={"coaddName": "deep",
                                                       "wcsName": "gbdesAstrometricFit",
                                                       "photoCalibName": "jointcal",
                                                       "fakesType": "fakes_"}):
    skyMap = cT.Input(
        doc="Input definition of geometry/bbox and projection/wcs for "
        "template exposures. Needed to test which tract to generate ",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        dimensions=("skymap",),
        storageClass="SkyMap",
    )

    exposure = cT.Input(
        doc="Exposure into which fakes are to be added.",
        name="calexp",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector")
    )

    fakeCats = cT.Input(
        doc="Set of catalogs of fake sources to draw inputs from. We "
            "concatenate the tract catalogs for detectorVisits that cover "
            "multiple tracts.",
        name="{fakesType}fakeSourceCat",
        storageClass="DataFrame",
        dimensions=("tract", "skymap"),
        deferLoad=True,
        multiple=True,
    )

    externalSkyWcsTractCatalog = cT.Input(
        doc=("Per-tract, per-visit wcs calibrations. These catalogs use the detector "
             "id for the catalog id, sorted on id for fast lookup."),
        name="{wcsName}SkyWcsCatalog",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit", "tract", "skymap"),
        deferLoad=True,
        multiple=True,
    )

    externalSkyWcsGlobalCatalog = cT.Input(
        doc=("Per-visit wcs calibrations computed globally (with no tract information). "
             "These catalogs use the detector id for the catalog id, sorted on id for "
             "fast lookup."),
        name="finalVisitSummary",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit"),
    )

    externalPhotoCalibTractCatalog = cT.Input(
        doc=("Per-tract, per-visit photometric calibrations. These catalogs use the "
             "detector id for the catalog id, sorted on id for fast lookup."),
        name="{photoCalibName}PhotoCalibCatalog",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit", "tract"),
        deferLoad=True,
        multiple=True,
    )

    externalPhotoCalibGlobalCatalog = cT.Input(
        doc=("Per-visit photometric calibrations. These catalogs use the "
             "detector id for the catalog id, sorted on id for fast lookup."),
        name="finalVisitSummary",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit"),
    )

    icSourceCat = cT.Input(
        doc="Catalog of calibration sources",
        name="icSrc",
        storageClass="SourceCatalog",
        dimensions=("instrument", "visit", "detector")
    )

    sfdSourceCat = cT.Input(
        doc="Catalog of calibration sources",
        name="src",
        storageClass="SourceCatalog",
        dimensions=("instrument", "visit", "detector")
    )

    outputExposure = cT.Output(
        doc="Exposure with fake sources added.",
        name="{fakesType}calexp",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector")
    )

    outputCat = cT.Output(
        doc="Source catalog produced in calibrate task with fakes also measured.",
        name="{fakesType}src",
        storageClass="SourceCatalog",
        dimensions=("instrument", "visit", "detector"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if not config.doApplyExternalGlobalPhotoCalib:
            self.inputs.remove("externalPhotoCalibGlobalCatalog")
        if not config.doApplyExternalTractPhotoCalib:
            self.inputs.remove("externalPhotoCalibTractCatalog")

        if not config.doApplyExternalGlobalSkyWcs:
            self.inputs.remove("externalSkyWcsGlobalCatalog")
        if not config.doApplyExternalTractSkyWcs:
            self.inputs.remove("externalSkyWcsTractCatalog")


class ProcessCcdWithFakesConfig(PipelineTaskConfig,
                                pipelineConnections=ProcessCcdWithFakesConnections):
    """Config for inserting fake sources

    Notes
    -----
    The default column names are those from the UW sims database.
    """

    doApplyExternalGlobalPhotoCalib = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Whether to apply an external photometric calibration via an "
            "`lsst.afw.image.PhotoCalib` object. Uses the "
            "`externalPhotoCalibName` config option to determine which "
            "calibration to use. Uses a global calibration."
    )

    doApplyExternalTractPhotoCalib = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Whether to apply an external photometric calibration via an "
            "`lsst.afw.image.PhotoCalib` object. Uses the "
            "`externalPhotoCalibName` config option to determine which "
            "calibration to use. Uses a per tract calibration."
    )

    externalPhotoCalibName = pexConfig.ChoiceField(
        doc="What type of external photo calib to use.",
        dtype=str,
        default="jointcal",
        allowed={"jointcal": "Use jointcal_photoCalib",
                 "fgcm": "Use fgcm_photoCalib",
                 "fgcm_tract": "Use fgcm_tract_photoCalib"}
    )

    doApplyExternalGlobalSkyWcs = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Whether to apply an external astrometric calibration via an "
            "`lsst.afw.geom.SkyWcs` object. Uses the "
            "`externalSkyWcsName` config option to determine which "
            "calibration to use. Uses a global calibration."
    )

    doApplyExternalTractSkyWcs = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Whether to apply an external astrometric calibration via an "
            "`lsst.afw.geom.SkyWcs` object. Uses the "
            "`externalSkyWcsName` config option to determine which "
            "calibration to use. Uses a per tract calibration."
    )

    externalSkyWcsName = pexConfig.ChoiceField(
        doc="What type of updated WCS calib to use.",
        dtype=str,
        default="gbdesAstrometricFit",
        allowed={"gbdesAstrometricFit": "Use gbdesAstrometricFit_wcs"}
    )

    coaddName = pexConfig.Field(
        doc="The name of the type of coadd used",
        dtype=str,
        default="deep",
    )

    srcFieldsToCopy = pexConfig.ListField(
        dtype=str,
        default=("calib_photometry_reserved", "calib_photometry_used", "calib_astrometry_used",
                 "calib_psf_candidate", "calib_psf_used", "calib_psf_reserved"),
        doc=("Fields to copy from the `src` catalog to the output catalog "
             "for matching sources Any missing fields will trigger a "
             "RuntimeError exception.")
    )

    matchRadiusPix = pexConfig.Field(
        dtype=float,
        default=3,
        doc=("Match radius for matching icSourceCat objects to sourceCat objects (pixels)"),
    )

    doMatchVisit = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Match visit to trim the fakeCat"
    )

    calibrate = pexConfig.ConfigurableField(target=CalibrateTask,
                                            doc="The calibration task to use.")

    insertFakes = pexConfig.ConfigurableField(target=InsertFakesTask,
                                              doc="Configuration for the fake sources")

    def setDefaults(self):
        super().setDefaults()
        self.calibrate.measurement.plugins["base_PixelFlags"].masksFpAnywhere.append("FAKE")
        self.calibrate.measurement.plugins["base_PixelFlags"].masksFpCenter.append("FAKE")
        self.calibrate.doAstrometry = False
        self.calibrate.doWriteMatches = False
        self.calibrate.doPhotoCal = False
        self.calibrate.doComputeSummaryStats = False
        self.calibrate.detection.reEstimateBackground = False


class ProcessCcdWithFakesTask(PipelineTask):
    """Insert fake objects into calexps.

    Add fake stars and galaxies to the given calexp, specified in the dataRef. Galaxy parameters are read in
    from the specified file and then modelled using galsim. Re-runs characterize image and calibrate image to
    give a new background estimation and measurement of the calexp.

    `ProcessFakeSourcesTask` inherits six functions from insertFakesTask that make images of the fake
    sources and then add them to the calexp.

    `addPixCoords`
        Use the WCS information to add the pixel coordinates of each source
        Adds an ``x`` and ``y`` column to the catalog of fake sources.
    `trimFakeCat`
        Trim the fake cat to about the size of the input image.
    `mkFakeGalsimGalaxies`
        Use Galsim to make fake double sersic galaxies for each set of galaxy parameters in the input file.
    `mkFakeStars`
        Use the PSF information from the calexp to make a fake star using the magnitude information from the
        input file.
    `cleanCat`
        Remove rows of the input fake catalog which have half light radius, of either the bulge or the disk,
        that are 0.
    `addFakeSources`
        Add the fake sources to the calexp.

    Notes
    -----
    The ``calexp`` with fake souces added to it is written out as the datatype ``calexp_fakes``.
    """

    _DefaultName = "processCcdWithFakes"
    ConfigClass = ProcessCcdWithFakesConfig

    def __init__(self, schema=None, butler=None, **kwargs):
        """Initalize things! This should go above in the class docstring
        """

        super().__init__(**kwargs)

        if schema is None:
            schema = SourceTable.makeMinimalSchema()
        self.schema = schema
        self.makeSubtask("insertFakes")
        self.makeSubtask("calibrate")

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        detectorId = inputs["exposure"].getInfo().getDetector().getId()

        if 'exposureIdInfo' not in inputs.keys():
            expId, expBits = butlerQC.quantum.dataId.pack("visit_detector", returnMaxBits=True)
            inputs['exposureIdInfo'] = ExposureIdInfo(expId, expBits)

        expWcs = inputs["exposure"].getWcs()
        tractId = None
        if not self.config.doApplyExternalGlobalSkyWcs and not self.config.doApplyExternalTractSkyWcs:
            if expWcs is None:
                self.log.info("No WCS for exposure %s so cannot insert fake sources.  Skipping detector.",
                              butlerQC.quantum.dataId)
                return None
            else:
                inputs["wcs"] = expWcs
        elif self.config.doApplyExternalGlobalSkyWcs:
            externalSkyWcsCatalog = inputs["externalSkyWcsGlobalCatalog"]
            row = externalSkyWcsCatalog.find(detectorId)
            if row is None:
                self.log.info("No %s external global sky WCS for exposure %s so cannot insert fake "
                              "sources.  Skipping detector.", self.config.externalSkyWcsName,
                              butlerQC.quantum.dataId)
                return None
            inputs["wcs"] = row.getWcs()
        elif self.config.doApplyExternalTractSkyWcs:
            externalSkyWcsCatalogList = inputs["externalSkyWcsTractCatalog"]
            if tractId is None:
                tractId = externalSkyWcsCatalogList[0].dataId["tract"]
            externalSkyWcsCatalog = None
            for externalSkyWcsCatalogRef in externalSkyWcsCatalogList:
                if externalSkyWcsCatalogRef.dataId["tract"] == tractId:
                    externalSkyWcsCatalog = externalSkyWcsCatalogRef.get()
                    break
            if externalSkyWcsCatalog is None:
                usedTract = externalSkyWcsCatalogList[-1].dataId["tract"]
                self.log.warn(
                    f"Warning, external SkyWcs for tract {tractId} not found. Using tract {usedTract} "
                    "instead.")
                externalSkyWcsCatalog = externalSkyWcsCatalogList[-1].get()
            row = externalSkyWcsCatalog.find(detectorId)
            if row is None:
                self.log.info("No %s external tract sky WCS for exposure %s so cannot insert fake "
                              "sources.  Skipping detector.", self.config.externalSkyWcsName,
                              butlerQC.quantum.dataId)
                return None
            inputs["wcs"] = row.getWcs()

        if not self.config.doApplyExternalGlobalPhotoCalib and not self.config.doApplyExternalTractPhotoCalib:
            inputs["photoCalib"] = inputs["exposure"].getPhotoCalib()
        elif self.config.doApplyExternalGlobalPhotoCalib:
            externalPhotoCalibCatalog = inputs["externalPhotoCalibGlobalCatalog"]
            row = externalPhotoCalibCatalog.find(detectorId)
            if row is None:
                self.log.info("No %s external global photoCalib for exposure %s so cannot insert fake "
                              "sources.  Skipping detector.", self.config.externalPhotoCalibName,
                              butlerQC.quantum.dataId)
                return None
            inputs["photoCalib"] = row.getPhotoCalib()
        elif self.config.doApplyExternalTractPhotoCalib:
            externalPhotoCalibCatalogList = inputs["externalPhotoCalibTractCatalog"]
            if tractId is None:
                tractId = externalPhotoCalibCatalogList[0].dataId["tract"]
            externalPhotoCalibCatalog = None
            for externalPhotoCalibCatalogRef in externalPhotoCalibCatalogList:
                if externalPhotoCalibCatalogRef.dataId["tract"] == tractId:
                    externalPhotoCalibCatalog = externalPhotoCalibCatalogRef.get()
                    break
            if externalPhotoCalibCatalog is None:
                usedTract = externalPhotoCalibCatalogList[-1].dataId["tract"]
                self.log.warn(
                    f"Warning, external PhotoCalib for tract {tractId} not found. Using tract {usedTract} "
                    "instead.")
                externalPhotoCalibCatalog = externalPhotoCalibCatalogList[-1].get()
            row = externalPhotoCalibCatalog.find(detectorId)
            if row is None:
                self.log.info("No %s external tract photoCalib for exposure %s so cannot insert fake "
                              "sources.  Skipping detector.", self.config.externalPhotoCalibName,
                              butlerQC.quantum.dataId)
                return None
            inputs["photoCalib"] = row.getPhotoCalib()

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, fakeCats, exposure, skyMap, wcs=None, photoCalib=None, exposureIdInfo=None,
            icSourceCat=None, sfdSourceCat=None, externalSkyWcsGlobalCatalog=None,
            externalSkyWcsTractCatalog=None, externalPhotoCalibGlobalCatalog=None,
            externalPhotoCalibTractCatalog=None):
        """Add fake sources to a calexp and then run detection, deblending and measurement.

        Parameters
        ----------
        fakeCats : `list` of `lsst.daf.butler.DeferredDatasetHandle`
                    Set of tract level fake catalogs that potentially cover
                    this detectorVisit.
        exposure : `lsst.afw.image.exposure.exposure.ExposureF`
                    The exposure to add the fake sources to
        skyMap : `lsst.skymap.SkyMap`
            SkyMap defining the tracts and patches the fakes are stored over.
        wcs : `lsst.afw.geom.SkyWcs`
                    WCS to use to add fake sources
        photoCalib : `lsst.afw.image.photoCalib.PhotoCalib`
                    Photometric calibration to be used to calibrate the fake sources
        exposureIdInfo : `lsst.obs.base.ExposureIdInfo`
        icSourceCat : `lsst.afw.table.SourceCatalog`
                    Default : None
                    Catalog to take the information about which sources were used for calibration from.
        sfdSourceCat : `lsst.afw.table.SourceCatalog`
                    Default : None
                    Catalog produced by singleFrameDriver, needed to copy some calibration flags from.

        Returns
        -------
        resultStruct : `lsst.pipe.base.struct.Struct`
            contains : outputExposure : `lsst.afw.image.exposure.exposure.ExposureF`
                       outputCat : `lsst.afw.table.source.source.SourceCatalog`

        Notes
        -----
        Adds pixel coordinates for each source to the fakeCat and removes objects with bulge or disk half
        light radius = 0 (if ``config.cleanCat = True``). These columns are called ``x`` and ``y`` and are in
        pixels.

        Adds the ``Fake`` mask plane to the exposure which is then set by `addFakeSources` to mark where fake
        sources have been added. Uses the information in the ``fakeCat`` to make fake galaxies (using galsim)
        and fake stars, using the PSF models from the PSF information for the calexp. These are then added to
        the calexp and the calexp with fakes included returned.

        The galsim galaxies are made using a double sersic profile, one for the bulge and one for the disk,
        this is then convolved with the PSF at that point.

        If exposureIdInfo is not provided then the SourceCatalog IDs will not be globally unique.
        """
        fakeCat = self.composeFakeCat(fakeCats, skyMap)

        if wcs is None:
            wcs = exposure.getWcs()

        if photoCalib is None:
            photoCalib = exposure.getPhotoCalib()

        if self.config.doMatchVisit:
            fakeCat = self.getVisitMatchedFakeCat(fakeCat, exposure)

        self.insertFakes.run(fakeCat, exposure, wcs, photoCalib)

        # detect, deblend and measure sources
        if exposureIdInfo is None:
            exposureIdInfo = ExposureIdInfo()
        returnedStruct = self.calibrate.run(exposure, exposureIdInfo=exposureIdInfo)
        sourceCat = returnedStruct.sourceCat

        sourceCat = self.copyCalibrationFields(sfdSourceCat, sourceCat, self.config.srcFieldsToCopy)

        resultStruct = pipeBase.Struct(outputExposure=exposure, outputCat=sourceCat)
        return resultStruct

    def composeFakeCat(self, fakeCats, skyMap):
        """Concatenate the fakeCats from tracts that may cover the exposure.

        Parameters
        ----------
        fakeCats : `list` of `lsst.daf.butler.DeferredDatasetHandle`
            Set of fake cats to concatenate.
        skyMap : `lsst.skymap.SkyMap`
            SkyMap defining the geometry of the tracts and patches.

        Returns
        -------
        combinedFakeCat : `pandas.DataFrame`
            All fakes that cover the inner polygon of the tracts in this
            quantum.
        """
        if len(fakeCats) == 1:
            return fakeCats[0].get()
        outputCat = []
        for fakeCatRef in fakeCats:
            cat = fakeCatRef.get()
            tractId = fakeCatRef.dataId["tract"]
            # Make sure all data is within the inner part of the tract.
            outputCat.append(cat[
                skyMap.findTractIdArray(cat[self.config.insertFakes.ra_col],
                                        cat[self.config.insertFakes.dec_col],
                                        degrees=False)
                == tractId])

        return pd.concat(outputCat)

    def getVisitMatchedFakeCat(self, fakeCat, exposure):
        """Trim the fakeCat to select particular visit

        Parameters
        ----------
        fakeCat : `pandas.core.frame.DataFrame`
                    The catalog of fake sources to add to the exposure
        exposure : `lsst.afw.image.exposure.exposure.ExposureF`
                    The exposure to add the fake sources to

        Returns
        -------
        movingFakeCat : `pandas.DataFrame`
            All fakes that belong to the visit
        """
        selected = exposure.getInfo().getVisitInfo().getId() == fakeCat["visit"]

        return fakeCat[selected]

    def copyCalibrationFields(self, calibCat, sourceCat, fieldsToCopy):
        """Match sources in calibCat and sourceCat and copy the specified fields

        Parameters
        ----------
        calibCat : `lsst.afw.table.SourceCatalog`
            Catalog from which to copy fields.
        sourceCat : `lsst.afw.table.SourceCatalog`
            Catalog to which to copy fields.
        fieldsToCopy : `lsst.pex.config.listField.List`
            Fields to copy from calibCat to SoourceCat.

        Returns
        -------
        newCat : `lsst.afw.table.SourceCatalog`
            Catalog which includes the copied fields.

        The fields copied are those specified by `fieldsToCopy` that actually exist
        in the schema of `calibCat`.

        This version was based on and adapted from the one in calibrateTask.
        """

        # Make a new SourceCatalog with the data from sourceCat so that we can add the new columns to it
        sourceSchemaMapper = afwTable.SchemaMapper(sourceCat.schema)
        sourceSchemaMapper.addMinimalSchema(sourceCat.schema, True)

        calibSchemaMapper = afwTable.SchemaMapper(calibCat.schema, sourceCat.schema)

        # Add the desired columns from the option fieldsToCopy
        missingFieldNames = []
        for fieldName in fieldsToCopy:
            if fieldName in calibCat.schema:
                schemaItem = calibCat.schema.find(fieldName)
                calibSchemaMapper.editOutputSchema().addField(schemaItem.getField())
                schema = calibSchemaMapper.editOutputSchema()
                calibSchemaMapper.addMapping(schemaItem.getKey(), schema.find(fieldName).getField())
            else:
                missingFieldNames.append(fieldName)
        if missingFieldNames:
            raise RuntimeError(f"calibCat is missing fields {missingFieldNames} specified in "
                               "fieldsToCopy")

        if "calib_detected" not in calibSchemaMapper.getOutputSchema():
            self.calibSourceKey = calibSchemaMapper.addOutputField(afwTable.Field["Flag"]("calib_detected",
                                                                   "Source was detected as an icSource"))
        else:
            self.calibSourceKey = None

        schema = calibSchemaMapper.getOutputSchema()
        newCat = afwTable.SourceCatalog(schema)
        newCat.reserve(len(sourceCat))
        newCat.extend(sourceCat, sourceSchemaMapper)

        # Set the aliases so it doesn't complain.
        for k, v in sourceCat.schema.getAliasMap().items():
            newCat.schema.getAliasMap().set(k, v)

        select = newCat["deblend_nChild"] == 0
        matches = afwTable.matchXy(newCat[select], calibCat, self.config.matchRadiusPix)
        # Check that no sourceCat sources are listed twice (we already know
        # that each match has a unique calibCat source ID, due to using
        # that ID as the key in bestMatches)
        numMatches = len(matches)
        numUniqueSources = len(set(m[1].getId() for m in matches))
        if numUniqueSources != numMatches:
            self.log.warning("%d calibCat sources matched only %d sourceCat sources", numMatches,
                             numUniqueSources)

        self.log.info("Copying flags from calibCat to sourceCat for %s sources", numMatches)

        # For each match: set the calibSourceKey flag and copy the desired
        # fields
        for src, calibSrc, d in matches:
            if self.calibSourceKey:
                src.setFlag(self.calibSourceKey, True)
            # src.assign copies the footprint from calibSrc, which we don't want
            # (DM-407)
            # so set calibSrc's footprint to src's footprint before src.assign,
            # then restore it
            calibSrcFootprint = calibSrc.getFootprint()
            try:
                calibSrc.setFootprint(src.getFootprint())
                src.assign(calibSrc, calibSchemaMapper)
            finally:
                calibSrc.setFootprint(calibSrcFootprint)

        return newCat


class ProcessCcdWithVariableFakesConnections(ProcessCcdWithFakesConnections):
    ccdVisitFakeMagnitudes = cT.Output(
        doc="Catalog of fakes with magnitudes scattered for this ccdVisit.",
        name="{fakesType}ccdVisitFakeMagnitudes",
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "detector"),
    )


class ProcessCcdWithVariableFakesConfig(ProcessCcdWithFakesConfig,
                                        pipelineConnections=ProcessCcdWithVariableFakesConnections):
    scatterSize = pexConfig.RangeField(
        dtype=float,
        default=0.4,
        min=0,
        max=100,
        doc="Amount of scatter to add to the visit magnitude for variable "
            "sources."
    )


class ProcessCcdWithVariableFakesTask(ProcessCcdWithFakesTask):
    """As ProcessCcdWithFakes except add variablity to the fakes catalog
    magnitude in the observed band for this ccdVisit.

    Additionally, write out the modified magnitudes to the Butler.
    """

    _DefaultName = "processCcdWithVariableFakes"
    ConfigClass = ProcessCcdWithVariableFakesConfig

    def run(self, fakeCats, exposure, skyMap, wcs=None, photoCalib=None, exposureIdInfo=None,
            icSourceCat=None, sfdSourceCat=None):
        """Add fake sources to a calexp and then run detection, deblending and measurement.

        Parameters
        ----------
        fakeCat : `pandas.core.frame.DataFrame`
                    The catalog of fake sources to add to the exposure
        exposure : `lsst.afw.image.exposure.exposure.ExposureF`
                    The exposure to add the fake sources to
        skyMap : `lsst.skymap.SkyMap`
            SkyMap defining the tracts and patches the fakes are stored over.
        wcs : `lsst.afw.geom.SkyWcs`
                    WCS to use to add fake sources
        photoCalib : `lsst.afw.image.photoCalib.PhotoCalib`
                    Photometric calibration to be used to calibrate the fake sources
        exposureIdInfo : `lsst.obs.base.ExposureIdInfo`
        icSourceCat : `lsst.afw.table.SourceCatalog`
                    Default : None
                    Catalog to take the information about which sources were used for calibration from.
        sfdSourceCat : `lsst.afw.table.SourceCatalog`
                    Default : None
                    Catalog produced by singleFrameDriver, needed to copy some calibration flags from.

        Returns
        -------
        resultStruct : `lsst.pipe.base.struct.Struct`
            Results Strcut containing:

            - outputExposure : Exposure with added fakes
              (`lsst.afw.image.exposure.exposure.ExposureF`)
            - outputCat : Catalog with detected fakes
              (`lsst.afw.table.source.source.SourceCatalog`)
            - ccdVisitFakeMagnitudes : Magnitudes that these fakes were
              inserted with after being scattered (`pandas.DataFrame`)

        Notes
        -----
        Adds pixel coordinates for each source to the fakeCat and removes objects with bulge or disk half
        light radius = 0 (if ``config.cleanCat = True``). These columns are called ``x`` and ``y`` and are in
        pixels.

        Adds the ``Fake`` mask plane to the exposure which is then set by `addFakeSources` to mark where fake
        sources have been added. Uses the information in the ``fakeCat`` to make fake galaxies (using galsim)
        and fake stars, using the PSF models from the PSF information for the calexp. These are then added to
        the calexp and the calexp with fakes included returned.

        The galsim galaxies are made using a double sersic profile, one for the bulge and one for the disk,
        this is then convolved with the PSF at that point.

        If exposureIdInfo is not provided then the SourceCatalog IDs will not be globally unique.
        """
        fakeCat = self.composeFakeCat(fakeCats, skyMap)

        if wcs is None:
            wcs = exposure.getWcs()

        if photoCalib is None:
            photoCalib = exposure.getPhotoCalib()

        if exposureIdInfo is None:
            exposureIdInfo = ExposureIdInfo()

        band = exposure.getFilter().bandLabel
        ccdVisitMagnitudes = self.addVariablity(fakeCat, band, exposure, photoCalib, exposureIdInfo)

        self.insertFakes.run(fakeCat, exposure, wcs, photoCalib)

        # detect, deblend and measure sources
        returnedStruct = self.calibrate.run(exposure, exposureIdInfo=exposureIdInfo)
        sourceCat = returnedStruct.sourceCat

        sourceCat = self.copyCalibrationFields(sfdSourceCat, sourceCat, self.config.srcFieldsToCopy)

        resultStruct = pipeBase.Struct(outputExposure=exposure,
                                       outputCat=sourceCat,
                                       ccdVisitFakeMagnitudes=ccdVisitMagnitudes)
        return resultStruct

    def addVariablity(self, fakeCat, band, exposure, photoCalib, exposureIdInfo):
        """Add scatter to the fake catalog visit magnitudes.

        Currently just adds a simple Gaussian scatter around the static fake
        magnitude. This function could be modified to return any number of
        fake variability.

        Parameters
        ----------
        fakeCat : `pandas.DataFrame`
            Catalog of fakes to modify magnitudes of.
        band : `str`
            Current observing band to modify.
        exposure : `lsst.afw.image.ExposureF`
            Exposure fakes will be added to.
        photoCalib : `lsst.afw.image.PhotoCalib`
            Photometric calibration object of ``exposure``.
        exposureIdInfo : `lsst.obs.base.ExposureIdInfo`
            Exposure id information and metadata.

        Returns
        -------
        dataFrame : `pandas.DataFrame`
            DataFrame containing the values of the magnitudes to that will
            be inserted into this ccdVisit.
        """
        expId = exposureIdInfo.expId
        rng = np.random.default_rng(expId)
        magScatter = rng.normal(loc=0,
                                scale=self.config.scatterSize,
                                size=len(fakeCat))
        visitMagnitudes = fakeCat[self.insertFakes.config.mag_col % band] + magScatter
        fakeCat.loc[:, self.insertFakes.config.mag_col % band] = visitMagnitudes
        return pd.DataFrame(data={"variableMag": visitMagnitudes})
