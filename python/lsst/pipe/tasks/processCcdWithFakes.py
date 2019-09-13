# This file is part of pipe tasks
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Insert fake sources into calexps
"""
from astropy.table import Table

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.daf.base as dafBase

from .insertFakes import InsertFakesTask
from lsst.meas.algorithms import SourceDetectionTask
from lsst.meas.base import (SingleFrameMeasurementTask, ApplyApCorrTask, CatalogCalculationTask,
                            PerTractCcdDataIdContainer)
from lsst.meas.deblender import SourceDeblendTask
from lsst.afw.table import SourceTable, IdFactory
from lsst.obs.base import ExposureIdInfo
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, CmdLineTask, PipelineTaskConnections
import lsst.pipe.base.connectionTypes as cT


__all__ = ["ProcessCcdWithFakesConfig", "ProcessCcdWithFakesTask"]


class ProcessCcdWithFakesConnections(PipelineTaskConnections, dimensions=("instrument", "visit", "detector"),
                                     defaultTemplates={}):

    exposure = cT.Input(
        doc="Exposure into which fakes are to be added.",
        name="calexp",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector")
    )

    fakeCat = cT.Input(
        doc="Catalog of fake sources to draw inputs from.",
        nameTemplate="{CoaddName}Coadd_fakeSourceCat",
        storageClass="Parquet",
        dimensions=("tract", "skymap")
    )

    wcs = cT.Input(
        doc="WCS information for the input exposure.",
        name="jointcal_wcs",
        storageClass="Wcs",
        dimensions=("Tract", "SkyMap", "Instrument", "Visit", "Detector")
    )

    photoCalib = cT.Input(
        doc="Calib information for the input exposure.",
        name="jointcal_photoCalib",
        storageClass="PhotoCalib",
        dimensions=("Tract", "SkyMap", "Instrument", "Visit", "Detector")
    )

    outputExposure = cT.Output(
        doc="Exposure with fake sources added.",
        name="fakes_calexp",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector")
    )

    outputCat = cT.Output(
        doc="Source catalog produced in calibrate task with fakes also measured.",
        name="src",
        storageClass="SourceCatalog",
        dimensions=("instrument", "visit", "detector"),
    )


class ProcessCcdWithFakesConfig(PipelineTaskConfig):
    """Config for inserting fake sources

    Notes
    -----
    The default column names are those from the UW sims database.
    """

    useUpdatedCalibs = pexConfig.Field(
        doc="Use updated calibs and wcs from jointcal?",
        dtype=bool,
        default=False,
    )

    coaddName = pexConfig.Field(
        doc="The name of the type of coadd used",
        dtype=str,
        default="deep",
    )

    insertFakes = pexConfig.ConfigurableField(target=InsertFakesTask,
                                              doc="Configuration for the fake sources")

    detection = pexConfig.ConfigurableField(target=SourceDetectionTask,
                                            doc="The detection task to use.")

    deblend = pexConfig.ConfigurableField(target=SourceDeblendTask, doc="The deblending task to use.")

    measurement = pexConfig.ConfigurableField(target=SingleFrameMeasurementTask,
                                              doc="The measurement task to use")

    applyApCorr = pexConfig.ConfigurableField(target=ApplyApCorrTask,
                                              doc="The apply aperture correction task to use.")

    catalogCalculation = pexConfig.ConfigurableField(target=CatalogCalculationTask,
                                                     doc="The catalog calculation ask to use.")

    def setDefaults(self):
        self.detection.reEstimateBackground = False
        super().setDefaults()
        self.measurement.plugins["base_PixelFlags"].masksFpAnywhere.append("FAKE")
        self.measurement.plugins["base_PixelFlags"].masksFpCenter.append("FAKE")


class ProcessCcdWithFakesTask(PipelineTask, CmdLineTask):
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

    def __init__(self, schema=None, **kwargs):
        """Initalize tings! This should go above in the class docstring
        """

        super().__init__(**kwargs)

        if schema is None:
            schema = SourceTable.makeMinimalSchema()
        self.schema = schema
        self.makeSubtask("insertFakes")
        self.algMetadata = dafBase.PropertyList()
        self.makeSubtask("detection", schema=self.schema)
        self.makeSubtask("deblend", schema=self.schema)
        self.makeSubtask("measurement", schema=self.schema, algMetadata=self.algMetadata)
        self.makeSubtask("applyApCorr", schema=self.schema)
        self.makeSubtask("catalogCalculation", schema=self.schema)

    def runDataRef(self, dataRef):
        """Read in/write out the required data products and add fake sources to the calexp.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.butlerSubset.ButlerDataRef`
            Data reference defining the ccd to have fakes added to it.
            Used to access the following data products:
                calexp
                jointcal_wcs
                jointcal_photoCalib

        Notes
        -----
        Uses the calibration and WCS information attached to the calexp for the posistioning and calibration
        of the sources unless the config option config.useUpdatedCalibs is set then it uses the
        meas_mosaic/jointCal outputs. The config defualts for the column names in the catalog of fakes are
        taken from the University of Washington simulations database. Operates on one ccd at a time.
        """
        exposureIdInfo = dataRef.get("expIdInfo")

        if self.config.insertFakes.fakeType == "snapshot":
            fakeCat = dataRef.get("fakeSourceCat").toDataFrame()
        elif self.config.insertFakes.fakeType == "static":
            fakeCat = dataRef.get("deepCoadd_fakeSourceCat").toDataFrame()
        else:
            fakeCat = Table.read(self.config.insertFakes.fakeType).to_pandas()

        calexp = dataRef.get("calexp")
        if self.config.useUpdatedCalibs:
            self.log.info("Using updated calibs from meas_mosaic/jointCal")
            wcs = dataRef.get("jointcal_wcs")
            photoCalib = dataRef.get("jointcal_photoCalib")
        else:
            wcs = calexp.getWcs()
            photoCalib = calexp.getCalib()

        resultStruct = self.run(fakeCat, calexp, wcs=wcs, photoCalib=photoCalib,
                                exposureIdInfo=exposureIdInfo)

        dataRef.put(resultStruct.outputExposure, "fakes_calexp")
        dataRef.put(resultStruct.outputCat, "fakes_src")

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        if 'exposureIdInfo' not in inputs.keys():
            expId, expBits = butlerQC.quantum.dataId.pack("visit_detector", returnMaxBits=True)
            inputs['exposureIdInfo'] = ExposureIdInfo(expId, expBits)

        if inputs["wcs"] is None:
            inputs["wcs"] = inputs["image"].getWcs()
        if inputs["photoCalib"] is None:
            inputs["photoCalib"] = inputs["image"].getCalib()

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    @classmethod
    def _makeArgumentParser(cls):
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "fakes_calexp", help="data ID with raw CCD keys [+ tract optionally], "
                               "e.g. --id visit=12345 ccd=1,2 [tract=0]",
                               ContainerClass=PerTractCcdDataIdContainer)
        return parser

    def run(self, fakeCat, exposure, wcs=None, photoCalib=None, exposureIdInfo=None):
        """Add fake sources to a calexp and then run detection, deblending and measurement.

        Parameters
        ----------
        fakeCat : `pandas.core.frame.DataFrame`
                    The catalog of fake sources to add to the exposure
        exposure : `lsst.afw.image.exposure.exposure.ExposureF`
                    The exposure to add the fake sources to
        wcs : `lsst.afw.geom.SkyWcs`
                    WCS to use to add fake sources
        photoCalib : `lsst.afw.image.photoCalib.PhotoCalib`
                    Photometric calibration to be used to calibrate the fake sources
        exposureIdInfo : `lsst.obs.base.ExposureIdInfo`

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

        if wcs is None:
            wcs = exposure.getWcs()

        if photoCalib is None:
            photoCalib = exposure.getCalib()

        self.insertFakes.run(fakeCat, exposure, wcs, photoCalib)

        # detect, deblend and measure sources
        if exposureIdInfo is None:
            exposureIdInfo = ExposureIdInfo()

        sourceIdFactory = IdFactory.makeSource(exposureIdInfo.expId, exposureIdInfo.unusedBits)
        table = SourceTable.make(self.schema, sourceIdFactory)
        table.setMetadata(self.algMetadata)

        detRes = self.detection.run(table=table, exposure=exposure, doSmooth=True)
        sourceCat = detRes.sources
        self.deblend.run(exposure=exposure, sources=sourceCat)
        self.measurement.run(measCat=sourceCat, exposure=exposure, exposureId=exposureIdInfo.expId)
        self.applyApCorr.run(catalog=sourceCat, apCorrMap=exposure.getInfo().getApCorrMap())
        self.catalogCalculation.run(sourceCat)

        resultStruct = pipeBase.Struct(outputExposure=exposure, outputCat=sourceCat)
        return resultStruct
