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

import os
import unittest

import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.geom
import lsst.meas.algorithms as measAlgorithms
import lsst.meas.base as measBase
import lsst.meas.extensions.shapeHSM  # To register its plugins
import lsst.pipe.base as pipeBase
import lsst.utils.tests
import numpy as np
from astropy.table import Table as astropyTable
from lsst.afw.math import ChebyshevBoundedField
from lsst.meas.base.tests import TestDataset
from lsst.pipe.tasks.measurementDriver import (
    ForcedMeasurementDriverTask,
    MultiBandMeasurementDriverTask,
    SingleBandMeasurementDriverTask,
)
from lsst.scarlet.lite.io import ScarletModelData

TESTDIR = os.path.abspath(os.path.dirname(__file__))


class MeasurementDriverTestCase(lsst.utils.tests.TestCase):
    """A test case for interpolation."""

    def setUp(self):
        # Set image size and edge width.
        self.nx = 100
        self.ny = 120
        self.edgeWidth = 4

        # Set background noise in units of counts.
        self.bgStdev = 20.0

        # Set the inner and outer radii for background annulus as multiples of
        # the PSF sigma.
        self.annulusInner = 5
        self.annulusOuter = 15

        # Set the name of the local background algorithm.
        self.localBgAlgName = "base_LocalBackground"

        # Photometric calibration parameters.
        self.meanCalibration = 1e-4
        self.calibrationErr = 1e-5

        # Sigma of the Gaussian PSF attached to the image in pixels.
        self.psfSigma = 2.5

        # Make a test dataset.
        self._makeTestData()

    def tearDown(self):
        del self.dataset
        del self.datasets

    def testSingleBandMeasurementDriver(self):
        """Test the single-band measurement driver."""

        # Confirm that the dataset has the expected PSF size.
        self.assertEqual(self.dataset.psfShape.getDeterminantRadius(), self.psfSigma)

        # Create a driver instance with the default configuration.
        # (Alternatively, we could create the config first and then pass it to
        # the driver, see the next test.)
        driver = SingleBandMeasurementDriverTask()

        # Access the configuration through the driver instance to modify it.
        driver.config.doScaleVariance = False
        driver.config.doDetect = True
        driver.config.doDeblend = True
        driver.config.doMeasure = True
        driver.config.doApCorr = True
        driver.config.doRunCatalogCalculation = True
        driver.config.detection.thresholdType = "pixel_stdev"
        driver.config.detection.thresholdValue = 5.0
        driver.config.detection.background.useApprox = False
        driver.config.deblend.tinyFootprintSize = 5
        driver.config.measurement.plugins.names |= [
            "base_SdssCentroid",
            "base_SdssShape",
            "ext_shapeHSM_HsmSourceMoments",
            self.localBgAlgName,
        ]
        driver.config.measurement.plugins[self.localBgAlgName].annulusInner = self.annulusInner
        driver.config.measurement.plugins[self.localBgAlgName].annulusOuter = self.annulusOuter
        driver.config.measurement.doReplaceWithNoise = True

        # Run the single-band driver on the test dataset.
        result = driver.run(self.dataset.exposure)

        # Check the background measurements. The flux should be near zero since
        # we started with a background-subtracted image and the error should
        # match expected noise level.
        bgValues = result.catalog.get(self.localBgAlgName + "_instFlux")
        bgStdevs = result.catalog.get(self.localBgAlgName + "_instFluxErr")
        for bgValue, bgStdev in zip(bgValues, bgStdevs):
            self.assertFloatsAlmostEqual(bgValue, 0.0, atol=3.0)
            self.assertFloatsAlmostEqual(bgStdev, self.bgStdev, rtol=7e-2)

        # Some key assertions to verify the result.
        self.assertIsInstance(result, pipeBase.Struct, "Expected a Struct result")
        self.assertTrue(hasattr(result, "catalog"), "Expected 'catalog' in result")
        self.assertTrue(hasattr(result, "backgroundList"), "Expected 'backgroundList' in result")
        self.assertIsInstance(result.catalog, afwTable.SourceCatalog, "Incorrect catalog type")
        self.assertIsInstance(result.backgroundList, afwMath.BackgroundList, "Incorrect backgroundList type")
        self.assertEqual(result.catalog.schema, driver.schema, "Catalog schema mismatch")
        self.assertEqual(len(result.catalog), 5, "Expected 5 sources in catalog")
        self.assertEqual(sum(result.catalog["parent"] == 0), 2, "Expected 2 parents in catalog")
        self.assertEqual(sum(result.catalog["parent"] != 0), 3, "Expected 3 children in catalog")
        self.assertEqual(result.catalog["deblend_nChild"].tolist(), [3, 0, 0, 0, 0])
        self.assertEqual(sum(result.catalog["parent"] == 1), 3, "Expected 3 children of parent 1 in catalog")
        self.assertFalse(
            result.catalog["base_SdssShape_flag"].any(), "Expected no objects with shape flag set"
        )
        self.assertTrue(
            np.allclose(
                result.catalog["ext_shapeHSM_HsmSourceMoments_xy"],
                [-0.01744399, 0.97273821, -0.00284368, 1.5027978, 0.00851671],
            )
        )

        # Re-run the driver with deblending disabled.
        driver.config.doDeblend = False
        result = driver.run(self.dataset.exposure)
        self.assertEqual(len(result.catalog), 2, "Expected 2 sources in catalog")
        self.assertEqual(sum(result.catalog["parent"] != 0), 0, "Expected no children in catalog")
        self.assertNotIn("deblend_nChild", result.catalog.schema, "Expected no deblend_nChild in catalog")

        # Re-run the driver with deblending and measurement disabled.
        driver.config.doDetect = True
        driver.config.doDeblend = False
        driver.config.doMeasure = False
        driver.config.doApCorr = False
        driver.config.doRunCatalogCalculation = False
        # The line below is only allowed when doRunCatalogCalculation is False.
        driver.config.measurement.slots.psfFlux = None
        result = driver.run(self.dataset.exposure)
        self.assertEqual(len(result.catalog), 2, "Expected 2 sources in catalog")
        self.assertEqual(sum(result.catalog["parent"] != 0), 0, "Expected no children in catalog")
        self.assertNotIn("base_SdssShape_flag", result.catalog.schema, "Expected no shape flag in catalog")
        self.assertNotIn("ext_shapeHSM_HsmSourceMoments_xx", result.catalog.schema, "No HSM moments expected")

        # Re-run the driver with deblending enabled again; this time detection
        # is disabled and a detection catalog is provided instead.
        driver.config.doDetect = False
        driver.config.doDeblend = True
        driver.config.doMeasure = True
        table = afwTable.SourceTable.make(
            afwTable.SourceTable.makeMinimalSchema(), measBase.IdGenerator().make_table_id_factory()
        )
        detect_catalog = (
            measAlgorithms.SourceDetectionTask(config=driver.config.detection)
            .run(table, self.dataset.exposure)
            .sources
        )
        result = driver.run(self.dataset.exposure, catalog=detect_catalog)
        self.assertEqual(result.catalog["deblend_nChild"].tolist(), [3, 0, 0, 0, 0])
        self.assertTrue(result.backgroundList is None, "No backgroundList expected when detection is skipped")

    def testMultiBandMeasurementDriver(self):
        """Test the multi-band measurement driver."""

        # Here we demonstrate configuring the task without first creating the
        # driver instance. We later pass the config to the driver instance.
        config = MultiBandMeasurementDriverTask().ConfigClass()
        config.detection.background.useApprox = False
        config.doScaleVariance = True
        config.scaleVariance.background.binSize = 16
        config.doDetect = True
        config.doDeblend = True
        config.doMeasure = True
        config.doApCorr = True
        config.doRunCatalogCalculation = True
        config.deblend.minSNR = 5.0
        config.deblend.maxIter = 20
        config.doConserveFlux = False
        config.removeScarletData = False
        config.updateFluxColumns = True
        config.measureOnlyInRefBand = False
        config.measurement.plugins.names |= [
            "base_SdssCentroid",
            "base_SdssShape",
            "ext_shapeHSM_HsmSourceMoments",
        ]
        config.measurement.doReplaceWithNoise = True
        driver = MultiBandMeasurementDriverTask(config=config)
        bands = list(self.datasets.keys())

        # Wrap exposures in lists so they're recognized as multiband inputs.
        mExposure = [self.datasets[band].exposure for band in bands]

        # Provide a dummy mDeconvolved to verify basic functionality.
        mDeconvolved = [self.datasets[band].exposure for band in bands]

        # Run the multi-band driver with explicitly provided mDeconvolved.
        result = driver.run(mExposure=mExposure, mDeconvolved=mDeconvolved, bands=bands, refBand="i")

        # Run again without mDeconvolved to check that the driver generates
        # its own deconvolved exposures internally.
        result = driver.run(mExposure=mExposure, bands=bands, refBand="r")

        # Some key assertions to verify the result.
        self.assertIsInstance(result, pipeBase.Struct, "Expected a Struct result")
        self.assertTrue(hasattr(result, "catalogs"), "Expected 'catalog' in result")
        self.assertTrue(hasattr(result, "backgroundList"), "Expected 'backgroundList' in result")
        self.assertTrue(hasattr(result, "modelData"), "Expected 'modelData' in result")
        self.assertIsInstance(result.backgroundList, afwMath.BackgroundList, "Incorrect backgroundList type")
        self.assertIsInstance(result.modelData, ScarletModelData, "Expected modelData to be ScarletModelData")

        moments_xx = {
            "g": [9.97041569, 17.27109132, 10.20672397, 14.32357901, 6.71280036, 17.2026577],
            "r": [9.97412956, 17.26848716, 10.17418857, 14.32000843, 6.70788763, 17.16900062],
            "i": [9.96995504, 17.27980126, 10.2004923, 14.31823749, 6.71062237, 17.19066008],
        }

        # Loop through each band and check the output catalogs.
        for band in bands:
            self.assertIn(band, result.catalogs, f"Expected catalog for band {band} in result")
            self.assertIsInstance(result.catalogs[band], afwTable.SourceCatalog, "Incorrect catalog type")
            cat = result.catalogs[band]
            self.assertEqual(cat.schema, driver.schema, "Catalog schema mismatch")
            self.assertEqual(len(cat), 6, "Expected 6 sources in catalog")
            self.assertEqual(sum(cat["parent"] == 0), 2, "Expected 2 parents in catalog")
            self.assertEqual(sum(cat["parent"] != 0), 4, "Expected 4 children in catalog")
            self.assertEqual(cat["deblend_nChild"].tolist(), [3, 1, 0, 0, 0, 0])
            self.assertEqual(sum(cat["parent"] == 1), 3, "Expected 3 children of parent 1 in catalog")
            self.assertEqual(sum(cat["parent"] == 2), 1, "Expected 1 child of parent 2 in catalog")
            self.assertEqual(sum(cat["parent"] == 3), 0, "Expected 0 child of parent 3 in catalog")
            self.assertEqual(sum(cat["base_SdssShape_flag"]), 0, "Expected no objects with shape flag set")
            self.assertTrue(np.allclose(cat["ext_shapeHSM_HsmSourceMoments_xx"], moments_xx[band]))

    def testForcedMeasurementDriver(self):
        """Test the forced measurement driver."""

        # Create an input catalog using the single-band driver.
        # This catalog will be used as input for the forced measurement driver.
        single_band_driver = SingleBandMeasurementDriverTask()
        single_band_driver.config.doScaleVariance = False
        single_band_driver.config.doDetect = True
        single_band_driver.config.detection.background.useApprox = False
        single_band_driver.config.doDeblend = True
        single_band_driver.config.doMeasure = True
        single_band_driver.config.doApCorr = False
        single_band_driver.config.doRunCatalogCalculation = False
        input_catalog = single_band_driver.run(self.dataset.exposure).catalog
        input_table = input_catalog.asAstropy()[1:5]["id", "coord_ra", "coord_dec"]

        # Now configure the forced measurement driver.
        config = ForcedMeasurementDriverTask.ConfigClass()
        config.doScaleVariance = True
        config.scaleVariance.background.binSize = 32
        config.doApCorr = True
        config.measurement.plugins.names = [
            "base_PixelFlags",
            "base_TransformedCentroidFromCoord",
            "base_PsfFlux",
            "base_CircularApertureFlux",
        ]
        config.measurement.slots.psfFlux = "base_PsfFlux"
        config.measurement.slots.centroid = "base_TransformedCentroidFromCoord"
        config.measurement.slots.shape = None
        config.measurement.doReplaceWithNoise = False

        # Instantiate the forced measurement driver and run it.
        driver = ForcedMeasurementDriverTask(config=config)
        table = driver.runFromAstropy(
            input_table,
            self.dataset.exposure,
            id_column_name="id",
            ra_column_name="coord_ra",
            dec_column_name="coord_dec",
            psf_footprint_scaling=2.0,
        )

        self.assertIsInstance(table, astropyTable, "Expected an astropy Table result")
        some_expected_columns = [
            "coord_ra",
            "base_TransformedCentroidFromCoord_y",
            "base_PixelFlags_flag_interpolated",
            "base_CircularApertureFlux_17_0_instFlux",
            "base_PsfFlux_instFlux",
            "base_PsfFlux_apCorr",
            "slot_PsfFlux_apCorrErr",
        ]
        missing = [col for col in some_expected_columns if col not in table.colnames]
        self.assertFalse(missing, f"Missing expected columns: {missing}")

        self.assertEqual(len(table), len(input_table), "Expected same number of sources as in input_table")
        expected1 = np.array([-0.11130552, -0.46470672, -0.90978149, -0.55470151])
        expected2 = np.array([0.91131311, -2.30220238, -1.22224883, 0.69578745])
        try:
            self.assertTrue(np.allclose(table["base_PsfFlux_apCorr"], expected1))
        except AssertionError:
            self.assertTrue(np.allclose(table["base_PsfFlux_apCorr"], expected2))

    def _makeTestData(self, bands: list[str] = ["g", "r", "i"]):
        """Make idealized test data with Gaussian sources and PSF, simple noise
        and footprints.

        Parameters
        ----------
        bands :
            List of band names to create test data for. The first band will be
            used for single-band tests.
        """
        # Create a minimal schema to house the sources.
        schema = TestDataset.makeMinimalSchema()

        # Create a bounding box for the images.
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Extent2I(self.nx, self.ny))

        # Store datasets per band.
        self.datasets = {}

        rng = np.random.RandomState(565)

        for i, band in enumerate(bands):
            # Create a simple, test dataset.
            dataset = TestDataset(bbox, psfSigma=self.psfSigma)

            # Use this to slightly vary per-band properties.
            scale_factor = 1 + 0.05 * i

            # First source is a point.
            dataset.addSource(330000.0 * scale_factor, lsst.geom.Point2D(49.5, 45.5))

            # Second source is a galaxy.
            dataset.addSource(
                400000.0 * scale_factor, lsst.geom.Point2D(76.3, 88.2), afwGeom.Quadrupole(11, 7, 1)
            )

            # Our third and fourth sources are also galaxies, but they are in
            # close proximity to each other and somewhat to the point source,
            # creating overlap to test deblending algorithms.
            dataset.addSource(
                480000.0 * scale_factor, lsst.geom.Point2D(31.7, 29.0), afwGeom.Quadrupole(8.2, 2.6, 1.5)
            )
            dataset.addSource(
                515000.0 * scale_factor, lsst.geom.Point2D(11.5, 32.0), afwGeom.Quadrupole(3.7, 4.0, 0)
            )

            # Get the exposure and catalog.
            dataset.exposure, dataset.catalog = dataset.realize(
                noise=self.bgStdev, schema=schema, randomSeed=1746 + i
            )

            # Set EDGE by masking the whole exposure and un-masking an inner
            # bounding box.
            edgeMask = dataset.exposure.mask.getPlaneBitMask("EDGE")
            dataset.exposure.mask.array |= edgeMask
            inner_bbox = dataset.exposure.getBBox()
            inner_bbox.grow(-self.edgeWidth)
            dataset.exposure[inner_bbox].mask.array &= ~edgeMask

            # Mask at least one pixel.
            dataset.exposure.maskedImage.mask[12, 8] = 3

            # Set the PhotoCalib for the exposure.
            photo_calib = afwImage.PhotoCalib(
                self.meanCalibration * scale_factor, self.calibrationErr * scale_factor
            )
            dataset.exposure.setPhotoCalib(photo_calib)

            # Make a random ApCorrMap.
            apCorrMap = afwImage.ApCorrMap()
            for idx, name in enumerate(measBase.getApCorrNameSet(), start=2):
                apCorrMap.set(name + "_instFlux", ChebyshevBoundedField(bbox, rng.randn(3, 3)))
                apCorrMap.set(name + "_instFluxErr", ChebyshevBoundedField(bbox, rng.randn(3, 3)))
            dataset.exposure.setApCorrMap(apCorrMap)

            # Save the dataset and catalog per band.
            self.datasets[band] = dataset

        # For single-band compatibility, use the first band.
        first_band = bands[0]
        self.dataset = self.datasets[first_band]
        self.afw_masked_image = self.dataset.exposure.getMaskedImage()
        self.afw_image = self.dataset.exposure.getImage()


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
