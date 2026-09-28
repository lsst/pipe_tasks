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

import pytest
import tempfile
import unittest

import astropy.units as u
from astropy.table import Table
import numpy as np

import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
from lsst.afw.cameraGeom.testUtils import CameraWrapper
import lsst.daf.butler
import lsst.daf.butler.tests as butlerTests
import lsst.geom
import lsst.images
import lsst.images.psfs
import lsst.meas.base.tests
import lsst.utils.tests
from lsst.pipe.tasks.postprocess import ConsolidateVisitSummaryTask, ModelExtendednessColumnAction

from utils import makeTestVisitInfo, make_exposure_record


class ModelExtendednessColumnActionTestCase(lsst.utils.tests.TestCase):
    """Demo test case."""

    def setUp(self):
        self.bands = ("g", "r", "i")
        action = ModelExtendednessColumnAction(bands=self.bands, min_n_good_to_shift_flux_ratio=1)
        self.action = action
        data = {
            action.size_column.format(axis="x"): [1., 3., 0.01, 0.4, 0.02],
            action.size_column.format(axis="y"): [0.2, 6, 0.01, 0.2, 0.01],
        }
        model = action.model_flux_name
        factors = np.array([1.01, 1.12, 0.995, 0.998, 1.6])
        fluxes = np.array([1.5e3, 2.5e3, 6.8e3, 3.4e3, 5.5e3])
        for column_flux, column_flux_err, factors in (
            (action.model_column_flux, action.model_column_flux_err, factors),
            (action.psf_column_flux, action.psf_column_flux_err, None),
        ):
            for idx, band in enumerate(action.bands):
                flux = np.sqrt(idx + 1.)*fluxes
                if factors is not None:
                    flux *= factors
                data[column_flux.format(band=band, model=model)] = flux
                data[column_flux_err.format(band=band, model=model)] = np.sqrt(flux)
        self.data = Table(data)

    def testExtendednessColumnAction(self):
        action = self.action
        with pytest.raises(ValueError):
            action.validate()
        action.bands_combined = {"gri": "g,r,i"}
        action.validate()
        schema = action.getInputSchema()
        n_values = len(self.data[schema[0][0]])
        assert all([len(self.data[col]) == n_values for col, _ in schema[1:]])

        result = self.action(self.data)
        columns_expected = [
            action.output_column.format(band=band)
            for band in list(action.bands) + list(action.bands_combined.keys())
        ]
        assert list(result.keys()) == columns_expected

        for column, values in result.items():
            assert len(values) == n_values
            assert all((values >= 0) & (values <= 1))


class ConsolidateVisitSummaryTestCase(lsst.utils.tests.TestCase):
    """Test that ConsolidateVisitSummaryTask makes the same row from a legacy
    Exposure and from the equivalent lsst.images.VisitImage.
    """

    def setUp(self):
        instrument = "testCam"
        self.visit = 1
        self.band = "r"
        physical_filter = "r_test"

        # lsst.images needs raw amplifier geometry and a FIELD_ANGLE
        # transform, which the TestDataset detector does not have.
        detector = list(CameraWrapper().camera)[0]
        dataset = lsst.meas.base.tests.TestDataset(detector.getBBox())
        dataset.addSource(instFlux=1e4, centroid=lsst.geom.Point2D(50, 60))
        exposure, _ = dataset.realize(noise=10.0, schema=dataset.makeMinimalSchema())
        exposure.setDetector(detector)
        exposure.info.setVisitInfo(makeTestVisitInfo(id=self.visit).copyWith(exposureTime=30.0))
        exposure.setFilter(afwImage.FilterLabel(band=self.band, physical=physical_filter))
        summaryStats = afwImage.ExposureSummaryStats()
        summaryStats.psfSigma = 2.5
        summaryStats.zeroPoint = 31.4
        exposure.info.setSummaryStats(summaryStats)
        bbox = lsst.geom.Box2D(exposure.getBBox())
        exposure.info.setValidPolygon(afwGeom.Polygon(
            [bbox.getMin(), lsst.geom.Point2D(bbox.maxX, bbox.minY), lsst.geom.Point2D(bbox.minX, bbox.maxY)]
        ))
        self.exposure = exposure

        self.repo_path = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.repo_path.cleanup)
        # A file datastore, so that reads go through the lsst.images
        # formatter's component support.
        config = lsst.daf.butler.Config()
        config["datastore", "cls"] = "lsst.daf.butler.datastores.fileDatastore.FileDatastore"
        self.repo = butlerTests.makeTestRepo(self.repo_path.name, config=config)
        self.enterContext(self.repo)
        instrumentRecord = self.repo.dimensions["instrument"].RecordClass(
            name=instrument, visit_max=1e6, exposure_max=1e6, detector_max=128,
            class_name="lsst.obs.base.instrument_tests.DummyCam",
        )
        self.repo.registry.syncDimensionData("instrument", instrumentRecord)
        butlerTests.addDataIdValue(self.repo, "physical_filter", physical_filter, band=self.band)
        butlerTests.addDataIdValue(self.repo, "detector", detector.getId())
        butlerTests.addDataIdValue(self.repo, "visit", self.visit, physical_filter=physical_filter)
        self.dataId = {"instrument": instrument, "visit": self.visit, "detector": detector.getId()}
        dimensions = {"instrument", "visit", "detector"}
        butlerTests.addDatasetType(self.repo, "legacy_image", dimensions, "ExposureF")
        butlerTests.addDatasetType(self.repo, "future_image", dimensions, "VisitImage")
        self.butler = butlerTests.makeTestCollection(self.repo, uniqueId=self.id())
        self.enterContext(self.butler)

    def _run(self, datasetType, image, input_image_type):
        """Put one image in the butler and consolidate it into a visit
        summary.
        """
        self.butler.put(image, datasetType, self.dataId)
        ref = self.butler.find_dataset(datasetType, self.dataId, dimension_records=True)
        config = ConsolidateVisitSummaryTask.ConfigClass()
        config.do_refit_pointing = False
        config.do_write_visit_geometry = False
        config.input_image_type = input_image_type
        task = ConsolidateVisitSummaryTask(config=config)
        return task.run(visit=self.visit, handles=[self.butler.getDeferred(ref)]).visitSummary

    def _checkFutureMatchesLegacy(self, unit):
        """Check the future-mode row against the legacy-mode row.

        Parameters
        ----------
        unit : `astropy.units.Unit`
            Units of the image pixels.

        Returns
        -------
        visitImage : `lsst.images.VisitImage`
            The converted image the future-mode row was made from.
        """
        visitImage = lsst.images.VisitImage.from_legacy(
            self.exposure, exposure_record=make_exposure_record(self.exposure), unit=unit
        )
        # The afw GaussianPsf from TestDataset cannot be serialized, and this
        # task does not read the PSF.
        visitImage.psf = lsst.images.psfs.GaussianPointSpreadFunction(
            2.0, stamp_size=21, bounds=visitImage.bbox
        )
        legacy = self._run("legacy_image", self.exposure, "legacy")[0]
        future = self._run("future_image", visitImage, "future")[0]

        self.assertEqual(future.getId(), legacy.getId())
        self.assertEqual(future["visit"], legacy["visit"])
        self.assertEqual(future["physical_filter"], legacy["physical_filter"])
        self.assertEqual(future["band"], legacy["band"])
        self.assertEqual(future.getBBox(), legacy.getBBox())
        self.assertEqual(future.getValidPolygon(), legacy.getValidPolygon())
        self.assertWcsAlmostEqualOverBBox(future.getWcs(), legacy.getWcs(), legacy.getBBox())
        self.assertFloatsAlmostEqual(
            future.getPhotoCalib().getCalibrationMean(),
            legacy.getPhotoCalib().getCalibrationMean(),
            rtol=1e-5,
        )
        self.assertEqual(future.getVisitInfo().id, legacy.getVisitInfo().id)
        self.assertEqual(future.getVisitInfo().exposureTime, legacy.getVisitInfo().exposureTime)
        self.assertEqual(future["psfSigma"], legacy["psfSigma"])
        self.assertEqual(future["zeroPoint"], legacy["zeroPoint"])
        return visitImage

    def testCalibratedImage(self):
        """Test an image calibrated to nJy, as calibrateImage writes."""
        self.exposure.setPhotoCalib(afwImage.PhotoCalib(1.0))
        visitImage = self._checkFutureMatchesLegacy(u.nJy)
        self.assertIsNone(visitImage.photometric_scaling)

    def testUncalibratedImage(self):
        """Test an image in instrumental units with a photometric scaling."""
        visitImage = self._checkFutureMatchesLegacy(u.electron)
        self.assertIsNotNone(visitImage.photometric_scaling)

    def testFullRequiresLegacy(self):
        """Test that full=True is rejected for VisitImage inputs."""
        config = ConsolidateVisitSummaryTask.ConfigClass()
        config.input_image_type = "future"
        config.full = True
        with self.assertRaises(ValueError):
            config.validate()


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
