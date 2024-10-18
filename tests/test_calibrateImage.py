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

import unittest
from unittest import mock
import tempfile

import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.daf.base
import lsst.daf.butler
import lsst.daf.butler.tests as butlerTests
import lsst.geom
import lsst.meas.algorithms
from lsst.meas.algorithms import testUtils
import lsst.meas.extensions.psfex
import lsst.meas.base
import lsst.meas.base.tests
import lsst.pipe.base.testUtils
from lsst.pipe.tasks.calibrateImage import CalibrateImageTask, NoPsfStarsToStarsMatchError
import lsst.utils.tests


class CalibrateImageTaskTests(lsst.utils.tests.TestCase):

    def setUp(self):
        # Different x/y dimensions so they're easy to distinguish in a plot,
        # and non-zero minimum, to help catch xy0 errors.
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(5, 4), lsst.geom.Point2I(205, 184))
        self.sky_center = lsst.geom.SpherePoint(245.0, -45.0, lsst.geom.degrees)
        self.photo_calib = 12.3
        dataset = lsst.meas.base.tests.TestDataset(bbox, crval=self.sky_center, calibration=self.photo_calib)
        # sqrt of area of a normalized 2d gaussian
        psf_scale = np.sqrt(4*np.pi*(dataset.psfShape.getDeterminantRadius())**2)
        noise = 10.0  # stddev of noise per pixel
        # Sources ordered from faintest to brightest.
        self.fluxes = np.array((6*noise*psf_scale,
                                12*noise*psf_scale,
                                45*noise*psf_scale,
                                150*noise*psf_scale,
                                400*noise*psf_scale,
                                1000*noise*psf_scale))
        self.centroids = np.array(((162, 22),
                                   (40, 70),
                                   (100, 160),
                                   (50, 120),
                                   (92, 35),
                                   (175, 154)), dtype=np.float32)
        for flux, centroid in zip(self.fluxes, self.centroids):
            dataset.addSource(instFlux=flux, centroid=lsst.geom.Point2D(centroid[0], centroid[1]))

        # Bright extended source in the center of the image: should not appear
        # in any of the output catalogs.
        center = lsst.geom.Point2D(100, 100)
        shape = lsst.afw.geom.Quadrupole(8, 9, 3)
        dataset.addSource(instFlux=500*noise*psf_scale, centroid=center, shape=shape)

        schema = dataset.makeMinimalSchema()
        self.truth_exposure, self.truth_cat = dataset.realize(noise=noise, schema=schema)
        # Add in a significant background, so we can test that the output
        # background is self-consistent with the calibrated exposure.
        self.truth_exposure.image += 500
        # To make it look like a version=1 (nJy fluxes) refcat
        self.truth_cat = self.truth_exposure.photoCalib.calibrateCatalog(self.truth_cat)
        self.ref_loader = testUtils.MockReferenceObjectLoaderFromMemory([self.truth_cat])
        metadata = lsst.daf.base.PropertyList()
        metadata.set("REFCAT_FORMAT_VERSION", 1)
        self.truth_cat.setMetadata(metadata)

        # TODO: a cosmic ray (need to figure out how to insert a fake-CR)
        # self.truth_exposure.image.array[10, 10] = 100000
        # self.truth_exposure.variance.array[10, 10] = 100000/noise

        # Copy the truth exposure, because CalibrateImage modifies the input.
        # Post-ISR ccds only contain: initial WCS, VisitInfo, filter
        self.exposure = afwImage.ExposureF(self.truth_exposure, deep=True)
        self.exposure.setWcs(self.truth_exposure.wcs)
        self.exposure.info.setVisitInfo(self.truth_exposure.visitInfo)
        # "truth" filter, to match the "truth" refcat.
        self.exposure.setFilter(lsst.afw.image.FilterLabel(physical='truth', band="truth"))

        # Test-specific configuration:
        self.config = CalibrateImageTask.ConfigClass()
        # We don't have many sources, so have to fit simpler models.
        self.config.psf_detection.background.approxOrderX = 1
        self.config.star_detection.background.approxOrderX = 1
        # Only insert 2 sky sources, for simplicity.
        self.config.star_sky_sources.nSources = 2
        # Use PCA psf fitter, as psfex fails if there are only 4 stars.
        self.config.psf_measure_psf.psfDeterminer = 'pca'
        # We don't have many test points, so can't match on complicated shapes.
        self.config.astrometry.matcher.numPointsForShape = 3
        # ApFlux has more noise than PsfFlux (the latter unrealistically small
        # in this test data), so we need to do magnitude rejection at higher
        # sigma, otherwise we can lose otherwise good sources.
        # TODO DM-39203: Once we are using Compensated Gaussian Fluxes, we
        # will use those fluxes here, and hopefully can remove this.
        self.config.astrometry.magnitudeOutlierRejectionNSigma = 9.0

        # Make a realistic id generator so that output catalog ids are useful.
        # NOTE: The id generator is used to seed the noise replacer during
        # measurement, so changes to values here can have subtle effects on
        # the centroids and fluxes mesaured on the image, which might cause
        # tests to fail.
        data_id = lsst.daf.butler.DataCoordinate.standardize(
            instrument="I",
            visit=self.truth_exposure.visitInfo.id,
            detector=12,
            universe=lsst.daf.butler.DimensionUniverse(),
        )
        self.config.id_generator.packer.name = "observation"
        self.config.id_generator.packer["observation"].n_observations = 10000
        self.config.id_generator.packer["observation"].n_detectors = 99
        self.config.id_generator.n_releases = 8
        self.config.id_generator.release_id = 2
        self.id_generator = self.config.id_generator.apply(data_id)

        # Something about this test dataset prefers a larger threshold here.
        self.config.star_selector["science"].unresolved.maximum = 0.2

    def _check_run(self, calibrate, result):
        """Test the result of CalibrateImage.run().

        Parameters
        ----------
        calibrate : `lsst.pipe.tasks.calibrateImage.CalibrateImageTask`
            Configured task that had `run` called on it.
        result : `lsst.pipe.base.Struct`
            Result of calling calibrate.run().
        """
        # Background should have 4 elements: 3 from compute_psf and one from
        # re-estimation during source detection.
        self.assertEqual(len(result.background), 4)

        # Both afw and astropy psf_stars catalogs should be populated.
        self.assertEqual(result.psf_stars["calib_psf_used"].sum(), 3)
        self.assertEqual(result.psf_stars_footprints["calib_psf_used"].sum(), 3)

        # Check that the summary statistics are reasonable.
        summary = result.exposure.info.getSummaryStats()
        self.assertFloatsAlmostEqual(summary.psfSigma, 2.0, rtol=1e-2)
        self.assertFloatsAlmostEqual(summary.ra, self.sky_center.getRa().asDegrees(), rtol=1e-7)
        self.assertFloatsAlmostEqual(summary.dec, self.sky_center.getDec().asDegrees(), rtol=1e-7)

        # Should have finite sky coordinates in the afw and astropy catalogs.
        self.assertTrue(np.isfinite(result.stars_footprints["coord_ra"]).all())
        self.assertTrue(np.isfinite(result.stars["coord_ra"]).all())

        # Returned photoCalib should be the applied value, not the ==1 one on the exposure.
        # Note that this is very approximate because we are basing this comparison
        # on just 2-3 stars.
        self.assertFloatsAlmostEqual(result.applied_photo_calib.getCalibrationMean(),
                                     self.photo_calib, rtol=1e-2)
        # Should have calibrated flux/magnitudes in the afw and astropy catalogs
        self.assertIn("slot_PsfFlux_flux", result.stars_footprints.schema)
        self.assertIn("slot_PsfFlux_mag", result.stars_footprints.schema)
        self.assertEqual(result.stars["slot_PsfFlux_flux"].unit, u.nJy)
        self.assertEqual(result.stars["slot_PsfFlux_mag"].unit, u.ABmag)

        # Should have detected all S/N >= 10 sources plus 2 sky sources, whether 1 or 2 snaps.
        self.assertEqual(len(result.stars), 7)
        # Did the psf flags get propagated from the psf_stars catalog?
        self.assertEqual(result.stars["calib_psf_used"].sum(), 3)

        # Check that all necessary fields are in the output.
        lsst.pipe.base.testUtils.assertValidOutput(calibrate, result)

    def test_run(self):
        """Test that run() returns reasonable values to be butler put.
        """
        calibrate = CalibrateImageTask(config=self.config)
        calibrate.astrometry.setRefObjLoader(self.ref_loader)
        calibrate.photometry.match.setRefObjLoader(self.ref_loader)
        result = calibrate.run(exposures=self.exposure)

        self._check_run(calibrate, result)

    def test_run_2_snaps(self):
        """Test that run() returns reasonable values to be butler put, when
        passed two exposures to combine as snaps.
        """
        calibrate = CalibrateImageTask(config=self.config)
        calibrate.astrometry.setRefObjLoader(self.ref_loader)
        calibrate.photometry.match.setRefObjLoader(self.ref_loader)
        # Halve the flux in each exposure to get the expected visit sum.
        self.exposure.image /= 2
        self.exposure.variance /= 2
        result = calibrate.run(exposures=[self.exposure, self.exposure])

        self._check_run(calibrate, result)

    def test_run_no_optionals(self):
        """Test that disabling optional outputs removes them from the output
        struct, as appropriate.
        """
        self.config.optional_outputs = None
        calibrate = CalibrateImageTask(config=self.config)
        calibrate.astrometry.setRefObjLoader(self.ref_loader)
        calibrate.photometry.match.setRefObjLoader(self.ref_loader)
        result = calibrate.run(exposures=self.exposure)

        self._check_run(calibrate, result)
        # These are the only optional outputs that require extra computation,
        # the others are included in the output struct regardless.
        self.assertNotIn("astrometry_matches", result.getDict())
        self.assertNotIn("photometry_matches", result.getDict())

    def test_compute_psf(self):
        """Test that our brightest sources are found by _compute_psf(),
        that a PSF is assigned to the expopsure.
        """
        calibrate = CalibrateImageTask(config=self.config)
        psf_stars, background, candidates = calibrate._compute_psf(self.exposure, self.id_generator)

        # Catalog ids should be very large from this id generator.
        self.assertTrue(all(psf_stars['id'] > 1000000000))

        # Background should have 3 elements: initial subtraction, and two from
        # re-estimation during the two detection passes.
        self.assertEqual(len(background), 3)

        # Only the point-sources with S/N > 50 should be in this output.
        self.assertEqual(psf_stars["calib_psf_used"].sum(), 3)
        # Sort in order of brightness, to easily compare with expected positions.
        psf_stars.sort(psf_stars.getPsfFluxSlot().getMeasKey())
        for record, flux, center in zip(psf_stars[::-1], self.fluxes, self.centroids[self.fluxes > 50]):
            self.assertFloatsAlmostEqual(record.getX(), center[0], rtol=0.01)
            self.assertFloatsAlmostEqual(record.getY(), center[1], rtol=0.01)
            # PsfFlux should match the values inserted.
            self.assertFloatsAlmostEqual(record["slot_PsfFlux_instFlux"], flux, rtol=0.01)

        # TODO: While debugging DM-32701, we're using PCA instead of psfex.
        # Check that we got a useable PSF.
        # self.assertIsInstance(self.exposure.psf, lsst.meas.extensions.psfex.PsfexPsf)
        self.assertIsInstance(self.exposure.psf, lsst.meas.algorithms.PcaPsf)
        # TestDataset sources have PSF radius=2 pixels.
        radius = self.exposure.psf.computeShape(self.exposure.psf.getAveragePosition()).getDeterminantRadius()
        self.assertFloatsAlmostEqual(radius, 2.0, rtol=1e-2)

        # To look at images for debugging (`setup display_ds9` and run ds9):
        # import lsst.afw.display
        # display = lsst.afw.display.getDisplay()
        # display.mtv(self.exposure)

    def test_measure_aperture_correction(self):
        """Test that _measure_aperture_correction() assigns an ApCorrMap to the
        exposure.
        """
        calibrate = CalibrateImageTask(config=self.config)
        psf_stars, background, candidates = calibrate._compute_psf(self.exposure, self.id_generator)

        # First check that the exposure doesn't have an ApCorrMap.
        self.assertIsNone(self.exposure.apCorrMap)
        calibrate._measure_aperture_correction(self.exposure, psf_stars)
        self.assertIsInstance(self.exposure.apCorrMap, afwImage.ApCorrMap)
        # We know that there are 2 fields from the normalization, plus more from
        # other configured plugins.
        self.assertGreater(len(self.exposure.apCorrMap), 2)

    def test_find_stars(self):
        """Test that _find_stars() correctly identifies the S/N>10 stars
        in the image and returns them in the output catalog.
        """
        calibrate = CalibrateImageTask(config=self.config)
        psf_stars, background, candidates = calibrate._compute_psf(self.exposure, self.id_generator)
        calibrate._measure_aperture_correction(self.exposure, psf_stars)

        stars = calibrate._find_stars(self.exposure, background, self.id_generator)

        # Catalog ids should be very large from this id generator.
        self.assertTrue(all(stars['id'] > 1000000000))

        # Background should have 4 elements: 3 from compute_psf and one from
        # re-estimation during source detection.
        self.assertEqual(len(background), 4)

        # Only 5 psf-like sources with S/N>10 should be in the output catalog,
        # plus two sky sources.
        self.assertEqual(len(stars), 7)
        self.assertTrue(stars.isContiguous())
        # Sort in order of brightness, to easily compare with expected positions.
        stars.sort(stars.getPsfFluxSlot().getMeasKey())
        for record, flux, center in zip(stars[::-1], self.fluxes, self.centroids[self.fluxes > 50]):
            self.assertFloatsAlmostEqual(record.getX(), center[0], rtol=0.01)
            self.assertFloatsAlmostEqual(record.getY(), center[1], rtol=0.01)
            self.assertFloatsAlmostEqual(record["slot_PsfFlux_instFlux"], flux, rtol=0.01)

    def test_astrometry(self):
        """Test that the fitted WCS gives good catalog coordinates.
        """
        calibrate = CalibrateImageTask(config=self.config)
        calibrate.astrometry.setRefObjLoader(self.ref_loader)
        psf_stars, background, candidates = calibrate._compute_psf(self.exposure, self.id_generator)
        calibrate._measure_aperture_correction(self.exposure, psf_stars)
        stars = calibrate._find_stars(self.exposure, background, self.id_generator)

        calibrate._fit_astrometry(self.exposure, stars)

        # Check that we got reliable matches with the truth coordinates.
        sky = stars["sky_source"]
        fitted = SkyCoord(stars[~sky]['coord_ra'], stars[~sky]['coord_dec'], unit="radian")
        truth = SkyCoord(self.truth_cat['coord_ra'], self.truth_cat['coord_dec'], unit="radian")
        idx, d2d, _ = fitted.match_to_catalog_sky(truth)
        np.testing.assert_array_less(d2d.to_value(u.milliarcsecond), 35.0)

    def test_photometry(self):
        """Test that the fitted photoCalib matches the one we generated,
        and that the exposure is calibrated.
        """
        calibrate = CalibrateImageTask(config=self.config)
        calibrate.astrometry.setRefObjLoader(self.ref_loader)
        calibrate.photometry.match.setRefObjLoader(self.ref_loader)
        psf_stars, background, candidates = calibrate._compute_psf(self.exposure, self.id_generator)
        calibrate._measure_aperture_correction(self.exposure, psf_stars)
        stars = calibrate._find_stars(self.exposure, background, self.id_generator)
        calibrate._fit_astrometry(self.exposure, stars)

        stars, matches, meta, photoCalib = calibrate._fit_photometry(self.exposure, stars)
        calibrate._apply_photometry(self.exposure, background)

        # NOTE: With this test data, PhotoCalTask returns calibrationErr==0,
        # so we can't check that the photoCal error has been set.
        self.assertFloatsAlmostEqual(photoCalib.getCalibrationMean(), self.photo_calib, rtol=1e-2)
        # The exposure should be calibrated by the applied photoCalib,
        # and the background should be calibrated to match.
        uncalibrated = self.exposure.image.clone()
        uncalibrated += background.getImage()
        uncalibrated /= self.photo_calib
        self.assertFloatsAlmostEqual(uncalibrated.array, self.truth_exposure.image.array, rtol=1e-2)
        # PhotoCalib on the exposure must be identically 1.
        self.assertEqual(self.exposure.photoCalib.getCalibrationMean(), 1.0)

        # Check that we got reliable magnitudes and fluxes vs. truth, ignoring
        # sky sources.
        sky = stars["sky_source"]
        fitted = SkyCoord(stars[~sky]['coord_ra'], stars[~sky]['coord_dec'], unit="radian")
        truth = SkyCoord(self.truth_cat['coord_ra'], self.truth_cat['coord_dec'], unit="radian")
        idx, _, _ = fitted.match_to_catalog_sky(truth)
        # Because the input variance image does not include contributions from
        # the sources, we can't use fluxErr as a bound on the measurement
        # quality here.
        self.assertFloatsAlmostEqual(stars[~sky]['slot_PsfFlux_flux'],
                                     self.truth_cat['truth_flux'][idx],
                                     rtol=0.1)
        self.assertFloatsAlmostEqual(stars[~sky]['slot_PsfFlux_mag'],
                                     self.truth_cat['truth_mag'][idx],
                                     rtol=0.01)

    def test_match_psf_stars(self):
        """Test that _match_psf_stars() flags the correct stars as psf stars
        and candidates.
        """
        calibrate = CalibrateImageTask(config=self.config)
        psf_stars, background, candidates = calibrate._compute_psf(self.exposure, self.id_generator)
        calibrate._measure_aperture_correction(self.exposure, psf_stars)
        stars = calibrate._find_stars(self.exposure, background, self.id_generator)

        # There should be no psf-related flags set at first.
        self.assertEqual(stars["calib_psf_candidate"].sum(), 0)
        self.assertEqual(stars["calib_psf_used"].sum(), 0)
        self.assertEqual(stars["calib_psf_reserved"].sum(), 0)

        # Reorder stars to be out of order with psf_stars (putting the sky
        # sources in front); this tests that I get the indexing right.
        stars.sort(stars.getCentroidSlot().getMeasKey().getX())
        stars = stars.copy(deep=True)
        # Re-number the ids: the matcher requires sorted ids: this is always
        # true in the code itself, but we've permuted them by sorting on
        # flux. We don't care what the actual ids themselves are here.
        stars["id"] = np.arange(len(stars))

        calibrate._match_psf_stars(psf_stars, stars)

        # Check that the three brightest stars have the psf flags transfered
        # from the psf_stars catalog by sorting in order of brightness.
        stars.sort(stars.getPsfFluxSlot().getMeasKey())
        # sort() above leaves the catalog non-contiguous.
        stars = stars.copy(deep=True)
        np.testing.assert_array_equal(stars["calib_psf_candidate"],
                                      [False, False, False, False, True, True, True])
        np.testing.assert_array_equal(stars["calib_psf_used"], [False, False, False, False, True, True, True])
        # Too few sources to reserve any in these tests.
        self.assertEqual(stars["calib_psf_reserved"].sum(), 0)

    def test_match_psf_stars_no_matches(self):
        """Check that _match_psf_stars handles the case of no cross-matches.
        """
        calibrate = CalibrateImageTask(config=self.config)
        # Make two catalogs that cannot have matches.
        stars = self.truth_cat[2:].copy(deep=True)
        psf_stars = self.truth_cat[:2].copy(deep=True)

        with self.assertRaisesRegex(NoPsfStarsToStarsMatchError,
                                    "No psf stars out of 2 matched 5 calib stars") as cm:
            calibrate._match_psf_stars(psf_stars, stars)
        self.assertEqual(cm.exception.metadata["n_psf_stars"], 2)
        self.assertEqual(cm.exception.metadata["n_stars"], 5)


class CalibrateImageTaskRunQuantumTests(lsst.utils.tests.TestCase):
    """Tests of ``CalibrateImageTask.runQuantum``, which need a test butler,
    but do not need real images.
    """
    def setUp(self):
        instrument = "testCam"
        exposure0 = 101
        exposure1 = 102
        visit = 100101
        detector = 42

        # Create a and populate a test butler for runQuantum tests.
        self.repo_path = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.repo = butlerTests.makeTestRepo(self.repo_path.name)

        # A complete instrument record is necessary for the id generator.
        instrumentRecord = self.repo.dimensions["instrument"].RecordClass(
            name=instrument, visit_max=1e6, exposure_max=1e6, detector_max=128,
            class_name="lsst.obs.base.instrument_tests.DummyCam",
        )
        self.repo.registry.syncDimensionData("instrument", instrumentRecord)

        # dataIds for fake data
        butlerTests.addDataIdValue(self.repo, "detector", detector)
        butlerTests.addDataIdValue(self.repo, "exposure", exposure0)
        butlerTests.addDataIdValue(self.repo, "exposure", exposure1)
        butlerTests.addDataIdValue(self.repo, "visit", visit)

        # inputs
        butlerTests.addDatasetType(self.repo, "postISRCCD", {"instrument", "exposure", "detector"},
                                   "ExposureF")
        butlerTests.addDatasetType(self.repo, "gaia_dr3_20230707", {"htm7"}, "SimpleCatalog")
        butlerTests.addDatasetType(self.repo, "ps1_pv3_3pi_20170110", {"htm7"}, "SimpleCatalog")

        # outputs
        butlerTests.addDatasetType(self.repo, "initial_pvi", {"instrument", "visit", "detector"},
                                   "ExposureF")
        butlerTests.addDatasetType(self.repo, "initial_stars_footprints_detector",
                                   {"instrument", "visit", "detector"},
                                   "SourceCatalog")
        butlerTests.addDatasetType(self.repo, "initial_stars_detector",
                                   {"instrument", "visit", "detector"},
                                   "ArrowAstropy")
        butlerTests.addDatasetType(self.repo, "initial_photoCalib_detector",
                                   {"instrument", "visit", "detector"},
                                   "PhotoCalib")
        # optional outputs
        butlerTests.addDatasetType(self.repo, "initial_pvi_background", {"instrument", "visit", "detector"},
                                   "Background")
        butlerTests.addDatasetType(self.repo, "initial_psf_stars_footprints_detector",
                                   {"instrument", "visit", "detector"},
                                   "SourceCatalog")
        butlerTests.addDatasetType(self.repo, "initial_psf_stars_detector",
                                   {"instrument", "visit", "detector"},
                                   "ArrowAstropy")
        butlerTests.addDatasetType(self.repo,
                                   "initial_astrometry_match_detector",
                                   {"instrument", "visit", "detector"},
                                   "Catalog")
        butlerTests.addDatasetType(self.repo,
                                   "initial_photometry_match_detector",
                                   {"instrument", "visit", "detector"},
                                   "Catalog")

        # dataIds
        self.exposure0_id = self.repo.registry.expandDataId(
            {"instrument": instrument, "exposure": exposure0, "detector": detector})
        self.exposure1_id = self.repo.registry.expandDataId(
            {"instrument": instrument, "exposure": exposure1, "detector": detector})
        self.visit_id = self.repo.registry.expandDataId(
            {"instrument": instrument, "visit": visit, "detector": detector})
        self.htm_id = self.repo.registry.expandDataId({"htm7": 42})

        # put empty data
        self.butler = butlerTests.makeTestCollection(self.repo)
        self.butler.put(afwImage.ExposureF(), "postISRCCD", self.exposure0_id)
        self.butler.put(afwImage.ExposureF(), "postISRCCD", self.exposure1_id)
        self.butler.put(afwTable.SimpleCatalog(), "gaia_dr3_20230707", self.htm_id)
        self.butler.put(afwTable.SimpleCatalog(), "ps1_pv3_3pi_20170110", self.htm_id)

    def tearDown(self):
        self.repo_path.cleanup()

    def test_runQuantum(self):
        task = CalibrateImageTask()
        lsst.pipe.base.testUtils.assertValidInitOutput(task)

        quantum = lsst.pipe.base.testUtils.makeQuantum(
            task, self.butler, self.visit_id,
            {"exposures": [self.exposure0_id],
             "astrometry_ref_cat": [self.htm_id],
             "photometry_ref_cat": [self.htm_id],
             # outputs
             "exposure": self.visit_id,
             "stars": self.visit_id,
             "stars_footprints": self.visit_id,
             "background": self.visit_id,
             "psf_stars": self.visit_id,
             "psf_stars_footprints": self.visit_id,
             "applied_photo_calib": self.visit_id,
             "initial_pvi_background": self.visit_id,
             "astrometry_matches": self.visit_id,
             "photometry_matches": self.visit_id,
             })
        mock_run = lsst.pipe.base.testUtils.runTestQuantum(task, self.butler, quantum)

        # Ensure the reference loaders have been configured.
        self.assertEqual(task.astrometry.refObjLoader.name, "gaia_dr3_20230707")
        self.assertEqual(task.photometry.match.refObjLoader.name, "ps1_pv3_3pi_20170110")
        # Check that the proper kwargs are passed to run().
        self.assertEqual(mock_run.call_args.kwargs.keys(), {"exposures", "result", "id_generator"})

    def test_runQuantum_2_snaps(self):
        task = CalibrateImageTask()
        lsst.pipe.base.testUtils.assertValidInitOutput(task)

        quantum = lsst.pipe.base.testUtils.makeQuantum(
            task, self.butler, self.visit_id,
            {"exposures": [self.exposure0_id, self.exposure1_id],
             "astrometry_ref_cat": [self.htm_id],
             "photometry_ref_cat": [self.htm_id],
             # outputs
             "exposure": self.visit_id,
             "stars": self.visit_id,
             "stars_footprints": self.visit_id,
             "background": self.visit_id,
             "psf_stars": self.visit_id,
             "psf_stars_footprints": self.visit_id,
             "applied_photo_calib": self.visit_id,
             "initial_pvi_background": self.visit_id,
             "astrometry_matches": self.visit_id,
             "photometry_matches": self.visit_id,
             })
        mock_run = lsst.pipe.base.testUtils.runTestQuantum(task, self.butler, quantum)

        # Ensure the reference loaders have been configured.
        self.assertEqual(task.astrometry.refObjLoader.name, "gaia_dr3_20230707")
        self.assertEqual(task.photometry.match.refObjLoader.name, "ps1_pv3_3pi_20170110")
        # Check that the proper kwargs are passed to run().
        self.assertEqual(mock_run.call_args.kwargs.keys(), {"exposures", "result", "id_generator"})

    def test_runQuantum_no_optional_outputs(self):
        # All the possible connections: we modify this to test each one by
        # popping off the removed connection, then re-setting it.
        connections = {"exposures": [self.exposure0_id, self.exposure1_id],
                       "astrometry_ref_cat": [self.htm_id],
                       "photometry_ref_cat": [self.htm_id],
                       # outputs
                       "exposure": self.visit_id,
                       "stars": self.visit_id,
                       "stars_footprints": self.visit_id,
                       "background": self.visit_id,
                       "psf_stars": self.visit_id,
                       "psf_stars_footprints": self.visit_id,
                       "applied_photo_calib": self.visit_id,
                       "initial_pvi_background": self.visit_id,
                       "astrometry_matches": self.visit_id,
                       "photometry_matches": self.visit_id,
                       }

        # Check that we can turn off one output at a time.
        for optional in ["psf_stars", "psf_stars_footprints", "astrometry_matches", "photometry_matches"]:
            config = CalibrateImageTask.ConfigClass()
            config.optional_outputs.remove(optional)
            task = CalibrateImageTask(config=config)
            lsst.pipe.base.testUtils.assertValidInitOutput(task)
            # Save the removed one for the next test.
            temp = connections.pop(optional)
            # This will fail with "Error in connection ..." if we don't pop
            # the optional item from the connections list just above.
            quantum = lsst.pipe.base.testUtils.makeQuantum(task, self.butler, self.visit_id, connections)
            # This confirms that the outputs did skip the removed one.
            self.assertNotIn(optional, quantum.outputs)
            # Restore the one we removed for the next test.
            connections[optional] = temp

    def test_lintConnections(self):
        """Check that the connections are self-consistent.
        """
        Connections = CalibrateImageTask.ConfigClass.ConnectionsClass
        lsst.pipe.base.testUtils.lintConnections(Connections)

    def test_runQuantum_exception(self):
        """Test exception handling in runQuantum.
        """
        task = CalibrateImageTask()
        lsst.pipe.base.testUtils.assertValidInitOutput(task)

        quantum = lsst.pipe.base.testUtils.makeQuantum(
            task, self.butler, self.visit_id,
            {"exposures": [self.exposure0_id],
             "astrometry_ref_cat": [self.htm_id],
             "photometry_ref_cat": [self.htm_id],
             # outputs
             "exposure": self.visit_id,
             "stars": self.visit_id,
             "stars_footprints": self.visit_id,
             "background": self.visit_id,
             "psf_stars": self.visit_id,
             "psf_stars_footprints": self.visit_id,
             "applied_photo_calib": self.visit_id,
             "initial_pvi_background": self.visit_id,
             "astrometry_matches": self.visit_id,
             "photometry_matches": self.visit_id,
             })

        # A generic exception should raise directly.
        msg = "mocked run exception"
        with (
            mock.patch.object(task, "run", side_effect=ValueError(msg)),
            self.assertRaisesRegex(ValueError, "mocked run exception")
        ):
            lsst.pipe.base.testUtils.runTestQuantum(task, self.butler, quantum, mockRun=False)

        # A AlgorimthError should write annotated partial outputs.
        error = lsst.meas.algorithms.MeasureApCorrError(name="test", nSources=100, ndof=101)

        def mock_run(exposures, result=None, id_generator=None):
            """Mock success through compute_psf, but failure after.
            """
            result.exposure = afwImage.ExposureF(10, 10)
            result.psf_stars_footprints = afwTable.SourceCatalog()
            result.psf_stars = afwTable.SourceCatalog().asAstropy()
            result.background = afwMath.BackgroundList()
            raise error

        with (
            mock.patch.object(task, "run", side_effect=mock_run),
            self.assertRaises(lsst.pipe.base.AnnotatedPartialOutputsError),
        ):
            with self.assertLogs("lsst.calibrateImage", level="DEBUG") as cm:
                lsst.pipe.base.testUtils.runTestQuantum(task,
                                                        self.butler,
                                                        quantum,
                                                        mockRun=False)

        logged = "\n".join(cm.output)
        self.assertIn("Task failed with only partial outputs", logged)
        self.assertIn("MeasureApCorrError", logged)

        # NOTE: This is an integration test of afw Exposure & SourceCatalog
        # metadata with the error annotation system in pipe_base.
        # Check that we did get the annotated partial outputs...
        pvi = self.butler.get("initial_pvi", self.visit_id)
        self.assertIn("Unable to measure aperture correction", pvi.metadata["failure.message"])
        self.assertIn("MeasureApCorrError", pvi.metadata["failure.type"])
        self.assertEqual(pvi.metadata["failure.metadata.ndof"], 101)
        stars = self.butler.get("initial_psf_stars_footprints_detector", self.visit_id)
        self.assertIn("Unable to measure aperture correction", stars.metadata["failure.message"])
        self.assertIn("MeasureApCorrError", stars.metadata["failure.type"])
        self.assertEqual(stars.metadata["failure.metadata.ndof"], 101)
        # ... but not the un-produced outputs.
        with self.assertRaises(FileNotFoundError):
            self.butler.get("initial_stars_footprints_detector", self.visit_id)


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
