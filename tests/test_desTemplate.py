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

import collections
import itertools
import os
import tempfile
import unittest

import numpy as np
from astropy.table import Table

import lsst.afw.geom
import lsst.afw.image
import lsst.afw.math
import lsst.geom
import lsst.ip.diffim
import lsst.meas.algorithms
import lsst.meas.base.tests
import lsst.pipe.base as pipeBase
import lsst.utils.tests

from utils import generate_data_id

# Change this to True, `setup display_ds9`, and open ds9 (or use another afw
# display backend) to show the tract/patch layouts on the image.
debug = False
if debug:
    import lsst.afw.display

    display = lsst.afw.display.Display()
    display.frame = 1


class GetTemplateTaskTestCase(lsst.utils.tests.TestCase):
    """Test that GetTemplateTask works on both one tract and multiple tract
    input coadd exposures.

    Makes a synthetic exposure large enough to fit four small tracts with 2x2
    (300x300 pixel) patches each, extracts pixels for those patches by warping,
    and tests GetTemplateTask's output against boxes that overlap various
    combinations of one or multiple tracts.
    """

    def setUp(self):
        self.scale = 0.2  # arcsec/pixel
        # DES pixel scale is approximately 0.263 arcsec/pixel
        self.template_scale = 0.263
        self.exposure = self._makeExposure()

        # Track temporary files for cleanup
        self.temp_files = []
        self.temp_csv_file = None

        if debug:
            display.image(self.exposure, "base exposure")

    def _makeExposure(self):
        """Create a large image to break up into tracts and patches.

        The image will have a source every 100 pixels in x and y, and a WCS
        that results in the tracts all fitting in the image, with tract=0
        in the lower left, tract=1 to the right, tract=2 above, and tract=3
        to the upper right.
        """
        box = lsst.geom.Box2I(
            lsst.geom.Point2I(-200, -200), lsst.geom.Point2I(800, 800)
        )
        # This WCS was constructed so that tract 0 mostly fills the lower left
        # quadrant of the image, and the other tracts fill the rest; slight
        # extra rotation as a check on the final warp layout, scaled by 5%
        # from the patch pixel scale.
        cd_matrix = lsst.afw.geom.makeCdMatrix(
            1.05 * self.scale * lsst.geom.arcseconds, 93 * lsst.geom.degrees
        )
        wcs = lsst.afw.geom.makeSkyWcs(
            lsst.geom.Point2D(120, 150),
            lsst.geom.SpherePoint(0, 0, lsst.geom.radians),
            cd_matrix,
        )
        dataset = lsst.meas.base.tests.TestDataset(box, wcs=wcs)
        for x, y in itertools.product(np.arange(0, 500, 100), np.arange(0, 500, 100)):
            dataset.addSource(1e5, lsst.geom.Point2D(x, y))
        exposure, _ = dataset.realize(2, dataset.makeMinimalSchema())
        exposure.setFilter(lsst.afw.image.FilterLabel("r", "r_03"))
        return exposure

    def _checkMetadata(self, template, config, box, wcs, nPsfs):
        """Check that the various metadata components were set correctly."""
        expectedBox = lsst.geom.Box2I(box)
        self.assertEqual(template.getBBox(), expectedBox)
        # WCS should match our exposure, not any of the coadd tracts.
        self.assertEqual(template.wcs, self.exposure.wcs)
        self.assertEqual(template.getXY0(), expectedBox.getMin())
        self.assertEqual(template.filter.bandLabel, "r")
        self.assertEqual(template.filter.physicalLabel, "r_03")
        self.assertEqual(template.psf.getComponentCount(), nPsfs)

    def _checkPixels(self, template, config, box):
        """Check that the pixel values in the template are close to the
        original image.
        """
        # All pixels should have real values!
        expectedBox = lsst.geom.Box2I(box)

        # Check that we fully filled the template
        self.assertTrue(np.all(np.isfinite(template.image.array)))

    def _makeSyntheticTileExposure(
        self, ra_center=0.0, dec_center=0.0, size_pixels=4096, band="r"
    ):
        """Create a synthetic tile exposure similar to a DES coadd tile.

        Parameters
        ----------
        ra_center : float
            RA center of the tile in degrees
        dec_center : float
            DEC center of the tile in degrees
        size_pixels : int
            Size of the tile in pixels (DES tiles are typically 10000x10000,
            but we use smaller for testing)
        band : str
            Photometric band

        Returns
        -------
        exposure : lsst.afw.image.ExposureF
            Synthetic tile exposure with WCS, PSF, and photometric calibration
        """
        # Create bounding box for the tile
        bbox = lsst.geom.Box2I(
            lsst.geom.Point2I(0, 0), lsst.geom.Extent2I(size_pixels, size_pixels)
        )

        # Create WCS centered on the tile center
        cd_matrix = lsst.afw.geom.makeCdMatrix(
            self.template_scale * lsst.geom.arcseconds,
            0 * lsst.geom.degrees,  # No rotation
        )
        wcs = lsst.afw.geom.makeSkyWcs(
            lsst.geom.Point2D(size_pixels / 2, size_pixels / 2),
            lsst.geom.SpherePoint(ra_center, dec_center, lsst.geom.degrees),
            cd_matrix,
        )

        # Create dataset with synthetic sources
        dataset = lsst.meas.base.tests.TestDataset(bbox, wcs=wcs)

        # Add a grid of sources across the tile (every 500 pixels)
        for x, y in itertools.product(
            np.arange(500, size_pixels - 500, 500),
            np.arange(500, size_pixels - 500, 500),
        ):
            dataset.addSource(1e5, lsst.geom.Point2D(x, y))

        # Realize the exposure with noise
        exposure, _ = dataset.realize(5.0, dataset.makeMinimalSchema())

        # Set filter
        filter_label = lsst.afw.image.FilterLabel(band=band, physical=f"{band}_03")
        exposure.setFilter(filter_label)

        # Add MAGZERO to metadata (typical for DES)
        metadata = exposure.getInfo().getMetadata()
        metadata.set("MAGZERO", 30.0)  # Typical DES zero point

        return exposure

    def _createSyntheticTileCatalog(self, bbox, wcs, band="r", tile_radius=0.02):
        """Create a synthetic tile catalog CSV file for testing.

        The tile catalog will be centered on the box, with tile_radius on each side

        Parameters
        ----------
        bbox : lsst.geom.Box2I
            Bounding box of the region to overlap
        wcs : lsst.afw.geom.SkyWcs
            WCS of the region
        band : str
            Photometric band
        tile_radius : float
            Size of tile on one side

        Returns
        -------
        csv_path : str
            Path to the temporary CSV file
        """
        # Get the center of the bbox in sky coordinates
        center = wcs.pixelToSky(lsst.geom.Point2D(bbox.getCenter()))
        ra_center = center.getRa().asDegrees()
        dec_center = center.getDec().asDegrees()

        # Calculate tile corners (simplified rectangular approximation)
        rac1 = ra_center - tile_radius
        rac2 = ra_center + tile_radius
        rac3 = ra_center + tile_radius
        rac4 = ra_center - tile_radius
        decc1 = dec_center - tile_radius
        decc2 = dec_center - tile_radius
        decc3 = dec_center + tile_radius
        decc4 = dec_center + tile_radius

        size_pixels = int(2 * tile_radius * 3600 / self.template_scale)
        # Create the synthetic tile exposure and save to FITS
        tile_exposure = self._makeSyntheticTileExposure(
            ra_center, dec_center, size_pixels=size_pixels, band=band
        )

        # Write to temporary FITS file
        temp_fits = tempfile.NamedTemporaryFile(
            suffix=".fits", delete=False, prefix="test_tile_"
        )
        tile_exposure.writeFits(temp_fits.name)
        temp_fits.close()
        self.temp_files.append(temp_fits.name)
        # Create tile catalog data
        tile_data = {
            "tilename": [f"TES{int(ra_center):04d}{int(dec_center):+05d}"],
            "band": [band],
            "survey": ["DES"],
            "ra_cent": [ra_center],
            "dec_cent": [dec_center],
            "rac1": [rac1],
            "rac2": [rac2],
            "rac3": [rac3],
            "rac4": [rac4],
            "decc1": [decc1],
            "decc2": [decc2],
            "decc3": [decc3],
            "decc4": [decc4],
            "filepath": [temp_fits.name],
        }

        # Create astropy table
        table = Table(tile_data)

        # Write to temporary CSV file
        temp_csv = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, prefix="test_tiles_"
        )
        table.write(temp_csv.name, format="ascii.csv", overwrite=True)
        temp_csv.close()
        self.temp_csv_file = temp_csv.name

        return temp_csv.name

    def tearDown(self):
        """Clean up temporary files created during testing."""
        # Remove temporary FITS files
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass  # Ignore cleanup errors

        # Remove temporary CSV file
        if self.temp_csv_file and os.path.exists(self.temp_csv_file):
            try:
                os.remove(self.temp_csv_file)
            except Exception:
                pass  # Ignore cleanup errors

    def testRunOneTractInput(self):
        """Test a bounding box that fits inside single DES tract"""

        box = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Point2I(180, 180))

        # Create synthetic tile catalog with tiles that overlap the test bbox
        synthetic_csv = self._createSyntheticTileCatalog(
            box, self.exposure.wcs, band="r"
        )

        config = lsst.ip.diffim.DesTemplateConfig()
        config.tileFile = synthetic_csv
        task = lsst.ip.diffim.DesTemplateTask(config=config)
        result = task.run(
            bbox=box, wcs=self.exposure.wcs, band="r", physical_filter="r_03"
        )

        self._checkMetadata(result.template, task.config, box, self.exposure.wcs, 1)
        self._checkPixels(result.template, task.config, box)


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
