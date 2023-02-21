# This file is part of pipe_tasks.
#
# LSST Data Management System
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
# See COPYRIGHT file at the top of the source tree.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#
"""Test HIPS code."""
import unittest
import numpy as np
import hpgeom as hpg

import lsst.utils.tests
import lsst.daf.butler
import lsst.afw.image
import lsst.skymap
import lsst.geom

from lsst.pipe.tasks.hips import (
    HighResolutionHipsTask,
    HighResolutionHipsConfig,
    HighResolutionHipsConnections
)


class MockCoaddImageHandle(lsst.daf.butler.DeferredDatasetHandle):
    """Simple object that looks like a Gen3 deferred dataset handle
    to an exposure.

    Parameters
    ----------
    exposure : `lsst.afw.image.ExposureF`
        Exposure to hold.
    """
    def __init__(self, exposure):
        self.exposure = exposure

    def get(self, **kwargs):
        """Retrieve the dataset using the API of the Gen3 Butler.

        Returns
        -------
        exposure : `lsst.afw.image.ExposureF`
            Exposure held in mock handle.
        """
        return self.exposure

    @property
    def dataId(self):
        return {'visit': 0, 'detector': 0}


class HipsTestCase(unittest.TestCase):
    def test_hips_single(self):
        """Test creating a single HIPS image."""
        np.random.seed(12345)

        config = HighResolutionHipsConfig()

        skymap = self._make_skymap()

        tract = 9597
        patch = 50
        tract_info = skymap[tract]
        patch_info = tract_info[patch]

        exposure = self._make_noise_exposure(patch_info)
        handles = [MockCoaddImageHandle(exposure)]

        center = patch_info.wcs.pixelToSky(patch_info.inner_bbox.getCenter())
        pixel = self._get_pixel(2**config.hips_order, center)

        hips_task = HighResolutionHipsTask(config=config)
        output = hips_task.run([pixel], handles)

        # Check that all the pixels are filled.
        npix = (np.isfinite(output.hips_exposures[pixel].image.array.ravel()).sum())
        self.assertEqual(npix, output.hips_exposures[pixel].image.array.size)

        # Check that metadata is correct
        self.assertEqual(output.hips_exposures[pixel].getPhotoCalib(), exposure.getPhotoCalib())
        self.assertEqual(output.hips_exposures[pixel].getFilter(), exposure.getFilter())

    def test_hips_double(self):
        """Test creating a HIPS image from two neighboring patches."""
        np.random.seed(12345)

        config = HighResolutionHipsConfig()

        skymap = self._make_skymap()

        tract = 9597
        patches = [50, 51]
        tract_info = skymap[tract]

        handles = []
        centers = []
        for patch in patches:
            patch_info = tract_info[patch]
            exposure = self._make_noise_exposure(patch_info)
            handles.append(MockCoaddImageHandle(exposure))
            centers.append(patch_info.wcs.pixelToSky(patch_info.inner_bbox.getCenter()))

        center = lsst.geom.SpherePoint(
            (centers[0].getRa().asDegrees() + centers[1].getRa().asDegrees())/2.*lsst.geom.degrees,
            (centers[0].getDec().asDegrees() + centers[1].getDec().asDegrees())/2.*lsst.geom.degrees
        )
        pixel = self._get_pixel(2**config.hips_order, center)

        # Just transform one, make sure it falls off the edge.
        hips_task = HighResolutionHipsTask(config=config)
        output = hips_task.run([pixel], [handles[0]])

        # Check that not all the pixels are filled.
        npix = (np.isfinite(output.hips_exposures[pixel].image.array.ravel()).sum())
        self.assertLess(npix, output.hips_exposures[pixel].image.array.size)

        # Transform both.
        hips_task = HighResolutionHipsTask(config=config)
        output = hips_task.run([pixel], handles)

        # Check that all the pixels are filled.
        npix = (np.isfinite(output.hips_exposures[pixel].image.array.ravel()).sum())
        self.assertEqual(npix, output.hips_exposures[pixel].image.array.size)

    def test_hips_none(self):
        """Test making a HIPS image with no overlapping inputs."""
        np.random.seed(12345)

        config = HighResolutionHipsConfig()

        skymap = self._make_skymap()

        tract = 9597
        patch = 50
        tract_info = skymap[tract]
        patch_info = tract_info[patch]

        exposure = self._make_noise_exposure(patch_info)
        handles = [MockCoaddImageHandle(exposure)]

        pixel = 0

        hips_task = HighResolutionHipsTask(config=config)
        output = hips_task.run([pixel], handles)

        # Check that there is no returned image
        self.assertEqual(len(output.hips_exposures), 0)

    def test_hips_connections(self):
        """Test that the HIPS connections validate properly."""
        config = HighResolutionHipsConfig()

        # Test that the connections validate
        _ = HighResolutionHipsConnections(config=config)

        # Test that changing hips_order will break things because of the
        # dimensions mismatch.
        config.hips_order = 5
        with self.assertRaises(ValueError):
            _ = HighResolutionHipsConnections(config=config)

        # I'd like to change the dimensions but I don't know how to do that.

    def _make_noise_exposure(self, patch_info):
        """Make a simple noise exposure.

        Parameters
        ----------
        patch_info : `lsst.skymap.PatchInfo`
            Patch info to use to make the exposure.

        Returns
        -------
        exposure : `lsst.afw.image.ExposureF`
            Noise exposure.
        """
        exposure = lsst.afw.image.ExposureF(patch_info.outer_bbox)
        exposure.image.array[:, :] = np.random.normal(scale=1.0, size=exposure.image.array.shape)
        exposure.setWcs(patch_info.wcs)
        exposure.setPhotoCalib(lsst.afw.image.PhotoCalib(calibrationMean=1.0))
        exposure.setFilter(lsst.afw.image.FilterLabel(band='i'))

        return exposure

    def _make_skymap(self):
        """Make a testing skymap.

        Returns
        -------
        skymap : `lsst.skymap.RingsSkyMap`
        """

        # Generate a skymap
        skymap_config = lsst.skymap.ringsSkyMap.RingsSkyMapConfig()
        skymap_config.numRings = 120
        skymap_config.projection = "TAN"
        skymap_config.tractOverlap = 1.0/60
        skymap_config.pixelScale = 0.168
        return lsst.skymap.ringsSkyMap.RingsSkyMap(skymap_config)

    def _get_pixel(self, nside, sphpoint):
        """Get the pixel value from a spherepoint.

        Parameters
        ----------
        nside : `int`
            Healpix nside
        sphpoint : `lsst.geom.SpherePoint`
            Point to compute pixel value.

        Returns
        -------
        pixel : `int`
            Healpix pixel (nest ordering)
        """
        pixel = hpg.angle_to_pixel(
            nside,
            sphpoint.getRa().asDegrees(),
            sphpoint.getDec().asDegrees(),
        )
        return pixel


if __name__ == "__main__":
    unittest.main()
