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
import lsst.utils.tests

from lsst.meas.astrom import ConvertCatalogCoordinatesConfig
from lsst.pipe.tasks.match_tract_catalog import MatchTractCatalogConfig, MatchTractCatalogTask
from lsst.pipe.tasks.match_tract_catalog_probabilistic import MatchTractCatalogProbabilisticTask
from lsst.skymap.discreteSkyMap import DiscreteSkyMap

from astropy.table import Table
import numpy as np


def _error_format(column):
    return f'{column}Err'


class MatchTractCatalogTaskTestCase(lsst.utils.tests.TestCase):
    """Test matching with some arbitrary mock data.

    This test is largely copied from diff_matched_tract_catalog, which
    implemented outputting a matched catalog separately from this task.
    """
    def setUp(self):
        # test wrapping from 0 to 360
        ra_cen = 0.
        ra = ra_cen + np.array([-5.1, -2.2, 0., 3.1, -3.2, 2.01, -4.1])/60
        dec_cen = -30.
        dec = dec_cen + np.array([-4.15, 1.15, 0, 2.15, -7.15, -3.05, 5.7])/60
        mag_g = np.array([23., 24., 25., 25.5, 26., 24.7, 23.3])
        mag_r = mag_g + [0.5, -0.2, -0.8, -0.5, -1.5, 0.8, -0.4]

        coord_format = ConvertCatalogCoordinatesConfig
        zeropoint = coord_format.mag_zeropoint_ref.default
        fluxes = tuple(10**(-0.4*(mag - zeropoint)) for mag in (mag_g, mag_r))
        # Percent error in measurement
        err_flux = np.array((0.02, 0.015, -0.035, 0.02, -0.04, 0.06, 0.01))
        # Absolute error
        eps_coord = np.array((2.3, 0.6, -1.7, 3.6, -2.4, 55.0, -40.8))
        err_coord = np.full_like(eps_coord, 0.02)
        eps_coord *= err_coord
        flags = np.ones_like(eps_coord, dtype=bool)

        bands = ['g', 'r']

        columns_flux = [f'flux_{band}' for band in bands]
        columns_flux_err = [_error_format(column) for column in columns_flux]

        column_ra_ref = coord_format.column_ref_coord1.default
        column_dec_ref = coord_format.column_ref_coord2.default
        column_ra_target = coord_format.column_target_coord1.default
        column_dec_target = coord_format.column_target_coord2.default

        column_ra_target_err, column_dec_target_err = [
            _error_format(col) for col in (column_ra_target, column_dec_target)
        ]

        data_ref = {
            column_ra_ref: ra[::-1],
            column_dec_ref: dec[::-1],
            columns_flux[0]: fluxes[0][::-1],
            columns_flux[1]: fluxes[1][::-1],
        }
        self.catalog_ref = Table(data=data_ref)

        data_target = {
            column_ra_target: ra + eps_coord,
            column_dec_target: dec + eps_coord,
            column_ra_target_err: err_coord,
            column_dec_target_err: err_coord,
            columns_flux[0]: fluxes[0]*(1 + err_flux),
            columns_flux[1]: fluxes[1]*(1 + err_flux),
            columns_flux_err[0]: np.sqrt(fluxes[0]),
            columns_flux_err[1]: np.sqrt(fluxes[1]),
            "detect_isPrimary": flags,
            "merge_peak_sky": ~flags,
        }
        self.catalog_target = Table(data=data_target)

        config = MatchTractCatalogConfig(
            coord_unit="deg",
            output_matched_catalog=True,
            refcat_sharding_type="none",
            target_sharding_type="none",
        )
        mtc = config.match_tract_catalog
        mtc.retarget(MatchTractCatalogProbabilisticTask)
        mtc.columns_ref_flux = [columns_flux[0], columns_flux[0]]
        mtc.columns_ref_meas = [column_ra_ref, column_dec_ref, columns_flux[0], columns_flux[0]]
        mtc.columns_target_meas = [
            column_ra_target, column_dec_target, columns_flux[0], columns_flux[0],
        ]
        mtc.columns_target_err = [
            column_ra_target_err, column_dec_target_err, columns_flux_err[0], columns_flux_err[0],
        ]
        config.validate()
        self.config = config
        self.skymap = DiscreteSkyMap(
            DiscreteSkyMap.ConfigClass(raList=[ra_cen], decList=[dec_cen], radiusList=[1.])
        )
        self.wcs = self.skymap[0].wcs

    def test_MatchTractCatalogTask(self):
        # These tables will have columns added to them in run
        columns_ref, columns_target = (list(x.columns) for x in (self.catalog_ref, self.catalog_target))
        task = MatchTractCatalogTask(None, config=self.config)
        task._add_tract_column_to_catalogs(self.catalog_ref, self.catalog_target, self.skymap)
        result = task.run(
            catalog_ref=self.catalog_ref,
            catalog_target=self.catalog_target,
            wcs=self.wcs,
        )
        columns_result = list(result.cat_output_matched.columns)
        columns_expect = list(columns_target) + [
            "tract", "patch", "match_candidate", "match_distance", "match_distanceErr",
        ]
        prefix = task.diff_matched_catalog.config.column_matched_prefix_ref
        columns_expect.extend((f"{prefix}{col}" for col in columns_ref))
        columns_expect.extend((
            f"{prefix}{col}" for col in ("tract", "patch", "flux_total", "match_candidate")
        ))
        self.assertListEqual(columns_expect, columns_result)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
