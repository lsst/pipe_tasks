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

import lsst.afw.geom as afwGeom
from lsst.meas.astrom import ConvertCatalogCoordinatesConfig
from lsst.pipe.tasks.diff_matched_tract_catalog import (
    DiffMatchedTractCatalogConfig, DiffMatchedTractCatalogTask, MatchedCatalogFluxesConfig,
)

from astropy.table import Table
import numpy as np
import pytest


def _error_format(column):
    return f'{column}Err'


class DiffMatchedTractCatalogTaskTestCase(lsst.utils.tests.TestCase):
    """DiffMatchedTractCatalogTask test case."""
    def setUp(self):
        ra_cen = 180.
        ra = ra_cen + np.array([-5.1, -2.2, 0., 3.1, -3.2, 2.01, -4.1])/60
        dec_cen = 0.
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

        n_points = len(ra)
        n_unmatched = 2
        n_matched = n_points - n_unmatched
        # Reorder some indices to make arbitrary differences
        idx_ref = np.empty(n_points, dtype=int)
        idx_ref[:n_matched] = np.arange(n_matched)[::-1]
        idx_ref[n_matched:] = np.arange(n_matched, n_points)
        data_ref = {
            column_ra_ref: ra[idx_ref],
            column_dec_ref: dec[idx_ref],
            columns_flux[0]: fluxes[0][idx_ref],
            columns_flux[1]: fluxes[1][idx_ref],
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
            DiffMatchedTractCatalogConfig.columns_target_select_true.default[0]: flags,
            DiffMatchedTractCatalogConfig.columns_target_select_false.default[0]: ~flags,
        }
        self.catalog_target = Table(data=data_target)

        # Make the last two rows unmatched (we set eps_coord very large)
        match_row = np.arange(len(ra))[::-1] - n_unmatched
        self.catalog_match_ref = Table(data={
            'match_candidate': flags,
            'match_row': match_row,
        })

        self.catalog_match_target = Table(data={
            'match_candidate': flags,
            'match_row': match_row,
        })

        columns_flux_configs = {
            band: MatchedCatalogFluxesConfig(
                column_ref_flux=columns_flux[idx],
                columns_target_flux=[columns_flux[idx]],
                columns_target_flux_err=[columns_flux_err[idx]],
            )
            for idx, band in enumerate(bands)
        }
        self.config = DiffMatchedTractCatalogConfig(
            columns_target_coord_err=[column_ra_target_err, column_dec_target_err],
            columns_flux=columns_flux_configs,
        )

        self.wcs = afwGeom.makeSkyWcs(crpix=lsst.geom.Point2D(9000, 9000),
                                      crval=lsst.geom.SpherePoint(ra_cen, dec_cen, lsst.geom.degrees),
                                      cdMatrix=afwGeom.makeCdMatrix(scale=0.2*lsst.geom.arcseconds))

    def tearDown(self):
        del self.catalog_ref
        del self.catalog_target
        del self.catalog_match_ref
        del self.catalog_match_target
        del self.config
        del self.wcs

    def test_DiffMatchedTractCatalogTask(self):
        # These tables will have columns added to them in run
        columns_ref, columns_target = (list(x.columns) for x in (self.catalog_ref, self.catalog_target))
        task = DiffMatchedTractCatalogTask(config=self.config)
        result = task.run(
            catalog_ref=self.catalog_ref,
            catalog_target=self.catalog_target,
            catalog_match_ref=self.catalog_match_ref,
            catalog_match_target=self.catalog_match_target,
            wcs=self.wcs,
        )
        # TODO: Remove pandas support in DM-46523
        with pytest.warns(FutureWarning):
            result_pd = task.run(
                catalog_ref=self.catalog_ref.to_pandas(),
                catalog_target=self.catalog_target.to_pandas(),
                catalog_match_ref=self.catalog_match_ref.to_pandas(),
                catalog_match_target=self.catalog_match_target.to_pandas(),
                wcs=self.wcs,
            )
            self.assertListEqual(list(result.cat_matched.columns), list(result_pd.cat_matched.columns))
            for column in result.cat_matched.columns:
                self.assertListEqual(list(result.cat_matched[column]), list(result_pd.cat_matched[column]))

        columns_result = list(result.cat_matched.columns)
        columns_expect = list(columns_target) + ["match_distance", "match_distanceErr"]
        prefix = DiffMatchedTractCatalogConfig.column_matched_prefix_ref.default
        columns_expect.extend((f'{prefix}{col}' for col in columns_ref))
        self.assertListEqual(columns_expect, columns_result)

    def test_spherical(self):
        task = DiffMatchedTractCatalogTask(config=self.config)
        task.config.coord_format.coords_spherical = not task.config.coord_format.coords_spherical
        task.run(
            catalog_ref=self.catalog_ref,
            catalog_target=self.catalog_target,
            catalog_match_ref=self.catalog_match_ref,
            catalog_match_target=self.catalog_match_target,
            wcs=self.wcs,
        )


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
