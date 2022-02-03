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
"""Test IsolatedStarAssociationTask.
"""
import unittest
import numpy as np
import pandas as pd

import lsst.utils.tests
import lsst.daf.butler
import lsst.skymap

from lsst.pipe.tasks.isolatedStarAssociation import (IsolatedStarAssociationConfig,
                                                     IsolatedStarAssociationTask)
from smatch.matcher import Matcher


class MockSourceTableReference(lsst.daf.butler.DeferredDatasetHandle):
    """Very simple object that looks like a Gen3 data reference to
    a sourceTable_visit.

    Parameters
    ----------
    source_table : `pandas.DataFrame`
        Dataframe of the source table.
    """
    def __init__(self, source_table):
        self.source_table = source_table

    def get(self, parameters={}, **kwargs):
        """Retrieve the specified dataset using the API of the Gen3 Butler.

        Parameters
        ----------
        parameters : `dict`, optional
            Parameter dictionary.  Supported key is ``columns``.

        Returns
        -------
        dataframe : `pandas.DataFrame`
            dataframe, cut to the specified columns.
        """
        if 'columns' in parameters:
            return self.source_table[parameters['columns']]
        else:
            return self.source_table.copy()


class IsolatedStarAssociationTestCase(lsst.utils.tests.TestCase):
    """Tests of IsolatedStarAssociationTask.

    These tests bypass the middleware used for accessing data and
    managing Task execution.
    """
    def setUp(self):
        self.skymap = self._make_skymap()
        self.tract = 9813
        self.data_refs = self._make_simdata(self.tract)
        self.visits = np.arange(len(self.data_refs)) + 1

        self.data_ref_dict = {visit: data_ref for visit, data_ref in zip(self.visits,
                                                                         self.data_refs)}

        config = IsolatedStarAssociationConfig()
        config.band_order = ['i', 'r']
        config.extra_columns = ['extra_column']
        config.source_selector['science'].doFlags = False
        config.source_selector['science'].doIsolated = False

        self.isolatedStarAssociationTask = IsolatedStarAssociationTask(config=config)

    def _make_skymap(self):
        """Make a testing skymap."""
        skymap_config = lsst.skymap.ringsSkyMap.RingsSkyMapConfig()
        skymap_config.numRings = 120
        skymap_config.projection = "TAN"
        skymap_config.tractOverlap = 1.0/60
        skymap_config.pixelScale = 0.168
        return lsst.skymap.ringsSkyMap.RingsSkyMap(skymap_config)

    def _make_simdata(self, tract):
        """Make simulated data tables and references."""
        np.random.seed(12345)

        n_visit_per_band = 5
        n_star_both = 50
        n_star_just_one = 5

        tract_info = self.skymap[tract]
        ctr = tract_info.ctr_coord
        ctr_ra = ctr.getRa().asDegrees()
        ctr_dec = ctr.getDec().asDegrees()

        ra_both = np.linspace(ctr_ra - 1.5, ctr_ra + 1.5, n_star_both)
        dec_both = np.linspace(ctr_dec - 1.5, ctr_dec + 1.5, n_star_both)

        ra_just_r = np.linspace(ctr_ra - 0.5, ctr_ra + 0.5, n_star_just_one)
        dec_just_r = np.linspace(ctr_dec + 0.2, ctr_dec + 0.2, n_star_just_one)
        ra_just_i = np.linspace(ctr_ra - 0.5, ctr_ra + 0.5, n_star_just_one)
        dec_just_i = np.linspace(ctr_dec - 0.2, ctr_dec - 0.2, n_star_just_one)

        ra_neighbor = np.array([ra_both[n_star_both//2] + 1./3600.])
        dec_neighbor = np.array([dec_both[n_star_both//2] + 1./3600.])

        # Create the r-band datarefs
        dtype = [('sourceId', 'i8'),
                 ('ra', 'f8'),
                 ('decl', 'f8'),
                 ('apFlux_12_0_instFlux', 'f4'),
                 ('apFlux_12_0_instFluxErr', 'f4'),
                 ('apFlux_12_0_instFlux_flag', '?'),
                 ('extendedness', 'f4'),
                 ('visit', 'i4'),
                 ('detector', 'i4'),
                 ('physical_filter', 'U10'),
                 ('band', 'U2'),
                 ('extra_column', 'f4')]

        id_counter = 0
        visit_counter = 1

        data_refs = []
        for band in ['r', 'i']:
            if band == 'r':
                filtername = 'R FILTER'
                ra_just = ra_just_r
                dec_just = dec_just_r
            else:
                filtername = 'I FILTER'
                ra_just = ra_just_i
                dec_just = dec_just_i

            star_ra = np.concatenate((ra_both, ra_neighbor, ra_just))
            star_dec = np.concatenate((dec_both, dec_neighbor, dec_just))

            nstar = len(star_ra)

            for i in range(n_visit_per_band):
                ras = np.random.normal(loc=star_ra, scale=0.2/3600.)
                decs = np.random.normal(loc=star_dec, scale=0.2/3600.)

                table = np.zeros(nstar, dtype=dtype)
                table['sourceId'] = np.arange(nstar) + id_counter
                table['ra'] = ras
                table['decl'] = decs
                table['apFlux_12_0_instFlux'] = 100.0
                table['apFlux_12_0_instFluxErr'] = 1.0
                table['physical_filter'] = filtername
                table['band'] = band
                table['extra_column'] = np.ones(nstar)
                table['visit'] = visit_counter
                table['detector'] = 1

                if i == 0:
                    # Make one star have low s/n
                    table['apFlux_12_0_instFlux'][0] = 1.0

                data_refs.append(MockSourceTableReference(pd.DataFrame(table)))

                id_counter += nstar
                visit_counter += 1

        self.n_visit_per_band = n_visit_per_band
        self.n_star_both = n_star_both
        self.n_star_just_one = n_star_just_one
        self.star_ras = np.concatenate((ra_both, ra_just_r, ra_just_i, ra_neighbor))
        self.star_decs = np.concatenate((dec_both, dec_just_r, dec_just_i, dec_neighbor))

        return data_refs

    def test_compute_unique_ids(self):
        """Test computation of unique ids."""
        ids1 = self.isolatedStarAssociationTask._compute_unique_ids(self.skymap,
                                                                    9813,
                                                                    10000)
        ids2 = self.isolatedStarAssociationTask._compute_unique_ids(self.skymap,
                                                                    9814,
                                                                    5000)
        ids = np.concatenate((ids1, ids2))
        self.assertEqual(len(np.unique(ids)), len(ids))

    def test_remove_neighbors(self):
        """Test removing close neighbors."""
        primary_star_cat = np.zeros(3, dtype=[('ra', 'f8'),
                                              ('decl', 'f8')])

        # Put two stars < 2" apart, and across the 0/360 boundary.
        primary_star_cat['ra'] = [0.7/3600., 360.0 - 0.7/3600., 1.0]
        primary_star_cat['decl'] = [5.0, 5.0, 5.0]

        cut_cat = self.isolatedStarAssociationTask._remove_neighbors(primary_star_cat)

        self.assertEqual(len(cut_cat), 1)
        np.testing.assert_almost_equal(1.0, cut_cat['ra'][0])

    def test_match_primary_stars(self):
        """Test matching primary stars."""
        # Stack all the observations; we do not want any cutting here.
        tables = []
        for data_ref in self.data_refs:
            df = data_ref.get()
            tables.append(df.to_records())
        obs_cat = np.concatenate(tables)

        primary_star_cat = self.isolatedStarAssociationTask._match_primary_stars(['i', 'r'],
                                                                                 obs_cat)

        # Ensure we found the right number of stars in each p
        test_i = (primary_star_cat['primary_band'] == 'i')
        self.assertEqual(test_i.sum(), self.n_star_both + self.n_star_just_one + 1)
        test_r = (primary_star_cat['primary_band'] == 'r')
        self.assertEqual(test_r.sum(), self.n_star_just_one)

        # Ensure that these stars all match to input stars within 1 arcsec.
        with Matcher(self.star_ras, self.star_decs) as matcher:
            idx, i1, i2, d = matcher.query_radius(primary_star_cat['ra'],
                                                  primary_star_cat['decl'],
                                                  1./3600.,
                                                  return_indices=True)
        self.assertEqual(i1.size, self.star_ras.size)

    def test_get_source_table_visit_columns(self):
        """Test making of source table visit columns."""
        all_columns, persist_columns = self.isolatedStarAssociationTask._get_source_table_visit_columns()

        # Make sure all persisted columns are in all columns.
        for col in persist_columns:
            self.assertTrue(col in all_columns)

        # And make sure extendedness is not in persisted columns.
        self.assertTrue('extendedness' not in persist_columns)

    def test_match_obsservations(self):
        """Test _match_observations observation to primary matching."""
        # Stack all the observations; we do not want any cutting here.
        tables = []
        for data_ref in self.data_refs:
            df = data_ref.get()
            tables.append(df.to_records())
        obs_cat = np.concatenate(tables)

        obs_cat = np.lib.recfunctions.append_fields(obs_cat,
                                                    ['obj_index'],
                                                    [np.zeros(obs_cat.size, dtype=np.int32)],
                                                    dtypes=['i4'],
                                                    usemask=False)

        primary_bands = ['i', 'r']

        primary_cat = np.zeros(self.star_ras.size,
                               dtype=self.isolatedStarAssociationTask._get_primary_dtype(primary_bands))
        primary_cat['ra'] = self.star_ras
        primary_cat['decl'] = self.star_decs

        obs_cat_sorted, primary_star_cat = self.isolatedStarAssociationTask._match_observations(['i', 'r'],
                                                                                                obs_cat,
                                                                                                primary_cat)
        # All the star observations should be matched
        self.assertEqual(obs_cat_sorted.size, obs_cat.size)

        # Full index tests are performed in test_run_isolated_star_association_task

    def test_make_all_star_obs(self):
        """Test appending all the star observations."""
        obs_cat = self.isolatedStarAssociationTask._make_all_star_obs(self.skymap[self.tract],
                                                                      self.data_ref_dict)

        # Make sure we don't have any low s/n observations.
        sn_min = np.min(obs_cat['apFlux_12_0_instFlux']/obs_cat['apFlux_12_0_instFluxErr'])
        self.assertGreater(sn_min, 10.0)

        # And make sure they are all within the tract outer boundary.
        poly = self.skymap[self.tract].outer_sky_polygon
        use = poly.contains(np.deg2rad(obs_cat['ra']), np.deg2rad(obs_cat['decl']))
        self.assertEqual(use.sum(), len(obs_cat))

    def test_run_isolated_star_association_task(self):
        """Test running the full task."""

        struct = self.isolatedStarAssociationTask.run(self.skymap,
                                                      self.tract,
                                                      self.data_ref_dict)

        star_obs_cat = struct.star_obs_cat
        star_cat = struct.star_cat

        # Check that observations are all unique ids
        self.assertEqual(np.unique(star_obs_cat['sourceId']).size, star_obs_cat.size)

        inner_tract_ids = self.skymap.findTractIdArray(self.star_ras,
                                                       self.star_decs,
                                                       degrees=True)
        inner_stars = (inner_tract_ids == self.tract)

        # There should be the same number of stars in the inner tract region,
        # taking away the 2 close neighbors.
        self.assertEqual(star_cat.size, np.sum(inner_stars) - 2)

        # Check the star indices
        for i in range(len(star_cat)):
            all_obs_star = star_obs_cat[star_cat['obs_cat_index'][i]:
                                        star_cat['obs_cat_index'][i] + star_cat['nobs'][i]]

            # Check that these all point to the correct object
            np.testing.assert_array_equal(all_obs_star['obj_index'], i)

            # Check these are all pointing to the same star position
            with Matcher(np.atleast_1d(star_cat['ra'][i]),
                         np.atleast_1d(star_cat['decl'][i])) as matcher:
                idx = matcher.query_radius(all_obs_star['ra'],
                                           all_obs_star['decl'],
                                           1./3600.)
            self.assertEqual(len(idx[0]), star_cat['nobs'][i])

            # Check per band indices
            for band in ['r', 'i']:
                band_obs_star = star_obs_cat[star_cat[f'obs_cat_index_{band}'][i]:
                                             star_cat[f'obs_cat_index_{band}'][i]
                                             + star_cat[f'nobs_{band}'][i]]
                with Matcher(np.atleast_1d(star_cat['ra'][i]),
                             np.atleast_1d(star_cat['decl'][i])) as matcher:
                    idx = matcher.query_radius(band_obs_star['ra'],
                                               band_obs_star['decl'],
                                               1./3600.)
                self.assertEqual(len(idx[0]), star_cat[f'nobs_{band}'][i])


class MyMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
