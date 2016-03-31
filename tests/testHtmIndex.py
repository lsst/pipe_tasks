from __future__ import absolute_import, division, print_function
#
# LSST Data Management System
# Copyright 2008-2016 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
import os
import tempfile
import shutil
import unittest
import string
from collections import Counter

import numpy

import lsst.utils
import lsst.afw.table as afwTable
import lsst.afw.geom as afwGeom
import lsst.afw.coord as afwCoord
import lsst.daf.persistence as dafPersist
from lsst.pipe.tasks.indexReferenceTask import IngestIndexedReferenceTask
from lsst.pipe.tasks.loadIndexedReferenceObjects import LoadIndexedReferenceObjectsTask

obs_test_dir = lsst.utils.getPackageDir('obs_test')
input_dir = os.path.join(obs_test_dir, "data", "input")


def make_coord(ra, dec):
    return afwCoord.IcrsCoord(afwGeom.Angle(ra, afwGeom.degrees), afwGeom.Angle(dec, afwGeom.degrees))


class HtmIndexTestCase(lsst.utils.tests.TestCase):
    @staticmethod
    def make_sky_catalog(out_path, size=1000):
        numpy.random.seed(123)
        ident = numpy.arange(1, size+1, dtype=int)
        ra = numpy.random.random(size)*360.
        dec = numpy.degrees(numpy.arccos(2.*numpy.random.random(size) - 1.))
        dec -= 90.
        a_mag = 16. + numpy.random.random(size)*4.
        a_mag_err = 0.01 + numpy.random.random(size)*0.2
        b_mag = 17. + numpy.random.random(size)*5.
        b_mag_err = 0.02 + numpy.random.random(size)*0.3
        is_photometric = numpy.random.randint(2, size=size)
        is_resolved = numpy.random.randint(2, size=size)
        is_variable = numpy.random.randint(2, size=size)
        extra_col1 = numpy.random.normal(size=size)
        extra_col2 = numpy.random.normal(1000., 100., size=size)

        def get_word(word_len):
            return "".join(numpy.random.choice([s for s in string.ascii_letters], word_len))
        extra_col3 = numpy.array([get_word(num) for num in numpy.random.randint(11, size=size)])

        dtype = numpy.dtype([('id', float), ('ra_icrs', float), ('dec_icrs', float), ('a', float),
                             ('a_err', float), ('b', float), ('b_err', float), ('is_phot', int),
                             ('is_res', int), ('is_var', int), ('val1', float), ('val2', float),
                             ('val3', '|S11')])

        arr = numpy.array(zip(ident, ra, dec, a_mag, a_mag_err, b_mag, b_mag_err, is_photometric, is_resolved,
                              is_variable, extra_col1, extra_col2, extra_col3), dtype=dtype)
        numpy.savetxt(out_path+"/ref.txt", arr, delimiter=",",
                      header="id,ra_icrs,dec_icrs,a,a_err,b,b_err,is_phot,is_res,is_var,val1,val2,val3",
                      fmt=["%i", "%.6g", "%.6g", "%.4g", "%.4g", "%.4g", "%.4g", "%i",
                           "%i", "%i", "%.2g", "%.2g", "%s"])
        numpy.savetxt(out_path+"/ref_test_delim.txt", arr, delimiter="|",
                      header="id,ra_icrs,dec_icrs,a,a_err,b,b_err,is_phot,is_res,is_var,val1,val2,val3",
                      fmt=["%i", "%.6g", "%.6g", "%.4g", "%.4g", "%.4g", "%.4g", "%i",
                           "%i", "%i", "%.2g", "%.2g", "%s"])
        return out_path+"/ref.txt", out_path+"/ref_test_delim.txt", arr

    @classmethod
    def setUpClass(cls):
        cls.out_path = tempfile.mkdtemp()
        test_cat_path = lsst.utils.getPackageDir('pipe_tasks')+'/tests/'+'test_cat.fits'
        cls.test_cat = afwTable.SourceCatalog.readFits(test_cat_path)
        ret = cls.make_sky_catalog(cls.out_path)
        cls.sky_catalog_file, cls.sky_catalog_file_delim, cls.sky_catalog = ret
        cls.test_ras = [210., 14.5, 93., 180., 286., 0.]
        cls.test_decs = [-90., -51., -30.1, 0., 27.3, 62., 90.]
        cls.search_radius = 3.
        cls.comp_cats = {}
        for ra in cls.test_ras:
            for dec in cls.test_decs:
                tupl = (ra, dec)
                cent = make_coord(*tupl)
                cls.comp_cats[tupl] = []
                for rec in cls.sky_catalog:
                    if make_coord(rec['ra_icrs'], rec['dec_icrs']).angularSeparation(cent).asDegrees()\
                       < cls.search_radius:
                        cls.comp_cats[tupl].append(rec['id'])

        cls.test_repo_path = cls.out_path+"/test_repo"
        config = IngestIndexedReferenceTask.ConfigClass()
        config.ra_name = 'ra_icrs'
        config.dec_name = 'dec_icrs'
        config.mag_column_list = ['a', 'b']
        config.id_name = 'id'
        config.mag_err_column_map = {'a': 'a_err', 'b': 'b_err'}
        IngestIndexedReferenceTask.parseAndRun(args=[input_dir, "--output", cls.test_repo_path,
                                               cls.sky_catalog_file], config=config)
        cls.test_butler = dafPersist.Butler(cls.test_repo_path)

    @classmethod
    def tearDownClass(cls):
        try:
            shutil.rmtree(cls.out_path)
        except Exception:
            print("WARNING: failed to remove temporary dir %r" % (cls.out_path,))
        del cls.out_path
        del cls.sky_catalog_file
        del cls.sky_catalog_file_delim
        del cls.sky_catalog
        del cls.test_ras
        del cls.test_decs
        del cls.search_radius
        del cls.comp_cats
        del cls.test_butler
        del cls.test_cat

    def testAgainstPersisted(self):
        pix_id = 671901
        data_id = IngestIndexedReferenceTask.make_data_id(pix_id)
        dataset_name = IngestIndexedReferenceTask.ConfigClass().ref_dataset_name
        self.assertTrue(self.test_butler.datasetExists(dataset_name, data_id))
        ref_cat = self.test_butler.get(dataset_name, data_id)
        ex1 = ref_cat.extract('*')
        ex2 = self.test_cat.extract('*')
        # compare sets as the order may be different
        self.assertEqual(set(ex1.keys()), set(ex2.keys()))
        for key in ex1:
            self.assertTrue(numpy.array_equal(ex1[key], ex2[key]))

    def testIngest(self):
        default_config = IngestIndexedReferenceTask.ConfigClass()
        # test ingest with default config
        # This should raise since I haven't specified the ra/dec/mag columns.
        self.assertRaises(ValueError, IngestIndexedReferenceTask.parseAndRun, args=[input_dir, "--output",
                          self.out_path+"/output", self.sky_catalog_file], config=default_config)
        # test with ~minimum config.  Mag errors are not technically necessary, but might as well test here
        default_config.ra_name = 'ra_icrs'
        default_config.dec_name = 'dec_icrs'
        default_config.mag_column_list = ['a', 'b']
        default_config.mag_err_column_map = {'a': 'a_err'}
        # should raise since all columns need an error column if any do
        self.assertRaises(ValueError, IngestIndexedReferenceTask.parseAndRun, args=[input_dir, "--output",
                          self.out_path+"/output", self.sky_catalog_file], config=default_config)
        # test with multiple files and correct config
        default_config.mag_err_column_map = {'a': 'a_err', 'b': 'b_err'}
        IngestIndexedReferenceTask.parseAndRun(
            args=[input_dir, "--output", self.out_path+"/output_multifile",
                  self.sky_catalog_file, self.sky_catalog_file],
            config=default_config)
        # test with config overrides
        default_config = IngestIndexedReferenceTask.ConfigClass()
        default_config.ra_name = 'ra'
        default_config.dec_name = 'dec'
        default_config.mag_column_list = ['a', 'b']
        default_config.mag_err_column_map = {'a': 'a_err', 'b': 'b_err'}
        default_config.ref_dataset_name = 'other_photo_astro_ref'
        default_config.level = 10
        default_config.is_photometric_name = 'is_phot'
        default_config.is_resolved_name = 'is_res'
        default_config.is_variable_name = 'is_var'
        default_config.id_name = 'id'
        default_config.extra_col_names = ['val1', 'val2', 'val3']
        default_config.file_reader.header_lines = 1
        default_config.file_reader.colnames = ['id', 'ra', 'dec', 'a', 'a_err', 'b', 'b_err', 'is_phot', 'is_res',
                                   'is_var', 'val1', 'val2', 'val3']
        default_config.file_reader.delimiter = '|'
        # this also tests changing the delimiter
        IngestIndexedReferenceTask.parseAndRun(
            args=[input_dir, "--output", self.out_path+"/output_override",
                  self.sky_catalog_file_delim], config=default_config)

    def testQuery(self):
        loader = LoadIndexedReferenceObjectsTask(self.test_butler)
        for tupl in self.comp_cats:
            cent = make_coord(*tupl)
            lcat = loader.loadSkyCircle(cent, afwGeom.Angle(self.search_radius, afwGeom.degrees),
                                        filterName='a')
            if lcat.refCat:
                # deep copy the catalog so it's contiguous in memory.  This lets us use numpy syntax.
                cat = lcat.refCat.copy(True)
                self.assertEqual(Counter(cat['id']), Counter(self.comp_cats[tupl]))
                # make sure there are no duplicate ids
                self.assertEqual(len(set(Counter(cat['id']).values())), 1)
                self.assertEqual(len(set(Counter(self.comp_cats[tupl]).values())), 1)
            else:
                self.assertEqual(len(self.comp_cats[tupl]), 0)


def suite():
    lsst.utils.tests.init()
    suites = []
    suites += unittest.makeSuite(HtmIndexTestCase)
    suites += unittest.makeSuite(lsst.utils.tests.MemoryTestCase)
    return unittest.TestSuite(suites)


def run(shouldExit=False):
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
