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

import numpy as np
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

from lsst.afw.table import SourceCatalog
import lsst.utils.tests
import lsst.pipe.base.testUtils
from lsst.verify import Name
from lsst.verify.tasks.testUtils import MetricTaskTestCase

from lsst.pipe.tasks.metrics import \
    NumberDeblendedSourcesMetricTask, NumberDeblendChildSourcesMetricTask


def _makeDummyCatalog(nParents, *, skyFlags=False, deblendFlags=False, nChildren=0, nGrandchildren=0):
    """Create a trivial catalog for testing deblending counts.

    Parameters
    ----------
    nParents : `int`
        The number of entries in the catalog prior to deblending.
    skyFlags : `bool`
        If set, the schema includes flags associated with sky sources,
        and one top-level source (the deblended one, if it exists) and any
        descendents are sky sources.
    deblendFlags : `bool`
        If set, the schema includes flags associated with the deblender.
    nChildren : `int`
        If positive, one source is deblended into ``nChildren`` children. This
        parameter is ignored if ``deblendFlags`` is `False`.
    nGrandchildren : `int`
        If positive, one source produced by ``nChildren`` is deblended into
        ``nGrandchildren`` children. This parameter is ignored if ``nChildren``
        is 0 or not applicable.

    Returns
    -------
    catalog : `lsst.afw.table.SourceCatalog`
        A new catalog with ``nParents + nChildren + nGrandchildren`` rows.
    """
    schema = SourceCatalog.Table.makeMinimalSchema()
    if skyFlags:
        schema.addField("sky_source", type="Flag", doc="Sky source.")
    if deblendFlags:
        # See https://community.lsst.org/t/4957 for flag definitions.
        # Do not use detect_ flags, as they are defined by a postprocessing
        # task and some post-deblend catalogs may not have them.
        schema.addField('deblend_nChild', type=np.int32, doc='')
        schema.addField('deblend_nPeaks', type=np.int32, doc='')
        schema.addField('deblend_parentNPeaks', type=np.int32, doc='')
        schema.addField('deblend_parentNChild', type=np.int32, doc='')
    catalog = SourceCatalog(schema)
    if nParents > 0:  # normally anti-pattern, but simplifies nested ifs
        for i in range(nParents):
            record = catalog.addNew()
            if deblendFlags:
                record["deblend_nPeaks"] = 1
        if skyFlags:
            record["sky_source"] = True
        if deblendFlags and nChildren > 0:
            children = _addChildren(catalog, record, nChildren)
            if nGrandchildren > 0:
                _addChildren(catalog, children[0], nGrandchildren)
    return catalog


def _addChildren(catalog, parent, nChildren):
    """Add children to a catalog source.

    Parameters
    ----------
    catalog : `lsst.afw.table.SourceCatalog`
        The catalog to update. Its schema must contain all supported
        deblender flags.
    parent : `lsst.afw.table.SourceRecord`
        The source record to serve as the parent for any new children. Must be
        an element of ``catalog`` (not validated).
    nChildren : `int`
        The number of children of ``parent`` to add to ``catalog``.

    Returns
    -------
    children : `list` [`lsst.afw.table.SourceRecord`]
        A list of the ``nChildren`` new children.
    """
    newRecords = []
    if nChildren > 0:
        parent["deblend_nChild"] = nChildren
        parent["deblend_nPeaks"] = nChildren
        parentId = parent.getId()
        for i in range(nChildren):
            child = catalog.addNew()
            child.setParent(parentId)
            child["deblend_parentNPeaks"] = nChildren
            child["deblend_parentNChild"] = nChildren
            if "sky_source" in parent.schema:
                child["sky_source"] = parent["sky_source"]
            newRecords.append(child)
    return newRecords


class TestNumDeblended(MetricTaskTestCase):

    METRIC_NAME = Name(metric="pipe_tasks.numDeblendedSciSources")

    @classmethod
    def makeTask(cls):
        return NumberDeblendedSourcesMetricTask()

    def testValid(self):
        catalog = _makeDummyCatalog(3, deblendFlags=True, nChildren=2)
        result = self.task.run(catalog)
        lsst.pipe.base.testUtils.assertValidOutput(self.task, result)
        meas = result.measurement

        self.assertEqual(meas.metric_name, self.METRIC_NAME)
        assert_quantity_allclose(meas.quantity, u.Quantity(1))

    def testEmptyCatalog(self):
        catalog = _makeDummyCatalog(0, deblendFlags=True)
        result = self.task.run(catalog)
        lsst.pipe.base.testUtils.assertValidOutput(self.task, result)
        meas = result.measurement

        self.assertEqual(meas.metric_name, self.METRIC_NAME)
        assert_quantity_allclose(meas.quantity, u.Quantity(0))

    def testNothingDeblended(self):
        catalog = _makeDummyCatalog(3, deblendFlags=True, nChildren=0)
        result = self.task.run(catalog)
        lsst.pipe.base.testUtils.assertValidOutput(self.task, result)
        meas = result.measurement

        self.assertEqual(meas.metric_name, self.METRIC_NAME)
        assert_quantity_allclose(meas.quantity, u.Quantity(0))

    def testSkyIgnored(self):
        catalog = _makeDummyCatalog(3, skyFlags=True, deblendFlags=True, nChildren=2)
        result = self.task.run(catalog)
        lsst.pipe.base.testUtils.assertValidOutput(self.task, result)
        meas = result.measurement

        self.assertEqual(meas.metric_name, self.METRIC_NAME)
        assert_quantity_allclose(meas.quantity, u.Quantity(0))

    def testMultiDeblending(self):
        catalog = _makeDummyCatalog(5, deblendFlags=True, nChildren=3, nGrandchildren=2)
        result = self.task.run(catalog)
        lsst.pipe.base.testUtils.assertValidOutput(self.task, result)
        meas = result.measurement

        self.assertEqual(meas.metric_name, self.METRIC_NAME)
        assert_quantity_allclose(meas.quantity, u.Quantity(1))

    def testNoDeblending(self):
        catalog = _makeDummyCatalog(3, deblendFlags=False)
        try:
            result = self.task.run(catalog)
        except lsst.pipe.base.NoWorkFound:
            # Correct behavior
            pass
        else:
            # Alternative correct behavior
            lsst.pipe.base.testUtils.assertValidOutput(self.task, result)
            meas = result.measurement
            self.assertIsNone(meas)


class TestNumDeblendChild(MetricTaskTestCase):

    METRIC_NAME = Name(metric="pipe_tasks.numDeblendChildSciSources")

    @classmethod
    def makeTask(cls):
        return NumberDeblendChildSourcesMetricTask()

    def testValid(self):
        catalog = _makeDummyCatalog(3, deblendFlags=True, nChildren=2)
        result = self.task.run(catalog)
        lsst.pipe.base.testUtils.assertValidOutput(self.task, result)
        meas = result.measurement

        self.assertEqual(meas.metric_name, self.METRIC_NAME)
        assert_quantity_allclose(meas.quantity, u.Quantity(2))

    def testEmptyCatalog(self):
        catalog = _makeDummyCatalog(0, deblendFlags=True)
        result = self.task.run(catalog)
        lsst.pipe.base.testUtils.assertValidOutput(self.task, result)
        meas = result.measurement

        self.assertEqual(meas.metric_name, self.METRIC_NAME)
        assert_quantity_allclose(meas.quantity, u.Quantity(0))

    def testNothingDeblended(self):
        catalog = _makeDummyCatalog(3, deblendFlags=True, nChildren=0)
        result = self.task.run(catalog)
        lsst.pipe.base.testUtils.assertValidOutput(self.task, result)
        meas = result.measurement

        self.assertEqual(meas.metric_name, self.METRIC_NAME)
        assert_quantity_allclose(meas.quantity, u.Quantity(0))

    def testSkyIgnored(self):
        catalog = _makeDummyCatalog(3, skyFlags=True, deblendFlags=True, nChildren=2)
        result = self.task.run(catalog)
        lsst.pipe.base.testUtils.assertValidOutput(self.task, result)
        meas = result.measurement

        self.assertEqual(meas.metric_name, self.METRIC_NAME)
        assert_quantity_allclose(meas.quantity, u.Quantity(0))

    def testMultiDeblending(self):
        catalog = _makeDummyCatalog(5, deblendFlags=True, nChildren=3, nGrandchildren=2)
        result = self.task.run(catalog)
        lsst.pipe.base.testUtils.assertValidOutput(self.task, result)
        meas = result.measurement

        self.assertEqual(meas.metric_name, self.METRIC_NAME)
        # Expect 2 from first-level children and 2 from subchildren
        assert_quantity_allclose(meas.quantity, u.Quantity(4))

    def testNoDeblending(self):
        catalog = _makeDummyCatalog(3, deblendFlags=False)
        try:
            result = self.task.run(catalog)
        except lsst.pipe.base.NoWorkFound:
            # Correct behavior
            pass
        else:
            # Alternative correct behavior
            lsst.pipe.base.testUtils.assertValidOutput(self.task, result)
            meas = result.measurement
            self.assertIsNone(meas)


# Hack around unittest's hacky test setup system
del MetricTaskTestCase


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
