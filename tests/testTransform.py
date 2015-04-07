#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2008-2015 AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
"""
Test the basic operation of measurement transformations.

We test measurement transforms in two ways:

First, we construct and run a simple TransformTask on the (mocked) results of
measurement tasks. The same test is carried out against both
SingleFrameMeasurementTask and ForcedMeasurementTask, on the basis that the
transformation system should be agnostic as to the origin of the source
catalog it is transforming.

Secondly, we use data from the obs_test package to demonstrate that the
transformtion system and its interface package are capable of operating on
data processed by the rest of the stack.

For the purposes of testing, we define a "TrivialMeasurement" plugin and
associated transformation. Rather than building a catalog by measuring
genuine SourceRecords, we directly populate a catalog following the
TrivialMeasurement schema, then check that it is transformed properly by the
TrivialMeasurementTransform.
"""
import contextlib
import math
import os
import shutil
import tempfile
import unittest

import eups

import lsst.afw.coord as afwCoord
import lsst.afw.table as afwTable
import lsst.meas.base as measBase
import lsst.utils.tests as utilsTests

from lsst.pipe.tasks.processCcd import ProcessCcdTask, ProcessCcdConfig
from lsst.pipe.tasks.transformMeasurement import (TransformConfig, TransformTask,
                                                  RunTransformConfig, RunTransformTask)

PLUGIN_NAME = "base_TrivialMeasurement"

# Rather than providing real WCS and calibration objects to the
# transformation, we use this simple placeholder to keep track of the number
# of times it is accessed.
class Placeholder(object):
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1


class TrivialMeasurementTransform(measBase.transforms.MeasurementTransform):
    def __init__(self, config, name, mapper):
        """Pass through all input fields to the output, and add a new field
        named after the measurement with the suffix "_transform".
        """
        measBase.transforms.MeasurementTransform.__init__(self, config, name, mapper)
        for key, field in mapper.getInputSchema().extract(name + "*").itervalues():
            mapper.addMapping(key)
        self.key = mapper.editOutputSchema().addField(name + "_transform", type="D", doc="transformed dummy")

    def __call__(self, inputCatalog, outputCatalog, wcs, calib):
        """Transform inputCatalog to outputCatalog.

        We update the wcs and calib placeholders to indicate that they have
        been seen in the transformation, but do not use their values.

        @param[in]  inputCatalog  SourceCatalog of measurements for transformation.
        @param[out] outputCatalog BaseCatalog of transformed measurements.
        @param[in]  wcs           Dummy WCS information; an instance of Placeholder.
        @param[in]  calib         Dummy calibration information; an instance of Placeholder.
        """
        if hasattr(wcs, "increment"):
            wcs.increment()
        if hasattr(calib, "increment"):
            calib.increment()
        inColumns = inputCatalog.getColumnView()
        outColumns = outputCatalog.getColumnView()
        outColumns[self.key] = -1.0 * inColumns[self.name]


class TrivialMeasurementBase(object):
    """Default values for a trivial measurement plugin, subclassed below"""
    @staticmethod
    def getExecutionOrder():
        return 0

    @staticmethod
    def getTransformClass():
        return TrivialMeasurementTransform

    def measure(self, measRecord, exposure):
        measRecord.set(self.key, 1.0)


@measBase.base.register(PLUGIN_NAME)
class SFTrivialMeasurement(TrivialMeasurementBase, measBase.sfm.SingleFramePlugin):
    """Single frame version of the trivial measurement"""
    def __init__(self, config, name, schema, metadata):
        measBase.sfm.SingleFramePlugin.__init__(self, config, name, schema, metadata)
        self.key = schema.addField(name, type="D", doc="dummy field")


@measBase.base.register(PLUGIN_NAME)
class ForcedTrivialMeasurement(TrivialMeasurementBase, measBase.forcedMeasurement.ForcedPlugin):
    """Forced frame version of the trivial measurement"""
    def __init__(self, config, name, schemaMapper, metadata):
        measBase.forcedMeasurement.ForcedPlugin.__init__(self, config, name, schemaMapper, metadata)
        self.key = schemaMapper.editOutputSchema().addField(name, type="D", doc="dummy field")


class TransformTestCase(utilsTests.TestCase):
    def _transformAndCheck(self, measConf, schema, transformTask):
        """Check the results of applying transformTask to a SourceCatalog.

        @param[in] measConf       Measurement plugin configuration.
        @param[in] schema         Input catalog schema.
        @param[in] transformTask  Instance of TransformTask to be applied.

        For internal use by this test case.
        """
        # There should now be one transformation registered per measurement plugin.
        self.assertEqual(len(measConf.plugins.names), len(transformTask.transforms))

        # Rather than do a real measurement, we use a dummy source catalog
        # containing a source at an arbitrary position.
        inCat = afwTable.SourceCatalog(schema)
        r = inCat.addNew()
        r.setCoord(afwCoord.Coord("00:00:00", "11:11:11"))
        r[PLUGIN_NAME] = 1.0

        wcs, calib = Placeholder(), Placeholder()
        outCat = transformTask.run(inCat, wcs, calib)

        # Check that all sources have been transformed appropriately.
        for inSrc, outSrc in zip(inCat, outCat):
            self.assertEqual(outSrc[PLUGIN_NAME], inSrc[PLUGIN_NAME])
            self.assertEqual(outSrc[PLUGIN_NAME + "_transform"], inSrc[PLUGIN_NAME] * -1.0)
            for field in transformTask.config.toDict()['copyFields']:
                self.assertEqual(outSrc.get(field), inSrc.get(field))

        # Check that the wcs and calib objects were accessed once per transform.
        self.assertEqual(wcs.count, len(transformTask.transforms))
        self.assertEqual(calib.count, len(transformTask.transforms))

    def testSingleFrameMeasurementTransform(self):
        """Test applying a transform task to the results of single frame measurement."""
        schema = afwTable.SourceTable.makeMinimalSchema()
        sfmConfig = measBase.SingleFrameMeasurementConfig(plugins=[PLUGIN_NAME])
        # We don't use slots in this test
        for key in sfmConfig.slots.keys():
            setattr(sfmConfig.slots, key, None)
        sfmTask = measBase.SingleFrameMeasurementTask(schema, config=sfmConfig)
        transformTask = TransformTask(measConfig=sfmConfig,
                                      pluginRegistry=measBase.sfm.SingleFramePlugin.registry,
                                      inputSchema=sfmTask.schema)
        self._transformAndCheck(sfmConfig, sfmTask.schema, transformTask)

    def testForcedMeasurementTransform(self):
        """Test applying a transform task to the results of forced measurement."""
        schema = afwTable.SourceTable.makeMinimalSchema()
        forcedConfig = measBase.ForcedMeasurementConfig(plugins=[PLUGIN_NAME])
        # We don't use slots in this test
        for key in forcedConfig.slots.keys():
            setattr(forcedConfig.slots, key, None)
        forcedTask = measBase.ForcedMeasurementTask(schema, config=forcedConfig)
        transformConfig = TransformConfig(copyFields=("objectId", "coord"))
        transformTask = TransformTask(measConfig=forcedConfig,
                                      pluginRegistry=measBase.forcedMeasurement.ForcedPlugin.registry,
                                      inputSchema=forcedTask.schema, config=transformConfig)
        self._transformAndCheck(forcedConfig, forcedTask.schema, transformTask)


@contextlib.contextmanager
def tempDirectory(*args, **kwargs):
    """A context manager which provides a temporary directory and automatically cleans up when done."""
    dirname = tempfile.mkdtemp(*args, **kwargs)
    yield dirname
    shutil.rmtree(dirname, ignore_errors=True)


class RunTransformTestCase(utilsTests.TestCase):
    def testInterface(self):
        obsTestDir = eups.productDir("obs_test")
        if obsTestDir is None:
            raise RuntimeError("obs_test must be setup")
        inputDir = os.path.join(obsTestDir, "data", "input")

        # Configure a ProcessCcd task such that it will return a minimal
        # number of measurements plus our test plugin.
        cfg = ProcessCcdConfig()
        cfg.measurement.value.plugins = ["base_SdssCentroid", "base_SkyCoord", PLUGIN_NAME]
        cfg.measurement.value.slots.shape = None
        cfg.measurement.value.slots.psfFlux = None
        cfg.measurement.value.slots.apFlux = None
        cfg.measurement.value.slots.instFlux = None
        cfg.measurement.value.slots.modelFlux = None

        # Process the test data with ProcessCcd then perform a transform.
        with tempDirectory() as tempDir:
            measResult = ProcessCcdTask.parseAndRun(args=[inputDir, "--output", tempDir, "--id", "visit=1"],
                                                    config=cfg, doReturnResults=True)
            trResult = RunTransformTask.parseAndRun(args=[tempDir, "--id", "visit=1"], doReturnResults=True)

        measSrcs = measResult.resultList[0].result.sources
        trSrcs = trResult.resultList[0].result

        # The length of the measured and transformed catalogs should be the same.
        self.assertEqual(len(measSrcs), len(trSrcs))

        # Each source should have been measured & transformed appropriately.
        for measSrc, trSrc in zip(measSrcs, trSrcs):
            # The TrivialMeasurement should be transformed as defined above.
            self.assertEqual(trSrc[PLUGIN_NAME], measSrc[PLUGIN_NAME])
            self.assertEqual(trSrc[PLUGIN_NAME + "_transform"], -1.0 * measSrc[PLUGIN_NAME])

            # The SdssCentroid should be transformed to celestial coordinates.
            # Checking that the full transformation has been done correctly is
            # out of scope for this test case; we just ensure that there's
            # plausible position in the transformed record.
            trCoord = afwTable.CoordKey(trSrcs.schema["base_SdssCentroid"]).get(trSrc)
            self.assertAlmostEqual(measSrc.getCoord().getLongitude(), trCoord.getLongitude())
            self.assertAlmostEqual(measSrc.getCoord().getLatitude(), trCoord.getLatitude())


def suite():
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(TransformTestCase)
    suites += unittest.makeSuite(RunTransformTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
