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
"""
import unittest

import lsst.utils
import lsst.afw.table as afwTable
import lsst.geom as geom
import lsst.meas.base as measBase
import lsst.utils.tests
from lsst.pipe.tasks.transformMeasurement import TransformConfig, TransformTask

PLUGIN_NAME = "base_TrivialMeasurement"

# Rather than providing real WCS and calibration objects to the
# transformation, we use this simple placeholder to keep track of the number
# of times it is accessed.


class Placeholder:

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
        for key, field in mapper.getInputSchema().extract(name + "*").values():
            mapper.addMapping(key)
        self.key = mapper.editOutputSchema().addField(name + "_transform", type="D", doc="transformed dummy")

    def __call__(self, inputCatalog, outputCatalog, wcs, photoCalib):
        """Transform inputCatalog to outputCatalog.

        We update the wcs and photoCalib placeholders to indicate that they have
        been seen in the transformation, but do not use their values.

        @param[in]  inputCatalog  SourceCatalog of measurements for transformation.
        @param[out] outputCatalog BaseCatalog of transformed measurements.
        @param[in]  wcs           Dummy WCS information; an instance of Placeholder.
        @param[in]  photoCalib         Dummy calibration information; an instance of Placeholder.
        """
        if hasattr(wcs, "increment"):
            wcs.increment()
        if hasattr(photoCalib, "increment"):
            photoCalib.increment()
        inColumns = inputCatalog.getColumnView()
        outColumns = outputCatalog.getColumnView()
        outColumns[self.key] = -1.0 * inColumns[self.name]


class TrivialMeasurementBase:

    """Default values for a trivial measurement plugin, subclassed below"""
    @staticmethod
    def getExecutionOrder():
        return 0

    @staticmethod
    def getTransformClass():
        return TrivialMeasurementTransform

    def measure(self, measRecord, exposure):
        measRecord.set(self.key, 1.0)


@measBase.register(PLUGIN_NAME)
class SFTrivialMeasurement(TrivialMeasurementBase, measBase.sfm.SingleFramePlugin):

    """Single frame version of the trivial measurement"""

    def __init__(self, config, name, schema, metadata):
        measBase.sfm.SingleFramePlugin.__init__(self, config, name, schema, metadata)
        self.key = schema.addField(name, type="D", doc="dummy field")


@measBase.register(PLUGIN_NAME)
class ForcedTrivialMeasurement(TrivialMeasurementBase, measBase.forcedMeasurement.ForcedPlugin):

    """Forced frame version of the trivial measurement"""

    def __init__(self, config, name, schemaMapper, metadata):
        measBase.forcedMeasurement.ForcedPlugin.__init__(self, config, name, schemaMapper, metadata)
        self.key = schemaMapper.editOutputSchema().addField(name, type="D", doc="dummy field")


class TransformTestCase(lsst.utils.tests.TestCase):

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
        r.setCoord(geom.SpherePoint(0.0, 11.19, geom.degrees))
        r[PLUGIN_NAME] = 1.0

        wcs, photoCalib = Placeholder(), Placeholder()
        outCat = transformTask.run(inCat, wcs, photoCalib)

        # Check that all sources have been transformed appropriately.
        for inSrc, outSrc in zip(inCat, outCat):
            self.assertEqual(outSrc[PLUGIN_NAME], inSrc[PLUGIN_NAME])
            self.assertEqual(outSrc[PLUGIN_NAME + "_transform"], inSrc[PLUGIN_NAME] * -1.0)
            for field in transformTask.config.toDict()['copyFields']:
                self.assertEqual(outSrc.get(field), inSrc.get(field))

        # Check that the wcs and photoCalib objects were accessed once per transform.
        self.assertEqual(wcs.count, len(transformTask.transforms))
        self.assertEqual(photoCalib.count, len(transformTask.transforms))

    def testSingleFrameMeasurementTransform(self):
        """Test applying a transform task to the results of single frame measurement."""
        schema = afwTable.SourceTable.makeMinimalSchema()
        sfmConfig = measBase.SingleFrameMeasurementConfig(plugins=[PLUGIN_NAME])
        # We don't use slots in this test
        for key in sfmConfig.slots:
            setattr(sfmConfig.slots, key, None)
        sfmTask = measBase.SingleFrameMeasurementTask(schema, config=sfmConfig)
        transformTask = TransformTask(measConfig=sfmConfig,
                                      inputSchema=sfmTask.schema, outputDataset="src")
        self._transformAndCheck(sfmConfig, sfmTask.schema, transformTask)

    def testForcedMeasurementTransform(self):
        """Test applying a transform task to the results of forced measurement."""
        schema = afwTable.SourceTable.makeMinimalSchema()
        forcedConfig = measBase.ForcedMeasurementConfig(plugins=[PLUGIN_NAME])
        # We don't use slots in this test
        for key in forcedConfig.slots:
            setattr(forcedConfig.slots, key, None)
        forcedConfig.copyColumns = {"id": "objectId", "parent": "parentObjectId"}
        forcedTask = measBase.ForcedMeasurementTask(schema, config=forcedConfig)
        transformConfig = TransformConfig(copyFields=("objectId", "coord_ra", "coord_dec"))
        transformTask = TransformTask(measConfig=forcedConfig,
                                      inputSchema=forcedTask.schema, outputDataset="forced_src",
                                      config=transformConfig)
        self._transformAndCheck(forcedConfig, forcedTask.schema, transformTask)


class MyMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
