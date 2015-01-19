#!/usr/bin/env python

#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

import os
import sys
import numpy as np

import eups
import lsst.afw.table              as afwTable
import lsst.afw.image              as afwImage
import lsst.meas.algorithms        as measAlg
from lsst.meas.algorithms.detection import SourceDetectionTask
from lsst.meas.base                 import SingleFrameMeasurementTask
from lsst.pipe.tasks.calibrateTask  import CalibrateTask
from lsst.pex.exceptions            import NotFoundError

def loadData():
    """Prepare the data we need to run the example"""

    # Load sample input from disk
    mypath = eups.productDir("afwdata")
    if not mypath:
        print >> sys.stderr, "Please setup afwdata and try again"
        sys.exit(1)

    imFile = os.path.join(mypath, "CFHT", "D4", "cal-53535-i-797722_small_1.fits")

    exposure = afwImage.ExposureF(imFile)

    # The old (meas_algorithms) SdssCentroid assumed this by default if it
    # wasn't specified; meas_base requires us to be explicit.
    psf = measAlg.DoubleGaussianPsf(11, 11, 0.01)
    exposure.setPsf(psf)

    im = exposure.getMaskedImage().getImage()
    im -= float(np.median(im.getArray()))

    return exposure

def run(display=False):
    exposure = loadData()
    schema = afwTable.SourceTable.makeMinimalSchema()
    #
    # Create the detection and measurement Tasks
    #
    config = SourceDetectionTask.ConfigClass()
    config.reEstimateBackground = False
    detectionTask = SourceDetectionTask(config=config, schema=schema)

    config = SingleFrameMeasurementTask.ConfigClass()
    # Use the minimum set of plugins required for testing
    config.plugins.names.clear()
    config.plugins.names.add("base_PeakCentroid")
    config.slots.centroid = "base_PeakCentroid"
    measureTask = SingleFrameMeasurementTask(schema, config=config)
    #
    # Create the output table
    #
    tab = afwTable.SourceTable.make(schema)
    #
    # Process the data
    #
    sources = detectionTask.run(tab, exposure, sigma=2).sources
    measureTask.measure(exposure, sources)

    config = CalibrateTask.ConfigClass()
    calibTask = CalibrateTask(config=config)
    newSources = calibTask.run(sources, measureTask.plugins).sources

    print len(sources), len(newSources)
    print sources.schema.compare(newSources.schema, sources.schema.EQUAL_KEYS)
    print sources.schema.compare(newSources.schema, sources.schema.EQUAL_NAMES)
    print sources.schema.compare(newSources.schema, sources.schema.EQUAL_DOCS)
    print sources.schema.compare(newSources.schema, sources.schema.EQUAL_UNITS)
    print sources.schema.compare(newSources.schema, sources.schema.IDENTICAL)
    print set(sources.schema.getNames()) - set(newSources.schema.getNames())
    print sources.schema == newSources.schema
    print sources.schema
    print newSources.schema

    print sources[0].getCentroid()
#    print sources[0].getPsfFlux()
    print newSources[0].getCentroid()
#    print newSources[0].getPsfFlux()

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Demonstrate the use of CalibrateTask")
    args = parser.parse_args()
    run()
