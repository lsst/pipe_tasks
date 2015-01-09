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
import lsst.daf.base               as dafBase
import lsst.afw.table              as afwTable
import lsst.afw.image              as afwImage
import lsst.afw.display.ds9        as ds9
import lsst.meas.algorithms        as measAlg
from lsst.meas.algorithms.detection import SourceDetectionTask
from lsst.meas.algorithms.measurement import SourceMeasurementTask
from lsst.pipe.tasks.measurePsf     import MeasurePsfTask

def loadData():
    """Prepare the data we need to run the example"""

    # Load sample input from disk
    mypath = eups.productDir("afwdata")
    if not mypath:
        print >> sys.stderr, "Please setup afwdata and try again"
        sys.exit(1)

    imFile = os.path.join(mypath, "CFHT", "D4", "cal-53535-i-797722_small_1.fits")

    exposure = afwImage.ExposureF(imFile)

    im = exposure.getMaskedImage().getImage()
    im -= float(np.median(im.getArray()))

    return exposure

def run(display=False):
    exposure = loadData()
    schema = afwTable.SourceTable.makeMinimalSchema()
    schema.setVersion(0)
    #
    # Create the detection and measurement Tasks
    #
    config = SourceDetectionTask.ConfigClass()
    config.reEstimateBackground = False
    detectionTask = SourceDetectionTask(config=config, schema=schema)

    config = SourceMeasurementTask.ConfigClass()
    config.slots.psfFlux = "flux.sinc"  # use of the psf flux is hardcoded in secondMomentStarSelector
    measureTask = SourceMeasurementTask(schema, config=config)
    #
    # Create the measurePsf task
    #
    config = MeasurePsfTask.ConfigClass()

    starSelector = config.starSelector.apply()
    starSelector.config.badFlags = ["flags.pixel.edge",  "flags.pixel.cr.center",
                                    "flags.pixel.interpolated.center", "flags.pixel.saturated.center"]

    psfDeterminer = config.psfDeterminer.apply()
    psfDeterminer.config.sizeCellX = 128
    psfDeterminer.config.sizeCellY = 128
    psfDeterminer.config.spatialOrder = 1
    psfDeterminer.config.nEigenComponents = 3
    
    measurePsfTask = MeasurePsfTask(config=config, schema=schema)
    #
    # Create the output table
    #
    tab = afwTable.SourceTable.make(schema)
    #
    # Process the data
    #
    sources = detectionTask.run(tab, exposure, sigma=2).sources
    measureTask.measure(exposure, sources)

    result = measurePsfTask.run(exposure, sources)
    psf = result.psf
    cellSet = result.cellSet

    if display:                         # display on ds9 (see also --debug argparse option)
        frame = 1
        ds9.mtv(exposure, frame=frame)

        with ds9.Buffering():
            for s in sources:
                xy = s.getCentroid()
                ds9.dot('+', *xy, frame=frame)
                if s.get("calib.psf.candidate"):
                    ds9.dot('x', *xy, ctype=ds9.YELLOW, frame=frame)
                if s.get("calib.psf.used"):
                    ds9.dot('o', *xy, size=4, ctype=ds9.RED, frame=frame)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Demonstrate the use of MeasurePsfTask")

    parser.add_argument('--debug', '-d', action="store_true", help="Load debug.py?", default=False)
    parser.add_argument('--ds9', action="store_true", help="Display sources on ds9", default=False)

    args = parser.parse_args()

    if args.debug:
        try:
            import debug
        except ImportError as e:
            print >> sys.stderr, e

    run(display=args.ds9)
