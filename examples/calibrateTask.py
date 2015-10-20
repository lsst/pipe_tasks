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

import lsst.utils
import lsst.pipe.base              as pipeBase
import lsst.daf.base               as dafBase
import lsst.afw.coord              as afwCoord
import lsst.afw.geom               as afwGeom
import lsst.afw.table              as afwTable
import lsst.afw.image              as afwImage
import lsst.afw.display.ds9        as ds9
from lsst.meas.astrom import AstrometryTask
from lsst.pipe.tasks.calibrate import CalibrateTask

np.random.seed(1)

def loadData(pixelScale=1.0):
    """Prepare the data we need to run the example"""

    # Load sample input from disk
    mypath = lsst.utils.getPackageDir('afwdata')
    imFile = os.path.join(mypath, "CFHT", "D4", "cal-53535-i-797722_small_1.fits")

    exposure = afwImage.ExposureF(imFile)
    # set the exposure time
    calib = afwImage.Calib()
    calib.setExptime(1.0)
    exposure.setCalib(calib)
    # add a filter
    filterName = "r"
    afwImage.Filter.define(afwImage.FilterProperty(filterName, 600, True))
    exposure.setFilter(afwImage.Filter(filterName))
    # and a trivial WCS (needed by MyAstrometryTask)
    pixelScale /= 3600.0                # degrees per pixel
    wcs = afwImage.makeWcs(afwCoord.Coord(afwGeom.PointD(15, 1)), afwGeom.PointD(0, 0),
                           pixelScale, 0.0, 0.0, pixelScale)
    exposure.setWcs(wcs)

    return exposure

class MyAstrometryTask(AstrometryTask):
    """An override for CalibrateTask's astrometry task"""
    def __init__(self, *args, **kwargs):
        super(MyAstrometryTask, self).__init__(*args, **kwargs)

    def run(self, exposure, sourceCat):
        """My run method that totally fakes the astrometric solution"""

        filterName = exposure.getFilter().getName()
        wcs = exposure.getWcs()
        #
        # Fake a reference catalogue by copying fluxes from the list of Sources
        #
        schema = afwTable.SimpleTable.makeMinimalSchema()
        schema.addField(afwTable.Field[float]("{}_flux".format(filterName), "Reference flux"))
        schema.addField(afwTable.Field[float]("photometric", "I am a reference star"))
        refCat = afwTable.SimpleCatalog(schema)

        for s in sourceCat:
            m = refCat.addNew()
            flux = 1e-3*s.getPsfFlux()*np.random.normal(1.0, 2e-2)
            m.set("{}_flux".format(filterName), flux)
            m.setCoord(wcs.pixelToSky(s.getCentroid()))

        refCat.get("photometric")[:] = True
        #
        # Perform the "match"
        #
        matches = []
        md = dafBase.PropertyList()
        for m, s in zip(refCat, sourceCat):
            matches.append(afwTable.ReferenceMatch(m, s, 0.0))

        return pipeBase.Struct(
            matches=matches,
            matchMeta=md
            )

def run(display=False):
    #
    # Create the task
    #
    config = CalibrateTask.ConfigClass()
    config.initialPsf.pixelScale = 0.185 # arcsec per pixel
    config.initialPsf.fwhm = 1.0
    config.astrometry.retarget(MyAstrometryTask)
    calibrateTask = CalibrateTask(config=config)
    #
    # Process the data
    #
    exposure = loadData(config.initialPsf.pixelScale)
    result = calibrateTask.run(exposure)

    exposure0, exposure = exposure, result.exposure
    sources = result.sources

    if display:                         # display on ds9 (see also --debug argparse option)
        frame = 1
        ds9.mtv(exposure, frame=frame)

        with ds9.Buffering():
            for s in sources:
                xy = s.getCentroid()
                ds9.dot('+', *xy, ctype=ds9.CYAN if s.get("flags_negative") else ds9.GREEN, frame=frame)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Demonstrate the use of CalibrateTask")

    parser.add_argument('--debug', '-d', action="store_true", help="Load debug.py?", default=False)
    parser.add_argument('--ds9', action="store_true", help="Display sources on ds9", default=False)

    args = parser.parse_args()

    if args.debug:
        try:
            import debug
        except ImportError as e:
            print >> sys.stderr, e

    run(display=args.ds9)
