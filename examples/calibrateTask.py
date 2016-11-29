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

from __future__ import print_function
import os
import sys
import numpy as np

import lsst.utils
import lsst.pipe.base as pipeBase
from lsst.obs.base import ExposureIdInfo
import lsst.afw.display as afwDisplay
import lsst.afw.table as afwTable
import lsst.afw.image as afwImage
from lsst.pex.config import Config
from lsst.afw.coord import Coord
from lsst.afw.geom import PointD
from lsst.meas.algorithms import LoadReferenceObjectsTask
from lsst.meas.astrom import createMatchMetadata

from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask
from lsst.pipe.tasks.calibrate import CalibrateTask

np.random.seed(1)

FilterName = "r"


def loadData(pixelScale=1.0):
    """Prepare the data we need to run the example"""

    # Load sample input from disk
    mypath = lsst.utils.getPackageDir('afwdata')
    imFile = os.path.join(mypath, "CFHT", "D4", "cal-53535-i-797722_small_1.fits")

    visitInfo = afwImage.makeVisitInfo(exposureTime=1.0)
    exposure = afwImage.ExposureF(imFile, visitInfo)
    # add a filter
    afwImage.Filter.define(afwImage.FilterProperty(FilterName, 600, True))
    exposure.setFilter(afwImage.Filter(FilterName))
    # and a trivial WCS (needed by MyAstrometryTask)
    pixelScale /= 3600.0  # degrees per pixel
    wcs = afwImage.makeWcs(Coord(PointD(15, 1)), PointD(0, 0), pixelScale, 0.0, 0.0, pixelScale)
    exposure.setWcs(wcs)

    return exposure


class MyAstrometryTask(pipeBase.Task):
    """An override for CalibrateTask's astrometry task that fakes a solution"""
    ConfigClass = Config
    _defaultName = "astrometry"

    def __init__(self, refObjLoader, schema=None, **kwargs):
        pipeBase.Task(**kwargs)

    def run(self, exposure, sourceCat):
        """Fake an astrometric solution

        Pretend the current solution is perfect
        and make a reference catalog that matches the source catalog
        """
        return self.loadAndMatch(exposure=exposure, sourceCat=sourceCat)

    def loadAndMatch(self, exposure, sourceCat):
        """!Fake loading and matching

        Copy the source catalog to a reference catalog and producing a match list
        """
        wcs = exposure.getWcs()
        refSchema = LoadReferenceObjectsTask.makeMinimalSchema(
            filterNameList=[FilterName],
            addIsPhotometric=True,
        )
        refCat = afwTable.SimpleCatalog(refSchema)
        refFluxKey = refSchema[FilterName + "_flux"].asKey()
        refIsPhotoKey = refSchema["photometric"].asKey()

        matches = lsst.afw.table.ReferenceMatchVector()
        for src in sourceCat:
            flux = 1e-3*src.getPsfFlux()*np.random.normal(1.0, 2e-2)
            refObj = refCat.addNew()
            refObj.set(refFluxKey, flux)
            refObj.setCoord(wcs.pixelToSky(src.getCentroid()))
            refObj.set(refIsPhotoKey, True)
            match = lsst.afw.table.ReferenceMatch(refObj, src, 0)
            matches.append(match)

        return pipeBase.Struct(
            refCat=refCat,
            matches=matches,
            matchMeta=createMatchMetadata(exposure),
        )


def run(display=False):
    #
    # Create the tasks
    #
    charImageConfig = CharacterizeImageTask.ConfigClass()
    charImageTask = CharacterizeImageTask(config=charImageConfig)

    config = CalibrateTask.ConfigClass()
    config.astrometry.retarget(MyAstrometryTask)
    calibrateTask = CalibrateTask(config=config)

    # load the data
    # Exposure ID and the number of bits required for exposure IDs are usually obtained from a data repo,
    # but here we pick reasonable values (there are 64 bits to share between exposure IDs and source IDs).
    exposure = loadData()
    exposureIdInfo = ExposureIdInfo(expId=1, expBits=5)

    # characterize the exposure to repair cosmic rays and fit a PSF model
    # display now because CalibrateTask modifies the exposure in place
    charRes = charImageTask.characterize(exposure=exposure, exposureIdInfo=exposureIdInfo)
    if display:
        displayFunc(charRes.exposure, charRes.sourceCat, frame=1)

    # calibrate the exposure
    calRes = calibrateTask.calibrate(exposure=charRes.exposure, exposureIdInfo=exposureIdInfo)
    if display:
        displayFunc(calRes.exposure, calRes.sourceCat, frame=2)


def displayFunc(exposure, sourceCat, frame):
    display = afwDisplay.getDisplay(frame)
    display.mtv(exposure)

    with display.Buffering():
        for s in sourceCat:
            xy = s.getCentroid()
            display.dot('+', *xy, ctype=afwDisplay.CYAN if s.get("flags_negative") else afwDisplay.GREEN)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Demonstrate the use of CalibrateTask")

    parser.add_argument('--debug', '-d', action="store_true", help="Load debug.py?", default=False)
    parser.add_argument('--display', action="store_true",
                        help="Display images in this example task (not using debug.py)", default=False)

    args = parser.parse_args()

    if args.debug:
        try:
            import debug
        except ImportError as e:
            print(e, file=sys.stderr)

    run(display=args.display)
