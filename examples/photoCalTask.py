#!/usr/bin/env python
from __future__ import absolute_import, division
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

import numpy as np

import eups
import lsst.utils
import lsst.afw.table              as afwTable
import lsst.afw.image              as afwImage
from lsst.meas.astrom import AstrometryTask
from lsst.pipe.tasks.photoCal import PhotoCalTask

def loadData():
    """Prepare the data we need to run the example"""
    
    # Load sample input from disk
    mypath = lsst.utils.getPackageDir('meas_astrom')

    # The .xy.fits file has sources in the range ~ [0,2000],[0,4500]
    exposure = afwImage.ExposureF(os.path.join(mypath, "tests", "v695833-e0-c000-a00.sci.fits"))
    #
    # We're using a subset of the exposure in the .xy file and this appears to confuse
    # meas_astrom; it needs to be called as
    #        astrom.determineWcs(srcCat, exposure, imageSize=(2048, 4612))
    #
    # Rather than fixing this we'll fix the input image 
    #
    if True:
        smi = exposure.getMaskedImage()
        mi = smi.Factory(2048, 4612)
        mi[0:smi.getWidth(), 0:smi.getHeight()] = smi
        exposure.setMaskedImage(mi)
        del mi; del smi

    # Set up local astrometry_net_data
    datapath = os.path.join(mypath, 'tests', 'astrometry_net_data', 'photocal')
    eupsObj = eups.Eups(root=datapath)
    ok, version, reason = eupsObj.setup('astrometry_net_data')
    if not ok:
        raise ValueError("Need photocal version of astrometry_net_data (from path: %s): %s" %
                         (datapath, reason))
    #
    # Read sources
    #
    srcCat = afwTable.SourceCatalog.readFits(os.path.join(mypath, "tests", "v695833-e0-c000.xy.fits"))
    srcCat.getPsfFluxErr()[:] = np.sqrt(srcCat.getPsfFlux())

    return exposure, srcCat

def run():
    exposure, srcCat = loadData()
    schema = srcCat.getSchema()
    #
    # Create the astrometry task
    #
    config = AstrometryTask.ConfigClass()
    config.refObjLoader.filterMap = {"_unknown_": "r"}
    config.matcher.sourceFluxType = "Psf" # sample catalog does not contain aperture flux
    aTask = AstrometryTask(config=config)
    #
    # And the photometry Task
    #
    config = PhotoCalTask.ConfigClass()
    config.applyColorTerms = False      # we don't have any available, so this suppresses a warning
    pTask = PhotoCalTask(config=config, schema=schema)
    #
    # The tasks may have added extra elements to the schema (e.g. AstrometryTask's centroidKey to
    # handle distortion; photometryTask's config.outputField).  If this is so, we need to add
    # these columns to the Source table.
    #
    # We wouldn't need to do this if we created the schema prior to measuring the exposure,
    # but in this case we read the sources from disk
    #
    if schema != srcCat.getSchema():    # the tasks added fields
        print("Adding columns to the source catalogue")
        cat = afwTable.SourceCatalog(schema)
        cat.table.defineCentroid(srcCat.table.getCentroidDefinition())
        cat.table.definePsfFlux(srcCat.table.getPsfFluxDefinition())

        scm = afwTable.SchemaMapper(srcCat.getSchema(), schema)
        for schEl in srcCat.getSchema():
            scm.addMapping(schEl.getKey(), True)

        cat.extend(srcCat, True, scm)   # copy srcCat to cat, adding new columns

        srcCat = cat; del cat
    #
    # Process the data
    #
    matches = aTask.run(exposure, srcCat).matches
    result = pTask.run(exposure, matches)

    calib = result.calib
    fm0, fm0Err = calib.getFluxMag0()

    print("Used %d calibration sources out of %d matches" % (len(result.matches), len(matches)))
    
    delta = result.arrays.refMag - result.arrays.srcMag
    q25, q75 = np.percentile(delta, [25, 75])
    print("RMS error is %.3fmmsg (robust %.3f, Calib says %.3f)" % (np.std(delta), 0.741*(q75 - q25),
                                                                    2.5/np.log(10)*fm0Err/fm0))
            
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Demonstrate the use of PhotoCalTask")

    parser.add_argument('--debug', '-d', action="store_true", help="Load debug.py?", default=False)

    args = parser.parse_args()

    if args.debug:
        try:
            import debug
        except ImportError as e:
            print("Could not import debug: %s" % (e,))

    run()
