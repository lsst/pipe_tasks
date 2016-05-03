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
"""Example use of DetectAndMeasureTask
"""

import os
import numpy as np

import lsst.utils
from lsst.daf.butlerUtils import ExposureIdInfo
from lsst.afw.detection import GaussianPsf
import lsst.afw.image as afwImage
from lsst.meas.astrom import displayAstrometry
from lsst.meas.algorithms import estimateBackground
from lsst.pipe.tasks.calibrate import DetectAndMeasureTask
from lsst.pipe.tasks.repair import RepairTask

np.random.seed(1)

FilterName = "r"

def loadData(psfSigma=1.5):
    """Prepare the data we need to run the example"""

    # Load sample input from disk
    mypath = lsst.utils.getPackageDir('afwdata')
    imFile = os.path.join(mypath, "CFHT", "D4", "cal-53535-i-797722_small_1.fits")

    exposure = afwImage.ExposureF(imFile)
    # add a filter
    afwImage.Filter.define(afwImage.FilterProperty(FilterName, 600, True))
    exposure.setFilter(afwImage.Filter(FilterName))
    # add a simple Gaussian PSF model
    psfModel = GaussianPsf(11, 11, psfSigma)
    exposure.setPsf(psfModel)

    return exposure


def run(display=False):
    """Subtract background, mask cosmic rays, then detect and measure
    """
    # Create the tasks; note that background estimation is performed by a function,
    # not a task, though it has a config
    repairConfig = RepairTask.ConfigClass()
    repairTask = RepairTask(config=repairConfig)

    backgroundConfig = estimateBackground.ConfigClass()

    damConfig = DetectAndMeasureTask.ConfigClass()
    damConfig.detection.thresholdValue = 5.0
    damConfig.detection.includeThresholdMultiplier = 1.0
    damConfig.measurement.doApplyApCorr = "yes"
    detectAndMeasureTask = DetectAndMeasureTask(config=damConfig)

    # load the data
    # Exposure ID and the number of bits required for exposure IDs are usually obtained from a data repo,
    # but here we pick reasonable values (there are 64 bits to share between exposure IDs and source IDs).
    exposure = loadData()
    exposureIdInfo = ExposureIdInfo(expId=1, expBits=5)

    # repair cosmic rays
    repairTask.run(exposure=exposure)

    # subtract an initial estimate of background level
    estBg, exposure = estimateBackground(
        exposure = exposure,
        backgroundConfig = backgroundConfig,
        subtract = True,
    )

    # detect and measure
    damRes = detectAndMeasureTask.run(exposure=exposure, exposureIdInfo=exposureIdInfo)
    if display:
        displayAstrometry(frame=2, exposure=damRes.exposure, sourceCat=damRes.sourceCat, pause=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Demonstrate the use of DetectAndMeasureTask")

    parser.add_argument('--display', action="store_true",
        help="Display the output image and source catalog", default=False)

    args = parser.parse_args()

    run(display=args.display)
