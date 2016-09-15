#!/usr/bin/env python

#
# LSST Data Management System
# Copyright 2014 LSST Corporation.
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
import numpy
import sys

import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.meas.algorithms as measAlg
from lsst.pipe.tasks.repair import RepairTask


def makeTestImage(xsize=200, ysize=100, nCR=15):

    randArr = numpy.random.poisson(1000., xsize*ysize)
    randArr = numpy.array(randArr.reshape(ysize, xsize), dtype=numpy.float32)  # force to ImageF
    factory = measAlg.GaussianPsfFactory()
    factory.addWing = False
    psf = factory.apply(4)  # FWHM in pixels

    img = afwImage.makeImageFromArray(randArr)
    var = afwImage.ImageF(img, True)  # copy constructor
    mask = afwImage.MaskU(xsize, ysize)

    xind = numpy.random.randint(0, xsize, nCR)
    yind = numpy.random.randint(0, ysize, nCR)

    # set some CRs
    for xi, yi in zip(xind, yind):
        xi, yi = int(xi), int(yi)
        img.set(xi, yi, 1e6)

    mi = afwImage.makeMaskedImage(img, mask, var)
    exp = afwImage.makeExposure(mi)
    exp.setPsf(psf)
    return exp


def addDefects(exp, nBadCols=10):
    img = exp.getMaskedImage().getImage()
    (xsize, ysize) = img.getDimensions()
    defectList = measAlg.DefectListT()

    # set some bad cols and add them to a defect list
    for xi in numpy.random.randint(0, xsize, nBadCols):
        yi = numpy.random.randint(0, ysize)
        xi, yi = int(xi), int(yi)
        bbox = afwGeom.Box2I(afwGeom.PointI(xi, 0), afwGeom.ExtentI(1, yi+1))
        subIm = afwImage.ImageF(img, bbox)
        subIm.set(1e7)
        defectList.push_back(measAlg.Defect(bbox))
    # set a 15 pixel box of defects at the upper left corner to demonstrate fallbackValue
    bbox = afwGeom.Box2I(afwGeom.PointI(0, ysize-15), afwGeom.ExtentI(15, 15))
    subIm = afwImage.ImageF(img, bbox)
    subIm.set(1e7)
    defectList.push_back(measAlg.Defect(bbox))
    return defectList


def runRepair(exp, defectList):
    repair = RepairTask(name="RepairTask")
    repair.run(exp, defects=defectList)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Demonstrate the use of RepairTask")

    parser.add_argument('--debug', '-d', action="store_true", help="Load debug.py?", default=False)
    parser.add_argument('--randSeed', '-s', type=int, help="Seed for the random number generator.",
                        default=1)

    args = parser.parse_args()

    numpy.random.seed(args.randSeed)

    if args.debug:
        try:
            import debug
        except ImportError as e:
            print >> sys.stderr, e

    exp = makeTestImage()
    defectList = addDefects(exp)
    runRepair(exp, defectList)
