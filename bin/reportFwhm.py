#!/usr/bin/env python
import math
import os
import sys

import numpy

import lsst.pex.config as pexConfig
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.meas.algorithms as measAlg
import lsst.pipe.base as pipeBase

FWHMPerSigma = 2 * math.sqrt(2 * math.log(2))

def computeGaussianWidth(psf, bbox):
    """Measures the gaussian width at 5 points and returns the largest value
    """
    gaussWidthList = []
    for x in (bbox.getMinX(), bbox.getMaxX()):
        for y in (bbox.getMinY(), bbox.getMaxY()):
            psfAttr = measAlg.PsfAttributes(psf, x, y)
            gaussWidth = psfAttr.computeGaussianWidth()
            gaussWidthList.append(gaussWidth)
    return max(gaussWidthList)

def reportFwhmAndRaDec(dataRefList):
    """Print the maximum FWHM and center RA/Dec of each image, sorted by filter name and FWHM

    @param[in] dataRefList: list of sensor-level data references
    """
    begLen = len(dataRefList)
    print "Processing %d exposures" % (begLen,)
    reportInterval = max(len(dataRefList) / 80, 5)
    dataList = []
    for ind, dataRef in enumerate(dataRefList):
        dataId = dataRef.dataId
        try:
            exposure = dataRef.get("calexp")
            bbox = exposure.getBBox(afwImage.PARENT)
            filterName = exposure.getFilter().getName()
            psf = dataRef.get("psf")
            maxGaussWidth = computeGaussianWidth(psf, bbox)
            maxFwhm = FWHMPerSigma * maxGaussWidth

            floatBBox = afwGeom.Box2D(bbox)
            ctrPixArr = (numpy.array(floatBBox.getMax()) + numpy.array(floatBBox.getMin())) / 2.0
            ctrPixPos = afwGeom.Point2D(*ctrPixArr)
            ctrSkyPos = exposure.getWcs().pixelToSky(ctrPixPos).getPosition()

            dataList.append((filterName, maxFwhm, ctrSkyPos, dataId))
            sys.stdout.write("\r%d of %d" % (ind+1, begLen))
            sys.stdout.flush()
        except Exception, e:
            print "\nFailed on %s: %s" % (dataId, e)
            continue
    endLen = len(dataList)
    print "\nProcessed %d exposures (skipped %d)" % (endLen, endLen - begLen)
    print "ID\tFWHM\tRA\tDec"

    dataList.sort()
    for filterName, maxFwhm, ctrSkyPos, dataId in dataList:
        print "%s\t%0.2f\t%0.5f\t%0.5f" % (dataId, maxFwhm, ctrSkyPos[0], ctrSkyPos[1])
    

if __name__ == "__main__":
    config = pexConfig.Config() # no configurable options
    parser = pipeBase.ArgumentParser(name="reportFwhm")
    parsedCmd = parser.parse_args(config=config)
    
    reportFwhmAndRaDec(parsedCmd.dataRefList)
