import numpy as np
import lsst.afw.detection as afwDetect
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math  as afwMath
import lsst.afw.display.ds9 as ds9

nAmp = 4

def getXPos(width, hwidth, x):
    """Return the amp that x is in, and the positions of its image in each amplifier"""
    amp = x//(hwidth//2)                # which amp am I in?  Assumes nAmp == 4
    assert nAmp == 4
    assert amp in range(nAmp)

    if amp == 0:
        xa = x                          # distance to amp
        xs = hwidth - x - 1             # symmetrical position within this half of the chip
        xx = (x, xs, hwidth + xa, hwidth + xs)
    elif amp == 1:
        xa = hwidth - x - 1             # distance to amp
        xs = hwidth - x                 # symmetrical position within this half of the chip
        xx = (xs - 1, x, hwidth + xs - 1, hwidth + x)
    elif amp == 2:
        xa = x - hwidth                 # distance to amp
        xs = width - x                  # symmetrical position within this half of the chip
        xx = (xa, width - x - 1, x, width - xa - 1)
    elif amp == 3:
        xa = x - hwidth                 # distance to amp
        xs = width - x                  # symmetrical position within this half of the chip
        xx = (width - x - 1, xa, width - xa - 1, x)

    return amp, xx

def subtractXTalk(mi, coeffs, minPixelToMask=45000, crosstalkStr="CROSSTALK"):
    """Subtract the crosstalk from MaskedImage mi given a set of coefficients

The pixels affected by signal over minPixelToMask have the crosstalkStr bit set
    """
    sctrl = afwMath.StatisticsControl()
    sctrl.setAndMask(mi.getMask().getPlaneBitMask("DETECTED"))
    bkgd = afwMath.makeStatistics(mi, afwMath.MEDIAN, sctrl).getValue()
    #
    # These are the pixels that are bright enough to cause crosstalk (more precisely,
    # the ones that we label as causing crosstalk; in reality all pixels cause crosstalk)
    #
    tempStr = "TEMP"                    # mask plane used to record the bright pixels that we need to mask
    mi.getMask().addMaskPlane(tempStr)
    fs = afwDetect.FootprintSet(mi, afwDetect.Threshold(minPixelToMask), tempStr)
    
    mi.getMask().addMaskPlane(crosstalkStr)
    ds9.setMaskPlaneColor(crosstalkStr, ds9.MAGENTA)
    fs.setMask(mi.getMask(), crosstalkStr) # the crosstalkStr bit will now be set whenever we subtract crosstalk
    crosstalk = mi.getMask().getPlaneBitMask(crosstalkStr)
    
    width, height = mi.getDimensions()
    for i in range(nAmp):
        bbox = afwGeom.BoxI(afwGeom.PointI(i*(width//nAmp), 0), afwGeom.ExtentI(width//nAmp, height))
        ampI = mi.Factory(mi, bbox)
        for j in range(nAmp):
            if i == j:
                continue

            bbox = afwGeom.BoxI(afwGeom.PointI(j*(width//nAmp), 0), afwGeom.ExtentI(width//nAmp, height))
            if (i + j)%2 == 1:
                ampJ = afwMath.flipImage(mi.Factory(mi, bbox), True, False) # no need for a deep copy
            else:
                ampJ = mi.Factory(mi, bbox, afwImage.LOCAL, True)

            msk = ampJ.getMask()
            msk &= crosstalk
                
            ampJ -= bkgd
            ampJ *= coeffs[j][i]

            ampI -= ampJ
    #
    # Clear the crosstalkStr bit in the original bright pixels, where tempStr is set
    #
    msk = mi.getMask()
    temp = msk.getPlaneBitMask(tempStr)
    xtalk_temp = crosstalk | temp
    np_msk = msk.getArray()
    np_msk[np.where(np.bitwise_and(np_msk, xtalk_temp) == xtalk_temp)] &= ~crosstalk

    msk.removeAndClearMaskPlane(tempStr, True)
