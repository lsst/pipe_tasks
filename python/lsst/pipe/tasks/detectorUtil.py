#!/usr/bin/env python

import lsst.afw.cameraGeom as cameraGeom

def detectorIsCcd(exposure):
    """Is the detector referred to by the exposure a Ccd?

    @param exposure Exposure to inspect
    @returns True if exposure's detector is a Ccd
    """
    det = exposure.getDetector()
    ccd = cameraGeom.cast_Ccd(det)
    return False if ccd is None else True

def detectorIsAmp(exposure):
    """Is the detector referred to by the exposure an Amp?

    @param exposure Exposure to inspect
    @returns True if exposure's detector is an Amp
    """
    det = exposure.getDetector()
    amp = cameraGeom.cast_Amp(det)
    return False if amp is None else True

def getCcd(exposure, allowRaise=True):
    """Get the Ccd referred to by an exposure

    @param exposure Exposure to inspect
    @param allowRaise  If False, return None if the CCD can't be found rather than raising an exception
    @returns Ccd
    """
    det = exposure.getDetector()
    ccd = cameraGeom.cast_Ccd(det)
    if ccd is not None:
        return ccd
    amp = cameraGeom.cast_Amp(det)
    if amp is not None:
        det = amp.getParent()
        ccd = cameraGeom.cast_Ccd(det)
        return ccd

    if allowRaise:
        raise RuntimeError("Can't find Ccd from detector.")
    else:
        return None

def getAmp(exposure):
    """Get the Amp referred to by an exposure

    @param exposure Exposure to inspect
    @returns Amp
    """
    det = exposure.getDetector()
    amp = cameraGeom.cast_Amp(det)      # None if detector is not an Amp
    return amp

def haveAmp(exposure, amp):
    """Does the detector referred to by the exposure contain this particular amp?

    @param exposure Exposure to inspect
    @param amp Amp for comparison
    @returns True if exposure contains this amp
    """
    det = exposure.getDetector()
    testAmp = cameraGeom.cast_Amp(det)
    if testAmp is None:
        # Exposure contains a CCD, which contains all its amps
        return True
    return True if testAmp.getId() == amp.getId() else False
