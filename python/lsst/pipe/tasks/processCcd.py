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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

from lsst.ip.isr import IsrTask
from lsst.pipe.tasks.calibrate import CalibrateTask
from lsst.pipe.tasks.photometry import PhotometryTask


def guessCcdId(ampIdList):
    """Guess the identifier for the CCD from the list of identifiers for the CCD's amplifiers"""
    ccdId = dict(ampIdList[0])
    for ampId in ampIdList[1:]:
        for key, value in ampId.items():
            if not ccdId.has_key(key) or ccdId[key] != value:
                del ccdId[key]
    if len(ccdId.keys()) == 0:
        raise ValueError("Unable to determine CCD identifier from %s" % ampIdList)
    return ccdId


class ProcessCcdConfig(pexConfig.Config):
    """Config for ProcessCcd"""
    doIsr = pexConfig.Field(dtype=bool, default=True, doc = "Perform ISR?")
    doCalibrate = pexConfig.Field(dtype=bool, default=True, doc = "Perform calibration?")
    doPhotometry = pexConfig.Field(dtype=bool, default=True, doc = "Perform photometry?")
    doWriteIsr = pexConfig.Field(dtype=bool, default=True, doc = "Write ISR results?")
    doWriteCalibrate = pexConfig.Field(dtype=bool, default=True, doc = "Write calibration results?")
    doWritePhotometry = pexConfig.Field(dtype=bool, default=True, doc = "Write photometry results?")
    isr = pexConfig.ConfigField(dtype=IsrTask.ConfigClass, doc="Instrumental Signature Removal")
    calibrate = pexConfig.ConfigField(dtype=CalibrateTask.ConfigClass, doc="Calibration")
    photometry = pexConfig.ConfigField(dtype=IsrTask.ConfigClass, doc="Photometry")


class ProcessCcdTask(pipeBase.Task):
    """Process a CCD"""
    ConfigClass = ProcessCcdConfig

    def __init__(self, config=ProcessCcdConfig(), *args, **kwargs):
        config = ProcessCcdConfig.load("/home/price/LSST/pipe/tasks/config/suprimecam.py")
        pipeBase.Task.__init__(self, config=config, *args, **kwargs)
        self.makeSubtask("isr", IsrTask, config=config.isr)
        self.makeSubtask("calibrate", CalibrateTask, config=config.calibrate)
        self.makeSubtask("photometry", PhotometryTask, config=config.photometry)


    def runButler(self, butler, idList):
        assert butler and idList
        if self.config.doIsr:
            exposureList = list()
            for ident in idList:
                isrStruct = self.isr.runButler(butler, ident)
                exposureList.append(isrStruct.postIsrExposure)
                # XXX do something with metadata

            ccdExposure = self.isr.doCcdAssembly(exposureList)
            if self.config.doWriteIsr:
                butler.put('postISRCCD', ccdExposure)
        else:
            ccdExposure = None

        ccdId = guessCcdId(idList)

        if self.config.doCalibrate:
            if ccdExposure is None:
                ccdExposure = butler.get('postISRCCD', ccdId)
            calib = self.calibrate.run(ccdExposure)
            del ccdExposure
            if self.config.doWriteCalibrate:
                butler.put('calexp', calib.exposure, ccdId)
                butler.put('psf', calib.psf, ccdId)
                butler.put('icSrc', calib.sources, ccdId)
                butler.put('icMatch', calib.matches, ccdId)
                #butler.put('apcorr', calib.apcorr, ccdId)
        else:
            calib = None

        if self.config.photometry:
            if calib is None:
                exposure = butler.get('calexp', ccdId)
                psf = butler.get('psf', ccdId)
                apCorr = None # butler.get('apcorr', ccdId)
            else:
                exposure = calib.exposure
                psf = calib.psf
                apCorr = calib.apcorr
            phot = self.photometry.run(exposure, psf, apcorr=apCorr)
            if self.config.doWritePhotometry:
                butler.put('src', phot.sources, ccdId)
        else:
            phot = None

        return Struct(exposure=exposure, psf=psf, apCorr=apCorr,
                      sources=phot.sources if phot else None,
                      matches=calib.matches if calib else None,
                      matchMeta=calib.matchMeta if calib else None,
                      )
