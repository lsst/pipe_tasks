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
import lsst.afw.detection as afwDet

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
    photometry = pexConfig.ConfigField(dtype=PhotometryTask.ConfigClass, doc="Photometry")


class ProcessCcdTask(pipeBase.Task):
    """Process a CCD"""
    ConfigClass = ProcessCcdConfig

    def __init__(self, config=ProcessCcdConfig(), *args, **kwargs):
        pipeBase.Task.__init__(self, config=config, *args, **kwargs)
        self.makeSubtask("isr", IsrTask, config=config.isr)
        self.makeSubtask("calibrate", CalibrateTask, config=config.calibrate)
        self.makeSubtask("photometry", PhotometryTask, config=config.photometry)


    def runButler(self, butler, dataIdList):
        assert butler and dataIdList
        if self.config.doIsr:
            exposureList = list()
            for ident in dataIdList:
                isrStruct = self.isr.runButler(butler, ident)
                exposureList.append(isrStruct.postIsrExposure)
                self.display('isr', exposure=isrStruct.postIsrExposure, pause=True)
                # XXX do something with metadata

            ccdExposure = self.isr.doCcdAssembly(exposureList)
            del exposureList
            self.display('ccdAssembly', exposure=ccdExposure)
            if self.config.doWriteIsr:
                butler.put('postISRCCD', ccdExposure)
        else:
            ccdExposure = None

        ccdId = guessCcdId(dataIdList)

        if self.config.doCalibrate:
            if ccdExposure is None:
                ccdExposure = butler.get('postISRCCD', ccdId)
            calib = self.calibrate.run(ccdExposure)
            ccdExposure = calib.exposure
            if self.config.doWriteCalibrate:
                butler.put(ccdExposure, 'calexp', ccdId)
                butler.put(afwDet.PersistableSourceVector(calib.sources), 'icSrc', ccdId)
                if calib.psf is not None:
                    butler.put(calib.psf, 'psf', ccdId)
                if calib.apCorr is not None:
                    #butler.put(calib.apCorr, 'apcorr', ccdId)
                    pass
                if calib.matches is not None:
                    butler.put(afwDet.PersistableSourceMatchVector(calib.matches, calib.matchMeta),
                               'icMatch', ccdId)
        else:
            calib = None

        if self.config.doPhotometry:
            if ccdExposure is None:
                ccdExposure = butler.get('calexp', ccdId)
            if calib is None:
                psf = butler.get('psf', ccdId)
                apCorr = None # butler.get('apcorr', ccdId)
            else:
                psf = calib.psf
                apCorr = calib.apCorr
            phot = self.photometry.run(ccdExposure, psf, apcorr=apCorr)
            if self.config.doWritePhotometry:
                butler.put(afwDet.PersistableSourceVector(phot.sources), 'src', ccdId)
        else:
            phot = None

        return pipeBase.Struct(exposure=ccdExposure, psf=psf, apCorr=apCorr,
                               sources=phot.sources if phot else None,
                               matches=calib.matches if calib else None,
                               matchMeta=calib.matchMeta if calib else None,
                               )
