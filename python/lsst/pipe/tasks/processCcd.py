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

class ProcessCcdConfig(pexConfig.Config):
    """Config for ProcessCcd"""
    doIsr = pexConfig.Field(dtype=bool, default=True, optional=False, doc = "Perform ISR?")
    doCalibrate = pexConfig.Field(dtype=bool, default=True, optional=False, doc = "Perform calibration?")
    doPhotometry = pexConfig.Field(dtype=bool, default=True, optional=False, doc = "Perform photometry?")
    isr = pexConfig.ConfigField(IsrTask.ConfigClass, doc="Instrumental Signature Removal", optional=False)
    calibrate = pexConfig.ConfigField(CalibrateTask.ConfigClass, doc="Calibration", optional=False)
    photometry = pexConfig.ConfigField(IsrTask.ConfigClass, doc="Photometry", optional=False)


class ProcessCcdTask(pipeBase.Task):
    """Process a CCD"""
    ConfigClass = ProcessCcdConfig

    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)
        self.makeSubtask("isr", IsrTask)
        self.makeSubtask("calibrate", CalibrateTask)
        self.makeSubtask("photometry", PhotometryTask)


    def run(self, butler, idList):
        assert butler and idList

        if self.config.doIsr:
            ccdExposure = self.isr.run(butler, idList)
            if self.config.doWriteIsr:
                butler.put('postISRCCD', ccdExposure)
        else:
            ccdExposure = None

        ### XXX How do I select the CCD identifier from the idList that contains the amps?
        ccdId = selectCcdId(idList)


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

        return Struct(
