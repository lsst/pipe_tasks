# 
# LSST Data Management System
# Copyright 2008, 2009, 2010, 2011 LSST Corporation.
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
import numpy as num
import lsst.pex.config as pexConfig
import lsst.afw.detection as afwDet
import lsst.meas.utils.sourceDetection as muDetection
import lsst.pipe.base as pipeBase
from lsst.ip.diffim import SnapPsfMatchTask
from lsst.pipe.tasks.photometry import PhotometryDiffTask
from lsst.pipe.tasks.calibrate import CalibrateTask

class SnapCombineConfig(pexConfig.Config):
    doDiffim = pexConfig.Field(
        dtype = bool,
        doc = "Perform difference imaging before combining",
        default = True,
    )

    diffim = pexConfig.ConfigField(dtype = SnapPsfMatchTask.ConfigClass, doc = "")
    photometry = pexConfig.ConfigField(dtype = PhotometryDiffTask.ConfigClass,  doc = "")
    calibrate = pexConfig.ConfigField(dtype = CalibrateTask.ConfigClass,  doc = "")

class SnapCombineTask(pipeBase.Task):
    ConfigClass = SnapCombineConfig

    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)
        self.makeSubtask("diffim", SnapPsfMatchTask)
        self.makeSubtask("photometry", PhotometryDiffTask)
 
    @pipeBase.timeMethod
    def run(self, snap0, snap1):
        if self.config.doDiffim:
            diffRet = self.diffim.run(snap0, snap1, "subtractExposures")
            diffExp = diffRet.subtractedImage
            diffExp.writeFits("/tmp/diff.fits")

            fakePsf, wcs = self.makeFakePsf(snap0)
            photRet = self.photometry.run(diffExp, fakePsf, wcs=wcs)
            sources = photRet.sources
            footprints = photRet.footprintSets
        
        return pipeBase.Struct(visitExposure = snap0,
                               metadata = self.metadata)

    def makeFakePsf(self, exposure):
        """Initialise the detection procedure by setting the PSF and WCS

        @param exposure Exposure to process
        @return PSF, WCS
        """
        assert exposure, "No exposure provided"
        
        wcs = exposure.getWcs()
        assert wcs, "No wcs in exposure"

        model = self.config.calibrate.model
        fwhm = self.config.calibrate.fwhm / wcs.pixelScale().asArcseconds()
        size = self.config.calibrate.size
        self.log.log(self.log.INFO, "makeFakePsf fwhm=%s pixels; size=%s pixels" % (fwhm, size))
        psf = afwDet.createPsf(model, size, size, fwhm/(2.0*num.sqrt(2*num.log(2.0))))
        return psf, wcs
