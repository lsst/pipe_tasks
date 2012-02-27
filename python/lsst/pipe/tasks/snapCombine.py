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
import lsst.pex.config as pexConfig
import lsst.meas.utils.sourceDetection as muDetection
import lsst.pipe.base as pipeBase
from lsst.ip.diffim import SnapPsfMatchTask
from lsst.pipe.tasks.photometry import PhotometryTask

class SnapCombineConfig(pexConfig.Config):
    doDiffim = pexConfig.Field(
        dtype = bool,
        doc = "Perform difference imaging before combining",
        default = True,
    )

    diffim = pexConfig.ConfigField(dtype = SnapPsfMatchTask.ConfigClass, doc = "")
    photometry  = pexConfig.ConfigField(dtype = PhotometryTask.ConfigClass,  doc = "")

class SnapCombineTask(pipeBase.Task):
    ConfigClass = SnapCombineConfig

    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)
        self.makeSubtask("diffim", SnapPsfMatchTask)
        self.makeSubtask("photometry", PhotometryDiffTask)
 
   @pipeBase.timeMethod
    def run(self, snap1, snap2):
        fakePsf, wcs = self.makeFakePsf(snap1)
        
        if self.config.doDiffim:
            diffRet = self.diffim.run(diffim, fakePsf, "subtractExposures")
            diffExp = diffRet.subtractedImage
            
            photRet = self.detect.run(diffExp, fakePsf)
            sources = photRet.sources
            footprints = photRet.footprintSet

