#!/usr/bin/env python

from lsst.pex.config import Field
from lsst.ip.isr.isrTask import IsrTaskConfig, IsrTask
from lsst.pipe.base import Struct

class HscIsrConfig(IsrTaskConfig):
    doSaturation = Field(doc="Mask saturated pixels?", dtype=bool, default=True)
    doOverscan = Field(doc="Do overscan subtraction?", dtype=bool, default=True)
    doBias = Field(doc="Do bias subtraction?", dtype=bool, default=False)
    doVariance = Field(doc="Calculate variance?", dtype=bool, default=True)
    doDark = Field(doc="Do dark subtraction?", dtype=bool, default=False)
    doFlat = Field(doc="Do flat-fielding?", dtype=bool, default=True)

class HscIsrTask(IsrTask):
    ConfigClass = HscIsrConfig
    def run(self, exposure, calibSet):
        exposure = self.doConversionForIsr(exposure, calibSet)
        if self.config.doSaturation:
            exposure = self.doSaturationDetection(exposure, calibSet)
        if self.config.doOverscan:
            exposure = self.doOverscanCorrection(exposure, calibSet)

        if self.config.doVariance:
            # Ideally, this should be done after bias subtraction, but CCD assembly demands a variance plane
            exposure = self.doVariance(exposure, calibSet)

        exposure = self.doCcdAssembly([exposure])

        if self.config.doBias:
            exposure = self.doBiasSubtraction(exposure, calibSet)
        if self.config.doDark:
            exposure = self.doDarkCorrection(exposure, calibSet)
        if self.config.doFlat:
            exposure = self.doFlatCorrection(exposure, calibSet)

        return Struct(postIsrExposure=exposure)

    def makeCalibDict(self, butler, dataId):
        ret = {}
        required = {"doBias": "bias",
                    "doDark": "dark",
                    "doFlat": "flat",
                    }
        for method in required.keys():
            if getattr(self.config, method):
                calib = required[method]
                ret[calib] = butler.get(calib, dataId)
        return ret

