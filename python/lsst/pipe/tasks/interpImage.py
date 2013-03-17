#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2008, 2009, 2010, 2011, 2012 LSST Corporation.
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
import math

import lsst.pex.config as pexConfig
import lsst.afw.detection as afwDetection
import lsst.afw.geom as afwGeom
import lsst.meas.algorithms as measAlg
import lsst.pipe.base as pipeBase
import lsst.ip.isr as ipIsr

__all__ = ["InterpImageConfig", "InterpMixinTask"]

FwhmPerSigma = 2 * math.sqrt(2 * math.log(2))

class InterpImageConfig(pexConfig.Config):
    """Config for InterpImageTask
    """
    interpKernelSizeFactor = pexConfig.Field(
        dtype = float,
        doc = "Interpolation kernel size = interpFwhm converted to pixels * interpKernelSizeFactor.",
        default = 3.0,
    )

class InterpImageTask(pipeBase.Task):
    """Interpolate over bad image pixels
    """
    ConfigClass = InterpImageConfig

    @pipeBase.timeMethod
    def interpolateOnePlane(self, maskedImage, planeName, fwhmPixels):
        """Interpolate over one mask plane, in place

        @param[in,out] maskedImage: MaskedImage over which to interpolate over edge pixels
        @param[in] fwhmPixels: FWHM of double Gaussian model to use for interpolation (pixels)
        @param[in] planeName: mask plane over which to interpolate
        @param[in] PSF to use to detect NaNs
        """
        self.log.info("Interpolate over %s pixels" % (planeName,))
        kernelSize = int(round(fwhmPixels * self.config.interpKernelSizeFactor))
        kernelDim = afwGeom.Point2I(kernelSize, kernelSize)
        coreSigma = fwhmPixels / FwhmPerSigma
        psfModel = measAlg.DoubleGaussianPsf(kernelDim[0], kernelDim[1], coreSigma, coreSigma * 2.5, 0.1)

        nanDefectList = ipIsr.getDefectListFromMask(maskedImage, planeName, growFootprints=0)
        measAlg.interpolateOverDefects(maskedImage, psfModel, nanDefectList, 0.0)
