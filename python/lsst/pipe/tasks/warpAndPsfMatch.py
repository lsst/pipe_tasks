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
import lsst.pex.config as pexConfig
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
import lsst.pipe.base as pipeBase
from lsst.ip.diffim import ModelPsfMatchTask

__all__ = ["WarpAndPsfMatchTask"]

class WarpAndPsfMatchConfig(pexConfig.Config):
    """Config for WarpAndPsfMatchTask
    """
    psfMatch = pexConfig.ConfigurableField(
        target = ModelPsfMatchTask,
        doc = "PSF matching model to model task",
    )
    warp = pexConfig.ConfigField(
        dtype = afwMath.Warper.ConfigClass,
        doc = "warper configuration",
    )


class WarpAndPsfMatchTask(pipeBase.Task):
    """A task to warp and PSF-match an exposure
    """
    ConfigClass = WarpAndPsfMatchConfig

    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)
        self.makeSubtask("psfMatch")
        self.warper = afwMath.Warper.fromConfig(self.config.warp)

    def run(self, exposure, wcs, modelPsf=None, maxBBox=None, destBBox=None, multX=None, multY=None):
        """PSF-match exposure (if modelPsf is not None) and warp

        Note that PSF-matching is performed before warping, which is incorrect:
        a position-dependent warping (as is used in the general case) will
        re-introduce a position-dependent PSF.  However, this is easier, and
        sufficient for now (until we are able to warp PSFs to determine the
        correct target PSF).

        @param[in,out] exposure: exposure to preprocess; PSF matching is done in place
        @param[in] wcs: desired WCS of temporary images
        @param[in] modelPsf: target PSF to which to match (or None)
        @param maxBBox: maximum allowed parent bbox of warped exposure (an afwGeom.Box2I or None);
            if None then the warped exposure will be just big enough to contain all warped pixels;
            if provided then the warped exposure may be smaller, and so missing some warped pixels;
            ignored if destBBox is not None
        @param destBBox: exact parent bbox of warped exposure (an afwGeom.Box2I or None);
            if None then maxBBox is used to determine the bbox, otherwise maxBBox is ignored

        @return a pipe_base Struct containing:
        - exposure: processed exposure
        """
        if modelPsf is not None:
            exposure = self.psfMatch.run(exposure, modelPsf).psfMatchedExposure
        with self.timer("warp"):
            exposure, covImage = self.warper.warpExposure(wcs, exposure, maxBBox=maxBBox, destBBox=destBBox,
                                                          multX=multX, multY=multY)
        return pipeBase.Struct(
            exposure=exposure,
            covImage=covImage,
        )
