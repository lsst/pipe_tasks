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
from __future__ import absolute_import, division, print_function
import lsst.pex.config as pexConfig
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.pipe.base as pipeBase
from lsst.ip.diffim import ModelPsfMatchTask
from lsst.meas.algorithms import WarpedPsf

__all__ = ["WarpAndPsfMatchTask"]


class WarpAndPsfMatchConfig(pexConfig.Config):
    """Config for WarpAndPsfMatchTask
    """
    psfMatch = pexConfig.ConfigurableField(
        target=ModelPsfMatchTask,
        doc="PSF matching model to model task",
    )
    warp = pexConfig.ConfigField(
        dtype=afwMath.Warper.ConfigClass,
        doc="warper configuration",
    )
    matchThenWarp = pexConfig.Field(
        dtype=bool,
        doc="Reverse order of warp and match operations to replicate legacy coadd temporary exposures",
        default=False,
    )


class WarpAndPsfMatchTask(pipeBase.Task):
    """A task to warp and PSF-match an exposure
    """
    ConfigClass = WarpAndPsfMatchConfig

    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)
        self.makeSubtask("psfMatch")
        self.warper = afwMath.Warper.fromConfig(self.config.warp)

    def run(self, exposure, wcs, modelPsf=None, maxBBox=None, destBBox=None):
        """Warp and PSF-match exposure (if modelPsf is not None)

        Parameters
        ----------
        exposure : :cpp:class: `lsst::afw::image::Exposure`
            Exposure to preprocess. PSF matching is done in place.
        wcs : :cpp:class:`lsst::afw::image::Wcs`
            Desired WCS of temporary images.
        modelPsf : :cpp:class: `lsst::meas::algorithms::KernelPsf` or None
            Target PSF to which to match.
            If None then exposures are not PSF matched.
        maxBBox : :cpp:class:`lsst::afw::geom::Box2I` or None
            Maximum allowed parent bbox of warped exposure.
            If None then the warped exposure will be just big enough to contain all warped pixels;
            if provided then the warped exposure may be smaller, and so missing some warped pixels;
            ignored if destBBox is not None.
        destBBox: :cpp:class: `lsst::afw::geom::Box2I` or None
            Exact parent bbox of warped exposure.
            If None then maxBBox is used to determine the bbox, otherwise maxBBox is ignored.


        Returns
        -------
        An lsst.pipe.base.Struct with the following fields:

        exposure : :cpp:class:`lsst::afw::image::Exposure`
            processed exposure
        """
        if self.config.matchThenWarp:
            # Legacy order of operations:
            # PSF-matching is performed before warping, which is incorrect.
            # a position-dependent warping (as is used in the general case) will
            # re-introduce a position-dependent PSF.
            if modelPsf is not None:
                exposure = self.psfMatch.run(exposure, modelPsf).psfMatchedExposure
            with self.timer("warp"):
                exposure = self.warper.warpExposure(wcs, exposure, maxBBox=maxBBox, destBBox=destBBox)
        else:
            if modelPsf is not None:
                # Warp PSF before overwriting exposure
                xyTransform = afwImage.XYTransformFromWcsPair(wcs, exposure.getWcs())
                psfWarped = WarpedPsf(exposure.getPsf(), xyTransform)

                if maxBBox is not None:
                    # grow warped region to provide sufficient area for PSF-matching
                    pixToGrow = 2 * max(self.config.psfMatch.kernel.active.sizeCellX,
                                        self.config.psfMatch.kernel.active.sizeCellY)
                    # replace with copy
                    maxBBox = afwGeom.Box2I(maxBBox)
                    maxBBox.grow(pixToGrow)

            with self.timer("warp"):
                exposure = self.warper.warpExposure(wcs, exposure, maxBBox=maxBBox, destBBox=destBBox)

            if modelPsf is not None:
                exposure.setPsf(psfWarped)
                exposure = self.psfMatch.run(exposure, modelPsf).psfMatchedExposure

        return pipeBase.Struct(
            exposure=exposure,
        )
