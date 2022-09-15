# This file is part of pipe_tasks.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ["WarpAndPsfMatchTask"]

import lsst.pex.config as pexConfig
import lsst.afw.math as afwMath
import lsst.geom as geom
import lsst.afw.geom as afwGeom
import lsst.pipe.base as pipeBase
from lsst.ip.diffim import ModelPsfMatchTask
from lsst.meas.algorithms import WarpedPsf


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


class WarpAndPsfMatchTask(pipeBase.Task):
    """A task to warp and PSF-match an exposure
    """
    ConfigClass = WarpAndPsfMatchConfig

    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)
        self.makeSubtask("psfMatch")
        self.warper = afwMath.Warper.fromConfig(self.config.warp)

    def run(self, exposure, wcs, modelPsf=None, maxBBox=None, destBBox=None,
            makeDirect=True, makePsfMatched=False):
        """Warp and optionally PSF-match exposure

        Parameters
        ----------
        exposure : :cpp:class: `lsst::afw::image::Exposure`
            Exposure to preprocess.
        wcs : :cpp:class:`lsst::afw::image::Wcs`
            Desired WCS of temporary images.
        modelPsf : :cpp:class: `lsst::meas::algorithms::KernelPsf` or None
            Target PSF to which to match.
        maxBBox : :cpp:class:`lsst::afw::geom::Box2I` or None
            Maximum allowed parent bbox of warped exposure.
            If None then the warped exposure will be just big enough to contain all warped pixels;
            if provided then the warped exposure may be smaller, and so missing some warped pixels;
            ignored if destBBox is not None.
        destBBox: :cpp:class: `lsst::afw::geom::Box2I` or None
            Exact parent bbox of warped exposure.
            If None then maxBBox is used to determine the bbox, otherwise maxBBox is ignored.
        makeDirect : bool
            Return an exposure that has been only warped?
        makePsfMatched : bool
            Return an exposure that has been warped and PSF-matched?

        Returns
        -------
        An lsst.pipe.base.Struct with the following fields:

        direct : :cpp:class:`lsst::afw::image::Exposure`
            warped exposure
        psfMatched : :cpp:class: `lsst::afw::image::Exposure`
            warped and psf-Matched temporary exposure
        """
        if makePsfMatched and modelPsf is None:
            raise RuntimeError("makePsfMatched=True, but no model PSF was provided")

        if not makePsfMatched and not makeDirect:
            self.log.warning("Neither makeDirect nor makePsfMatched requested")

        # Warp PSF before overwriting exposure
        xyTransform = afwGeom.makeWcsPairTransform(exposure.getWcs(), wcs)
        psfWarped = WarpedPsf(exposure.getPsf(), xyTransform)

        if makePsfMatched and maxBBox is not None:
            # grow warped region to provide sufficient area for PSF-matching
            pixToGrow = 2 * max(self.psfMatch.kConfig.sizeCellX,
                                self.psfMatch.kConfig.sizeCellY)
            # replace with copy
            maxBBox = geom.Box2I(maxBBox)
            maxBBox.grow(pixToGrow)

        with self.timer("warp"):
            exposure = self.warper.warpExposure(wcs, exposure, maxBBox=maxBBox, destBBox=destBBox)
            exposure.setPsf(psfWarped)

        if makePsfMatched:
            try:
                exposurePsfMatched = self.psfMatch.run(exposure, modelPsf).psfMatchedExposure
            except Exception as e:
                exposurePsfMatched = None
                self.log.info("Cannot PSF-Match: %s", e)

        return pipeBase.Struct(
            direct=exposure if makeDirect else None,
            psfMatched=exposurePsfMatched if makePsfMatched else None
        )
