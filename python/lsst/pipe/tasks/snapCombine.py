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

__all__ = ["SnapCombineConfig", "SnapCombineTask"]

import lsst.pex.config as pexConfig
import lsst.daf.base as dafBase
import lsst.afw.image as afwImage
import lsst.pipe.base as pipeBase
from lsst.coadd.utils import addToCoadd, setCoaddEdgeBits
from lsst.utils.timer import timeMethod


class SnapCombineConfig(pexConfig.Config):
    badMaskPlanes = pexConfig.ListField(
        dtype=str,
        doc="Mask planes that, if set, the associated pixels are not included in the combined exposure.",
        default=(),
    )


class SnapCombineTask(pipeBase.Task):
    """Combine two snaps into a single visit image.
    """

    ConfigClass = SnapCombineConfig
    _DefaultName = "snapCombine"

    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)

    @timeMethod
    def run(self, snap0, snap1):
        """Combine two snaps, returning the combined image.

        Parameters
        ----------
        snap0 : `lsst.afw.image.Exposure`
            Snapshot exposure 0.
        snap1 : `lsst.afw.image.Exposure`
            Snapshot exposure 1.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``exposure``
                Snap-combined exposure.
        """
        combined = self._add_snaps(snap0, snap1)

        return pipeBase.Struct(
            exposure=combined,
        )

    def _add_snaps(self, snap0, snap1):
        """Add two snap exposures together, returning a new exposure.

        Parameters
        ----------
        snap0 : `lsst.afw.image.Exposure`
            Snap exposure 0.
        snap1 : `lsst.afw.image.Exposure`
            Snap exposure 1.

        Returns
        -------
        combinedExp : `lsst.afw.image.Exposure`
            Combined exposure.
        """
        combinedExp = snap0.Factory(snap0, True)
        combinedMi = combinedExp.maskedImage
        combinedMi.set(0)

        weightMap = combinedMi.image.Factory(combinedMi.getBBox())
        weight = 1.0
        badPixelMask = afwImage.Mask.getPlaneBitMask(self.config.badMaskPlanes)
        addToCoadd(combinedMi, weightMap, snap0.maskedImage, badPixelMask, weight)
        addToCoadd(combinedMi, weightMap, snap1.maskedImage, badPixelMask, weight)

        # pre-scaling the weight map instead of post-scaling the combinedMi saves a bit of time
        # because the weight map is a simple Image instead of a MaskedImage
        weightMap *= 0.5  # so result is sum of both images, instead of average
        combinedMi /= weightMap
        setCoaddEdgeBits(combinedMi.getMask(), weightMap)

        # note: none of the inputs has a valid PhotoCalib object, so that is not touched
        # Filter was already copied

