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
    bad_mask_planes = pexConfig.ListField(
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
        self.log.info("Merging two snaps with exposure ids: %s, %s", snap0.visitInfo.id, snap1.visitInfo.id)
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
        combined : `lsst.afw.image.Exposure`
            Combined exposure.
        """
        combined = snap0.Factory(snap0, True)
        combined.maskedImage.set(0)

        weights = combined.maskedImage.image.Factory(combined.maskedImage.getBBox())
        weight = 1.0
        bad_mask = afwImage.Mask.getPlaneBitMask(self.config.bad_mask_planes)
        addToCoadd(combined.maskedImage, weights, snap0.maskedImage, bad_mask, weight)
        addToCoadd(combined.maskedImage, weights, snap1.maskedImage, bad_mask, weight)

        # pre-scaling the weight map instead of post-scaling the combined.maskedImage saves a bit of time
        # because the weight map is a simple Image instead of a MaskedImage
        weights *= 0.5  # so result is sum of both images, instead of average
        combined.maskedImage /= weights
        setCoaddEdgeBits(combined.maskedImage.getMask(), weights)

        combined.info.setVisitInfo(self._merge_visit_info(snap0.visitInfo, snap1.visitInfo))

        return combined

    def _merge_visit_info(self, info0, info1):
        """Merge the visitInfo values from the two exposures.

        In particular:
         * id will be the id of snap 0.
         * date will be the average of the dates.
         * exposure time will be the sum of the times.

        Parameters
        ----------
        info0, info1 : `lsst.afw.image.VisitInfo`
            Metadata to combine.

        Returns
        -------
        info : `lsst.afw.image.VisitInfo`
            Combined metadata.

        """
        time = info0.exposureTime + info1.exposureTime
        date = (info0.date.get() + info1.date.get()) / 2.0
        result = info0.copyWith(exposureTime=time,
                                date=dafBase.DateTime(date)
                                )
        return result
