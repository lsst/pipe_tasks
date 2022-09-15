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

__all__ = ["MakeDiscreteSkyMapConfig", "MakeDiscreteSkyMapTask"]

import lsst.sphgeom

import lsst.geom as geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.skymap import DiscreteSkyMap, BaseSkyMap
from lsst.utils.timer import timeMethod


class MakeDiscreteSkyMapConfig(pexConfig.Config):
    """Config for MakeDiscreteSkyMapTask.
    """

    coaddName = pexConfig.Field(
        doc="coadd name, e.g. deep, goodSeeing, chiSquared",
        dtype=str,
        default="deep",
    )
    skyMap = pexConfig.ConfigField(
        dtype=BaseSkyMap.ConfigClass,
        doc="SkyMap configuration parameters, excluding position and radius"
    )
    borderSize = pexConfig.Field(
        doc="additional border added to the bounding box of the calexps, in degrees",
        dtype=float,
        default=0.0
    )
    doAppend = pexConfig.Field(
        doc="append another tract to an existing DiscreteSkyMap on disk, if present?",
        dtype=bool,
        default=False
    )
    doWrite = pexConfig.Field(
        doc="persist the skyMap?",
        dtype=bool,
        default=True,
    )

    def setDefaults(self):
        self.skyMap.tractOverlap = 0.0


class MakeDiscreteSkyMapTask(pipeBase.Task):
    """Make a DiscreteSkyMap in a repository, using the bounding box of a set of calexps.

    The command-line and run signatures and config are sufficiently different from MakeSkyMapTask
    that we don't inherit from it, but it is a replacement, so we use the same config/metadata names.
    """

    ConfigClass = MakeDiscreteSkyMapConfig
    _DefaultName = "makeDiscreteSkyMap"

    @timeMethod
    def run(self, wcs_bbox_tuple_list, oldSkyMap=None):
        """Make a SkyMap from the bounds of the given set of calexp metadata.

        Parameters
        ----------
        wcs_bbox_tuple_list : `iterable`
           A list of tuples with each element expected to be a (Wcs, Box2I) pair.
        oldSkyMap : `lsst.skymap.DiscreteSkyMap`, optional
           The SkyMap to extend if appending.

        Returns
        -------
        skyMap : `lsst.pipe.base.Struct`
            Sky map returned as a struct with attributes:

            ``skyMap``
                The returned SkyMap (`lsst.skyMap.SkyMap`).
        """
        self.log.info("Extracting bounding boxes of %d images", len(wcs_bbox_tuple_list))
        points = []
        for wcs, boxI in wcs_bbox_tuple_list:
            boxD = geom.Box2D(boxI)
            points.extend(wcs.pixelToSky(corner).getVector() for corner in boxD.getCorners())
        if len(points) == 0:
            raise RuntimeError("No data found from which to compute convex hull")
        self.log.info("Computing spherical convex hull")
        polygon = lsst.sphgeom.ConvexPolygon.convexHull(points)
        if polygon is None:
            raise RuntimeError(
                "Failed to compute convex hull of the vertices of all calexp bounding boxes; "
                "they may not be hemispherical."
            )
        circle = polygon.getBoundingCircle()

        skyMapConfig = DiscreteSkyMap.ConfigClass()
        if oldSkyMap:
            skyMapConfig.raList.extend(oldSkyMap.config.raList)
            skyMapConfig.decList.extend(oldSkyMap.config.decList)
            skyMapConfig.radiusList.extend(oldSkyMap.config.radiusList)
        configIntersection = {k: getattr(self.config.skyMap, k)
                              for k in self.config.skyMap.toDict()
                              if k in skyMapConfig}
        skyMapConfig.update(**configIntersection)
        circleCenter = lsst.sphgeom.LonLat(circle.getCenter())
        skyMapConfig.raList.append(circleCenter[0].asDegrees())
        skyMapConfig.decList.append(circleCenter[1].asDegrees())
        circleRadiusDeg = circle.getOpeningAngle().asDegrees()
        skyMapConfig.radiusList.append(circleRadiusDeg + self.config.borderSize)
        skyMap = DiscreteSkyMap(skyMapConfig)

        for tractInfo in skyMap:
            wcs = tractInfo.getWcs()
            posBox = geom.Box2D(tractInfo.getBBox())
            pixelPosList = (
                posBox.getMin(),
                geom.Point2D(posBox.getMaxX(), posBox.getMinY()),
                posBox.getMax(),
                geom.Point2D(posBox.getMinX(), posBox.getMaxY()),
            )
            skyPosList = [wcs.pixelToSky(pos).getPosition(geom.degrees) for pos in pixelPosList]
            posStrList = ["(%0.3f, %0.3f)" % tuple(skyPos) for skyPos in skyPosList]
            self.log.info("tract %s has corners %s (RA, Dec deg) and %s x %s patches",
                          tractInfo.getId(), ", ".join(posStrList),
                          tractInfo.getNumPatches()[0], tractInfo.getNumPatches()[1])
        return pipeBase.Struct(
            skyMap=skyMap
        )
