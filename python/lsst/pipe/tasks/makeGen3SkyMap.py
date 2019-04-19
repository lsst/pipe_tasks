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
import lsst.afw.geom as afwGeom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.daf.butler import Butler, DatasetType
from lsst.skymap import skyMapRegistry


class MakeGen3SkyMapConfig(pexConfig.Config):
    """Config for MakeGen3SkyMapTask
    """
    datasetTypeName = pexConfig.Field(
        doc="Name assigned to created skymap in butler registry",
        dtype=str,
        default="deepCoadd_skyMap",
    )
    name = pexConfig.Field(
        doc="Name assigned to created skymap in butler registry",
        dtype=str,
        default=None,
        optional=True
    )
    skyMap = skyMapRegistry.makeField(
        doc="type of skyMap",
        default="dodeca",
    )
    doWrite = pexConfig.Field(
        doc="persist the skyMap? If False then run generates the sky map and returns it, "
            "but does not save it to the data repository",
        dtype=bool,
        default=True,
    )

    def validate(self):
        if self.name is None:
            raise ValueError("The name field must be set to the name of the specific "
                             "skymap to use when writing to the butler")


class MakeGen3SkyMapTask(pipeBase.Task):
    ConfigClass = MakeGen3SkyMapConfig
    _DefaultName = "makeGen3SkyMap"

    """This is a task to construct and optionally save a SkyMap into a gen3
    butler repository.

    Parameters
    ----------
    config : `MakeGen3SkyMapConfig` or None
        Instance of a configuration class specifying task options, a default
        config is created if value is None
    butler : `str`
        Path to a gen3 butler
    collection : `str`
        The name of the collection which the created skymap will be inserted
    """

    def __init__(self, *, config=None, butler=None, collection=None, **kwargs):
        if butler is None:
            raise ValueError("A path to a butler must be given")
        if collection is None:
            raise ValueError("The collection to use with this butler must be provided")
        super().__init__(config=config, **kwargs)
        self.butler = Butler(butler, run=collection)
        self.collection = collection

    def run(self):
        """Construct and optionally save a SkyMap into a gen3 repository"""
        skyMap = self.config.skyMap.apply()
        self.logSkyMapInfo(skyMap)
        skyMapHash = skyMap.getSha1()
        if self.config.doWrite:
            try:
                existing, = self.butler.registry.query("SELECT skymap FROM skymap WHERE hash=:hash",
                                                       hash=skyMapHash)
                raise RuntimeError(
                    (f"SkyMap with name {existing.name} and hash {skyMapHash} already exist in "
                     f"the butler collection {self.collection}, SkyMaps must be unique within "
                     "a collection")
                )
            except ValueError:
                self.log.info(f"Inserting SkyMap {self.config.name} with hash={skyMapHash}")
                with self.butler.registry.transaction():
                    skyMap.register(self.config.name, self.butler.registry)
                    self.butler.registry.registerDatasetType(DatasetType(name=self.config.datasetTypeName,
                                                                         dimensions=["skymap"],
                                                                         storageClass="SkyMap"))
                    self.butler.put(skyMap, self.config.datasetTypeName, {"skymap": self.config.name})

        return pipeBase.Struct(
            skyMap=skyMap
        )

    def logSkyMapInfo(self, skyMap):
        """!Log information about a sky map
        @param[in] skyMap  sky map (an lsst.skyMap.SkyMap)
        """
        self.log.info("sky map has %s tracts" % (len(skyMap),))
        for tractInfo in skyMap:
            wcs = tractInfo.getWcs()
            posBox = afwGeom.Box2D(tractInfo.getBBox())
            pixelPosList = (
                posBox.getMin(),
                afwGeom.Point2D(posBox.getMaxX(), posBox.getMinY()),
                posBox.getMax(),
                afwGeom.Point2D(posBox.getMinX(), posBox.getMaxY()),
            )
            skyPosList = [wcs.pixelToSky(pos).getPosition(afwGeom.degrees) for pos in pixelPosList]
            posStrList = ["(%0.3f, %0.3f)" % tuple(skyPos) for skyPos in skyPosList]
            self.log.info("tract %s has corners %s (RA, Dec deg) and %s x %s patches" %
                          (tractInfo.getId(), ", ".join(posStrList),
                           tractInfo.getNumPatches()[0], tractInfo.getNumPatches()[1]))
