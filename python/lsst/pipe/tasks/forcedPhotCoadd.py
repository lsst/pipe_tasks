#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2008-2013 LSST Corporation.
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

import lsst.afw.table
import lsst.pipe.base
from lsst.pex.config import Config, Field
from .coaddBase import ExistingCoaddDataIdContainer
from .forcedPhotImage import ForcedPhotImageTask

__all__ = ("ForcedPhotCoaddTask",)

class ForcedPhotCoaddConfig(ForcedPhotImageTask.ConfigClass):
    coaddName = Field(
        doc = "Coadd name on which to do forced measurements: typically one of deep or goodSeeing.",
        dtype = str,
        default = "deep",
    )

class ForcedPhotCoaddTask(ForcedPhotImageTask):
    """Run forced measurement on coadded images
    """

    ConfigClass = ForcedPhotCoaddConfig
    _DefaultName = "forcedPhotCoadd"

    def __init__(self, **kwargs):
        ForcedPhotImageTask.__init__(self, **kwargs)
        self.dataPrefix = self.config.coaddName + "Coadd_"

    def getExposure(self, dataRef):
        name = self.config.coaddName + "Coadd"
        return dataRef.get(name) if dataRef.datasetExists(name) else None

    def makeIdFactory(self, dataRef):
        expBits = dataRef.get(self.config.coaddName + "CoaddId_bits")
        expId = long(dataRef.get(self.config.coaddName + "CoaddId"))
        return lsst.afw.table.IdFactory.makeSource(expId, 64 - expBits)

    def fetchReferences(self, dataRef, exposure):
        skyMap = dataRef.get(self.dataPrefix + "skyMap", immediate=True)
        tractInfo = skyMap[dataRef.dataId["tract"]]
        patch = tuple(int(v) for v in dataRef.dataId["patch"].split(","))
        patchInfo = tractInfo.getPatchInfo(patch)
        return self.references.fetchInPatches(dataRef, patchList=[patchInfo])

    @classmethod
    def _makeArgumentParser(cls):
        parser = lsst.pipe.base.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "deepCoadd", help="data ID, e.g. --id tract=12345 patch=1,2",
                               ContainerClass=ExistingCoaddDataIdContainer)
        return parser

    def _getConfigName(self):
        """Return the name of the config dataset
        """
        return "%s_forcedPhotCoadd_config" % (self.config.coaddName,)
    
    def _getMetadataName(self):
        """Return the name of the metadata dataset
        """
        return "%s_forcedPhotCoadd_metadata" % (self.config.coaddName,)
