#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
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
import lsst.afw.coord as afwCoord
import lsst.afw.geom as afwGeom
import lsst.skymap
import lsst.pipe.base as pipeBase

class CoaddArgumentParser(pipeBase.ArgumentParser):
    """A version of lsst.pipe.base.ArgumentParser specialized for coaddition.
    
    Required because butler.subset does not support patch and tract
    """
    def _makeDataRefList(self, namespace):
        """Make namespace.dataRefList from namespace.dataIdList
        """
        datasetType = namespace.config.coaddName + "Coadd"
        validKeys = namespace.butler.getKeys(datasetType=datasetType, level=self._dataRefLevel)

        namespace.dataRefList = []
        for dataId in namespace.dataIdList:
            # tract and patch are required
            for key in validKeys:
                if key not in dataId:
                    self.error("--id must include " + key)
            dataRef = namespace.butler.dataRef(
                datasetType = datasetType,
                dataId = dataId,
            )
            namespace.dataRefList.append(dataRef)
