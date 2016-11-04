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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

import lsst.pex.config
import lsst.afw.table
import lsst.pipe.base


class MockSelectImagesTask(lsst.pipe.base.Task):
    """Simple select images task that just returns all the objects we simulated.
    """

    ConfigClass = lsst.pex.config.Config

    def runDataRef(self, dataRef, coordList, makeDataRefList=True, selectDataList=[]):
        observations = dataRef.butlerSubset.butler.get("observations", tract=dataRef.dataId["tract"])
        assert(makeDataRefList)  # this is all we make, so the user better want it
        butler = dataRef.butlerSubset.butler
        visitKey = observations.getSchema().find("visit").key
        ccdKey = observations.getSchema().find("ccd").key
        dataRefList = []
        for record in observations:
            dataId = {"visit": record.getI(visitKey), "ccd": record.getI(ccdKey)}
            dataRefList.append(butler.dataRef(datasetType="calexp", dataId=dataId))
        return lsst.pipe.base.Struct(dataRefList=dataRefList)
