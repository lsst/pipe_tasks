#
# LSST Data Management System
# Copyright 2008-2018 AURA/LSST.
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
from collections import defaultdict

from lsst.coadd.utils import CoaddDataIdContainer

class TractQADataIdContainer(CoaddDataIdContainer):

    def makeDataRefList(self, namespace):
        """Make self.refList from self.idList

        Generates a list of data references given tract and/or patch.
        This is a copy of `TractDataIdContainer`, except that it doesn't
        require "filter", since the QA data products it is designed to serve
        are merged over filters.
        """
        def getPatchRefList(tract):
            return [namespace.butler.dataRef(datasetType=self.datasetType,
                                             tract=tract.getId(),
                                             patch="%d,%d" % patch.getIndex()) for patch in tract]

        tractRefs = defaultdict(list)  # Data references for each tract
        for dataId in self.idList:

            skymap = self.getSkymap(namespace)

            if "tract" in dataId:
                tractId = dataId["tract"]
                if "patch" in dataId:
                    tractRefs[tractId].append(namespace.butler.dataRef(datasetType=self.datasetType,
                                                                       tract=tractId,
                                                                       patch=dataId['patch']))
                else:
                    tractRefs[tractId] += getPatchRefList(skymap[tractId])
            else:
                tractRefs = dict((tract.getId(), tractRefs.get(tract.getId(), []) + getPatchRefList(tract))
                                 for tract in skymap)

        self.refList = list(tractRefs.values())
