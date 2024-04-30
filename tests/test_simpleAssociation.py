# This file is part of pipe_tasks.

# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np
import pandas as pd
import unittest

import lsst.afw.table as afwTable
import lsst.geom as geom
import lsst.utils.tests
from lsst.pipe.tasks.associationUtils import toIndex
from lsst.pipe.tasks.simpleAssociation import SimpleAssociationTask


class TestSimpleAssociation(lsst.utils.tests.TestCase):

    def setUp(self):
        simpleAssoc = SimpleAssociationTask()

        self.nDiaObjects = 10
        self.diaObjRas = np.linspace(45, 46, self.nDiaObjects)
        self.diaObjDecs = np.linspace(45, 46, self.nDiaObjects)
        # Copy a coord to get multiple matches.
        self.diaObjRas[3] = self.diaObjRas[2] + 0.1/3600
        self.diaObjDecs[3] = self.diaObjDecs[2] + 0.1/3600
        self.diaObjects = [
            simpleAssoc.createDiaObject(objId, ra, dec)
            for objId, ra, dec in zip(
                np.arange(self.nDiaObjects, dtype=int),
                self.diaObjRas,
                self.diaObjDecs)]

        self.hpIndices = [toIndex(simpleAssoc.config.nside,
                                  diaObj["ra"],
                                  diaObj["dec"])
                          for diaObj in self.diaObjects]

        self.newDiaObjectVisit = 1236
        # Drop in two copies of the DiaObject locations to make DiaSources.
        diaSourceList = [
            {"visit": 1234,
             "detector": 42,
             "diaSourceId": idx,
             "diaObjectId": 0,
             "ra": ra,
             "dec": dec}
            for idx, (ra, dec) in enumerate(zip(self.diaObjRas,
                                                self.diaObjDecs))]
        self.coordList = [
            [geom.SpherePoint(diaSrc["ra"], diaSrc["dec"], geom.degrees)]
            for diaSrc in diaSourceList]
        moreDiaSources = [
            {"visit": 1235,
             "detector": 43,
             "diaSourceId": idx + self.nDiaObjects,
             "diaObjectId": 0,
             "ra": ra,
             "dec": dec}
            for idx, (ra, dec) in enumerate(zip(self.diaObjRas,
                                                self.diaObjDecs))]
        for idx in range(self.nDiaObjects):
            self.coordList[idx].append(
                geom.SpherePoint(moreDiaSources[idx]["ra"],
                                 moreDiaSources[idx]["dec"],
                                 geom.degrees))
        diaSourceList.extend(moreDiaSources)

        self.nNewDiaSources = 2
        # Drop in two more DiaSources that are unassociated.
        diaSourceList.append({"visit": 1236,
                              "detector": 44,
                              "diaSourceId": len(diaSourceList),
                              "diaObjectId": 0,
                              "ra": 0.0,
                              "dec": 0.0})
        diaSourceList.append({"visit": 1236,
                              "detector": 45,
                              "diaSourceId": len(diaSourceList),
                              "diaObjectId": 0,
                              "ra": 1.0,
                              "dec": 89.0})
        self.diaSources = pd.DataFrame(data=diaSourceList)

    def tearDown(self):
        del self.diaObjects
        del self.hpIndices
        del self.diaSources
        del self.coordList

    def testRun(self):
        """Test the full run method of the simple associator.
        """
        simpleAssoc = SimpleAssociationTask()
        result = simpleAssoc.run(self.diaSources)

        # Test the number of expected DiaObjects are created.
        self.assertEqual(len(result.diaObjects),
                         self.nDiaObjects + self.nNewDiaSources)

        # Test that DiaSources are assigned the correct ``diaObjectId``
        assocDiaObjects = result.diaObjects
        assocDiaSources = result.assocDiaSources.reset_index().set_index(["diaObjectId", "diaSourceId"])
        for idx, (diaObjId, diaObj) in enumerate(assocDiaObjects.iterrows()):
            if idx < 10:
                self.assertEqual(len(assocDiaSources.loc[diaObjId]), 2)
            else:
                self.assertEqual(len(assocDiaSources.loc[diaObjId]), 1)

    def testUpdateCatalogs(self):
        """Test adding data to existing DiaObject/Source catalogs.
        """
        matchIndex = 4
        diaSrc = self.diaSources.iloc[matchIndex]
        self.diaObjects[matchIndex]["diaObjectId"] = 1234
        visit = diaSrc["visit"]
        detector = diaSrc["detector"]
        diaSourceId = diaSrc["diaSourceId"]
        self.diaSources[("visit,detector")] = list(zip(self.diaSources["visit"],
                                                       self.diaSources["detector"]))
        self.diaSources.set_index(["visit,detector", "diaSourceId"], inplace=True)

        simpleAssoc = SimpleAssociationTask()
        simpleAssoc.updateCatalogs(matchIndex,
                                   diaSrc,
                                   self.diaSources,
                                   visit,
                                   detector,
                                   diaSourceId,
                                   self.diaObjects,
                                   self.coordList,
                                   self.hpIndices)
        self.assertEqual(len(self.hpIndices), self.nDiaObjects)
        self.assertEqual(len(self.coordList), self.nDiaObjects)
        # Should be 3 source coordinates.
        self.assertEqual(len(self.coordList[matchIndex]), 3)
        self.assertEqual(len(self.diaObjects), self.nDiaObjects)
        self.assertEqual(self.diaSources.loc[((visit, detector), diaSourceId), "diaObjectId"].iloc[0],
                         self.diaObjects[matchIndex]["diaObjectId"])

    def testAddDiaObject(self):
        """Test adding data to existing DiaObjects/Sources.
        """
        diaSrc = self.diaSources.iloc[-1]
        visit = diaSrc["visit"]
        detector = diaSrc["detector"]
        diaSourceId = diaSrc["diaSourceId"]
        self.diaSources[("visit,detector")] = list(zip(self.diaSources["visit"], self.diaSources["detector"]))
        self.diaSources.set_index(["visit,detector", "diaSourceId"], inplace=True)
        idCat = afwTable.SourceCatalog(
            afwTable.SourceTable.make(afwTable.SourceTable.makeMinimalSchema()))

        simpleAssoc = SimpleAssociationTask()
        simpleAssoc.addNewDiaObject(diaSrc,
                                    self.diaSources,
                                    visit,
                                    detector,
                                    diaSourceId,
                                    self.diaObjects,
                                    idCat,
                                    self.coordList,
                                    self.hpIndices)
        self.assertEqual(len(self.hpIndices), self.nDiaObjects + 1)
        self.assertEqual(len(self.coordList), self.nDiaObjects + 1)
        self.assertEqual(len(self.diaObjects), self.nDiaObjects + 1)
        self.assertEqual(self.diaSources.loc[((visit, detector), diaSourceId), "diaObjectId"].iloc[0],
                         idCat[0].get("id"))

    def testFindMatches(self):
        """Test the simple brute force matching algorithm.
        """
        simpleAssoc = SimpleAssociationTask()
        # No match
        matchResult = simpleAssoc.findMatches(
            0.0,
            0.0,
            2*simpleAssoc.config.tolerance,
            self.hpIndices,
            self.diaObjects)
        self.assertIsNone(matchResult.dists)
        self.assertIsNone(matchResult.matches)

        # One match
        matchResult = simpleAssoc.findMatches(
            self.diaObjRas[4],
            self.diaObjDecs[4],
            2*simpleAssoc.config.tolerance,
            self.hpIndices,
            self.diaObjects)
        self.assertEqual(len(matchResult.dists), 1)
        self.assertEqual(len(matchResult.matches), 1)
        self.assertEqual(matchResult.matches[0], 4)

        # 2 match
        matchResult = simpleAssoc.findMatches(
            self.diaObjRas[2],
            self.diaObjDecs[2],
            2*simpleAssoc.config.tolerance,
            self.hpIndices,
            self.diaObjects)
        self.assertEqual(len(matchResult.dists), 2)
        self.assertEqual(len(matchResult.matches), 2)
        self.assertEqual(matchResult.matches[0], 2)
        self.assertEqual(matchResult.matches[1], 3)


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
