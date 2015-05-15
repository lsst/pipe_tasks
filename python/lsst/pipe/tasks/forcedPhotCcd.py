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
import lsst.afw.image
import lsst.pipe.base
from lsst.pex.config import Field
from .forcedPhotImage import ForcedPhotImageTask
from .dataIds import PerTractCcdDataIdContainer

try:
    from lsst.meas.mosaic import applyMosaicResults
except ImportError:
    applyMosaicResults = None

__all__ = ("ForcedPhotCcdTask",)


class ForcedPhotCcdConfig(ForcedPhotImageTask.ConfigClass):
    doApplyUberCal = Field(
        dtype = bool,
        doc = "Apply meas_mosaic ubercal results to input calexps?",
        default = True
    )

class ForcedPhotCcdTask(ForcedPhotImageTask):
    """Run forced measurement on CCD images
    """

    ConfigClass = ForcedPhotCcdConfig
    _DefaultName = "forcedPhotCcd"
    dataPrefix = ""  # Name to prepend to all input and output datasets (e.g. 'goodSeeingCoadd_')

    def makeIdFactory(self, dataRef):
        expBits = dataRef.get("ccdExposureId_bits")
        expId = long(dataRef.get("ccdExposureId"))
        return lsst.afw.table.IdFactory.makeSource(expId, 64 - expBits)        

    def fetchReferences(self, dataRef, exposure):
        references = lsst.afw.table.SourceCatalog(self.references.schema)
        badParents = set()
        unfiltered = self.references.fetchInBox(dataRef, exposure.getBBox(lsst.afw.image.PARENT), exposure.getWcs())
        for record in unfiltered:
            if record.getFootprint() is None or record.getFootprint().getArea() == 0:
                if record.getParent() != 0:
                    self.log.warn("Skipping reference %s (child of %s) with bad Footprint" %
                    (record.getId(), record.getParent()))
                else:
                    self.log.warn("Skipping reference parent %s with bad Footprint" % (record.getId(),))
                    badParents.add(record.getId())
            elif record.getParent() not in badParents:
                references.append(record)
        references.sort()   # need to ensure catalog is in ID order so find methods work
        return references

    def getExposure(self, dataRef):
        """Read input exposure to measure

        @param dataRef       Data reference from butler
        """
        if not dataRef.datasetExists(self.dataPrefix + "calexp"):
            return None
        exposure = dataRef.get("calexp", immediate=True)
        if not self.config.doApplyUberCal:
            return exposure
        if applyMosaicResults is None:
            raise RuntimeError(
                "Cannot use improved calibrations for %s because meas_mosaic could not be imported."
                % dataRef.dataId
                )
        else:
            try:
                applyMosaicResults(dataRef, calexp=exposure)
            except Exception as err:
                return None
        return exposure

    @classmethod
    def _makeArgumentParser(cls):
        parser = lsst.pipe.base.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "forced_src", help="data ID, with raw CCD keys + tract",
                               ContainerClass=PerTractCcdDataIdContainer)
        return parser
