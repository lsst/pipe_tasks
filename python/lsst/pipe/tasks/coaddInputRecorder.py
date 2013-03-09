#!/usr/bin/env python
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
import math
import numpy

import lsst.pex.config as pexConfig
import lsst.afw.table as afwTable
import lsst.afw.image as afwImage
import lsst.pipe.base as pipeBase

__all__ = ["CoaddInputRecorderTask"]

class CoaddInputRecorderConfig(pexConfig.Config):
    """Config for CoaddInputRecorderTask

    The inputRecorder section of the various coadd tasks' configs should generally agree,
    or the schemas created by earlier tasks (like MakeCoaddTempExpTask) will not contain
    the fields filled by later tasks (like AssembleCoaddTask).
    """
    saveEmptyCcds = pexConfig.Field(
        dtype=bool, default=False, optional=False,
        doc=("Add records for CCDs we iterated over but did not add a coaddTempExp"
             " due to a lack of unmasked pixels in the coadd footprint.")
    )
    saveErrorCcds = pexConfig.Field(
        dtype=bool, default=False, optional=False,
        doc=("Add records for CCDs we iterated over but did not add a coaddTempExp"
             " due to an exception (often due to the calexp not being found on disk).")
    )
    saveVisitGoodPix = pexConfig.Field(
        dtype=bool, default=False, optional=False,
        doc=("Save the total number of good pixels in each coaddTempExp (redundant with a sum of"
             " good pixels in associated CCDs)")
    )
    saveCcdWeights = pexConfig.Field(
        dtype=bool, default=True, optional=False,
        doc=("Save weights in the CCDs table as well as the visits table?"
             " (This is necessary for easy construction of CoaddPsf, but otherwise duplicate information.)")
    )

class CoaddInputRecorderTask(pipeBase.Task):
    """Subtask that handles filling a CoaddInputs object for a coadd exposure, tracking the CCDs and
    visits that went into a coadd.

    The interface here is a little messy, but I think this is at least partly a product of a bit of
    messiness in the coadd code it's plugged into.  I hope #2590 might result in a better design.
    """

    ConfigClass = CoaddInputRecorderConfig

    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)        
        self.visitSchema = afwTable.ExposureTable.makeMinimalSchema()
        if self.config.saveVisitGoodPix:
            self.visitGoodPixKey = self.visitSchema.addField("goodpix", type=int,
                                                             doc="Number of good pixels in the coaddTempExp")
        self.visitWeightKey = self.visitSchema.addField("weight", type=float,
                                                        doc="Weight for this visit in the coadd")
        self.ccdSchema = afwTable.ExposureTable.makeMinimalSchema()
        self.ccdCcdKey = self.ccdSchema.addField("ccd", type=int, doc="cameraGeom CCD serial number")
        self.ccdVisitKey = self.ccdSchema.addField("visit", type=numpy.int64,
                                                   doc="Foreign key for the visits (coaddTempExp) catalog")
        self.ccdGoodPixKey = self.ccdSchema.addField("goodpix", type=int,
                                                     doc="Number of good pixels in this CCD")
        if self.config.saveCcdWeights:
            self.ccdWeightKey = self.ccdSchema.addField("weight", type=float,
                                                        doc="Weight for this visit in the coadd")

    def makeCoaddInputs(self):
        """Create a CoaddInputs object using the schemas defined by the config parameters.
        """
        return afwImage.CoaddInputs(self.visitSchema, self.ccdSchema)

    def makeCoaddTempExp(self, coaddInputs, visit, nGoodPix, coaddTempExp):
        """Called once for each coaddTempExp by MakeCoaddTempExpTask, to add a single record
        to the visits table and attach the CoaddInputs object to the exposure.  Returns
        the record, so subclasses can call the base class method and fill additional fields.
        """
        coaddTempExp.getInfo().setCoaddInputs(coaddInputs)
        record = coaddInputs.visits.addNew()
        record.setId(visit)
        if self.config.saveVisitGoodPix:
            record.setI(self.visitGoodPixKey, nGoodPix)
        record.setPsf(coaddTempExp.getPsf())
        record.setWcs(coaddTempExp.getWcs())
        record.setBBox(coaddTempExp.getBBox(afwImage.PARENT))
        return record

    def makeCcdRecord(self, coaddInputs, visit, calexp, dataRef):
        """Called once for each CCD by MakeCoaddTempExpTask, to create a record for the ccds catalog.
        The record is *not* immediately added to the catalog, reflecting the fact that we may
        not want to add it later, depending on the inputRecorder configuration and how many good
        pixels are found (see saveCcdRecord).
        """
        record = coaddInputs.ccds.getTable().makeRecord()
        record.setId(dataRef.get("ccdExposureId", immediate=True))
        record.setL(self.ccdVisitKey, visit)
        try:
            record.setI(self.ccdCcdKey, calexp.getDetector().getId().getSerial())
        except:
            self.log.warn("Error getting detector serial number in visit %d; using -1" % visit)
            record.setI(self.ccdCcdKey, -1)
        if calexp is not None:
            record.setPsf(calexp.getPsf())
            record.setWcs(calexp.getWcs())
            record.setBBox(calexp.getBBox(afwImage.PARENT))
        return record

    def saveCcdRecord(self, coaddInputs, record, nGoodPix):
        """Save a CCD inputRecorder record previously created by makeCcdRecord.

        If the nGoodPix is zero, the record will only be saved if config.saveEmptyCcds is True.
        """
        if nGoodPix != 0 or self.config.saveEmptyCcds:
            record.setI(self.ccdGoodPixKey, nGoodPix)
            coaddInputs.ccds.append(record)

    def addVisitToCoadd(self, coaddInputs, coaddTempExp, weight):
        """Called by AssembleCoaddTask when adding (a subset of) a coaddTempExp to a coadd.  The
        base class impementation extracts the CoaddInputs from the coaddTempExp and appends
        them to the given coaddInputs, filling in the weight column(s).

        Note that the passed coaddTempExp may be a subimage, but that this method will only be
        called for the first subimage

        Returns the record for the visit to allow subclasses to fill in additional fields.
        Warns and returns None if the inputRecorder catalogs for the coaddTempExp are not usable.
        """
        tempExpInputs = coaddTempExp.getInfo().getCoaddInputs()
        if len(tempExpInputs.visits) != 1:
            self.log.warn("CoaddInputs for coaddTempExp should have exactly one record in visits table "
                          "(found %d).  CoaddInputs for this visit will not be saved."
                          % len(tempExpInputs.visits))
            return None
        inputVisitRecord = tempExpInputs.visits[0];
        outputVisitRecord = coaddInputs.visits.addNew()
        outputVisitRecord.assign(inputVisitRecord)
        outputVisitRecord.setD(self.visitWeightKey, weight)
        for inputCcdRecord in tempExpInputs.ccds:
            if inputCcdRecord.getL(self.ccdVisitKey) != inputVisitRecord.getId():
                self.log.warn("CoaddInputs for coaddTempExp with id %d contains CCDs with visit=%d. "
                              "CoaddInputs may be unreliable."
                              % (inputVisitRecord.getId(), inputCcdRecord.getL(self.ccdVisitKey)))
            outputCcdRecord = coaddInputs.ccds.addNew()
            outputCcdRecord.assign(inputCcdRecord)
            if self.config.saveCcdWeights:
                outputCcdRecord.setD(self.ccdWeightKey, weight)
        return inputVisitRecord
