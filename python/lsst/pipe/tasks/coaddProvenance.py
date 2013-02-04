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

__all__ = ["CoaddProvenanceTask"]

class CoaddProvenanceConfig(pexConfig.Config):
    """Config for CoaddProvenanceTask

    The provenance section of the various coadd tasks' configs should generally agree,
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

class CoaddProvenanceTask(pipeBase.Task):
    """Subtask that handles provenance for coadds, filling the CoaddInputs object in the final Exposure."""
    ConfigClass = CoaddProvenanceConfig

    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)        
        self.visitSchema = afwTable.ExposureTable.makeMinimalSchema()
        if self.config.saveVisitGoodPix:
            self.visitGoodPixKey = self.visitSchema.addField("goodpix", type=int,
                                                             doc="Number of good pixels in the coaddTempExp")
        self.visitWeightKey = self.visitSchema.addField("weight", type=float,
                                                        doc="Weight for this visit in the coadd")
        self.ccdSchema = afwTable.ExposureTable.makeMinimalSchema()
        self.ccdVisitIdKey = self.ccdSchema.addField("visitId", type=numpy.int64,
                                                     doc="Foreign key for the visits (coaddTempExp) catalog");
        self.ccdGoodPixKey = self.ccdSchema.addField("goodpix", type=int,
                                                     doc="Number of good pixels in this CCD")
        if self.config.saveCcdWeights:
            self.ccdWeightKey = self.ccdSchema.addField("weight", type=float,
                                                        doc="Weight for this visit in the coadd")
        
    def makeCoaddInputs(self):
        return afwImage.CoaddInputs(self.visitSchema, self.ccdSchema)

    def makeCoaddTempExp(self, coaddInputs, visitId, nGoodPix, coaddTempExp):
        """
        Called once for each coaddTempExp by MakeCoaddTempExpTask, to add a single record
        to the visits table and attach the CoaddInputs object to the exposure.  Returns
        the record, so subclasses can call the base class method and fill additional fields.
        """
        coaddTempExp.getInfo().setCoaddInputs(coaddInputs)
        record = coaddInputs.visits.addNew()
        record.setId(visitId)
        if self.config.saveVisitGoodPix:
            record.setI(self.visitGoodPixKey, nGoodPix)
        record.setPsf(coaddTempExp.getPsf())
        record.setWcs(coaddTempExp.getWcs())
        return record

    def addCcdToTempExp(self, coaddInputs, visitId, nGoodPix, calexp, dataRef):
        """
        Called once for each CCD by MakeCoaddTempExpTask, to add a record to the ccds table.
        Returns the record, so subclasses can call the base class method and fill additional fields.

        Should be prepared for calexp to be None if self.config.saveErrorCcds is True.
        """
        record = coaddInputs.ccds.addNew()
        record.setId(dataRef.get("ccdExposureId"))
        record.setL(self.ccdVisitIdKey, visitId)
        record.setI(self.ccdGoodPixKey, nGoodPix)
        if calexp is not None:
            record.setPsf(calexp.getPsf())
            record.setWcs(calexp.getWcs())
        return record

    def addVisitToCoadd(self, coaddInputs, coaddTempExp, weight):
        """
        Called by AssembleCoaddTask when adding (a subset of) a coaddTempExp to a coadd.  The
        base class impementation extracts the CoaddInputs from the coaddTempExp and appends
        them to the given coaddInputs, filling in the weight column(s).

        Note that the passed coaddTempExp may be a subimage, but that this method will only be
        called for the first subimage

        Returns the record for the visit to allow subclasses to fill in additional fields.
        Warns and returns None if the provenance catalogs for the coaddTempExp are not usable.
        """
        tempExpInputs = coaddTempExp.getCoaddInputs()
        if len(tempExpInputs.visits) != 1:
            self.log.warn("Provenance for coaddTempExp should have exactly one record in visits table "
                          "(found %d).  Provenance for this visit will not be saved."
                          % len(tempExpInputs.visits))
            return None
        inputVisitRecord = tempExpInputs.visits[0];
        outputVisitRecord = coaddInputs.visits.addNew()
        outputVisitRecord.assign(inputVisitRecord)
        outputVisitRecord.setD(self.visitWeightKey, weight)
        for inputCcdRecord in tempExpInputs.ccds:
            if inputCcdRecord.getL(self.ccdVisitIdKey) != inputVisitRecord.getId():
                self.log.warn("Provenance for coaddTempExp with id %d contains CCDs with visitId=%d. "
                              "Provenance may be unreliable."
                              % (inputVisitRecord.getId(), inputCcdRecord.getL(self.ccdVisitIdKey)))
            outputCcdRecord = coaddInputs.ccds.addNew()
            outputCcdRecord.assign(inputCcdRecord)
            if self.config.saveCcdWeights:
                outputCcdRecord.setD(self.ccdWeightKey, weight)
        return inputVisitRecord
