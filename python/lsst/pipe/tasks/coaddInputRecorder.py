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

import numpy

import lsst.pex.config as pexConfig
import lsst.afw.table as afwTable
import lsst.afw.image as afwImage
import lsst.pipe.base as pipeBase
from lsst.meas.algorithms import CoaddPsf, makeCoaddApCorrMap

__all__ = ["CoaddInputRecorderTask"]


class CoaddInputRecorderConfig(pexConfig.Config):
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
        dtype=bool, default=True, optional=False,
        doc=("Save the total number of good pixels in each coaddTempExp (redundant with a sum of"
             " good pixels in associated CCDs)")
    )
    saveCcdWeights = pexConfig.Field(
        dtype=bool, default=True, optional=False,
        doc=("Save weights in the CCDs table as well as the visits table?"
             " (This is necessary for easy construction of CoaddPsf, but otherwise duplicate information.)")
    )


class CoaddTempExpInputRecorder:
    """A helper class for CoaddInputRecorderTask, managing the CoaddInputs object for that single
    CoaddTempExp.  This will contain a single 'visit' record for the CoaddTempExp and a number of 'ccd'
    records.

    Should generally be created by calling CoaddInputRecorderTask.makeCoaddTempExp().

    Parameters
    ----------
    task : `lsst.pipe.tasks.coaddInputRecorder.CoaddInputRecorderTask`
        The CoaddInputRecorderTask that is utilising us.
    visitId : `int`
        Identifier (integer) for the visit.
    num : `int`
        Number of CCDs for this visit that overlap this patch (for reserving memory).
    """

    def __init__(self, task, visitId, num=0):
        self.task = task
        self.coaddInputs = self.task.makeCoaddInputs()
        self.coaddInputs.visits.reserve(1)
        if num > 0:
            self.coaddInputs.ccds.reserve(num)
        self.visitRecord = self.coaddInputs.visits.addNew()
        self.visitRecord.setId(visitId)

    def addCalExp(self, calExp, ccdId, nGoodPix):
        """Add a 'ccd' record for a calexp just added to the CoaddTempExp.

        Parameters
        ----------
        calExp : `lsst.afw.image.Exposure`
            Calibrated exposure just added to the CoaddTempExp, or None in case of
            failures that should nonetheless be tracked.  Should be the original
            calexp, in that it should contain the original Psf and Wcs, not the
            warped and/or matched ones.
        ccdId : `int`
            A unique numeric ID for the Exposure.
        nGoodPix : `int`
            Number of good pixels this image will contribute to the CoaddTempExp.
            If saveEmptyCcds is not set and this value is zero, no record will be
            added.
        """
        if nGoodPix == 0 and not self.task.config.saveEmptyCcds:
            return
        record = self.coaddInputs.ccds.addNew()
        record.setId(ccdId)
        record.setL(self.task.ccdVisitKey, self.visitRecord.getId())
        try:
            record.setI(self.task.ccdCcdKey, calExp.getDetector().getId())
        except Exception as e:
            self.task.log.warning("Error getting detector serial number in visit %d; using -1; error=%s",
                                  self.visitRecord.getId(), e)
            record.setI(self.task.ccdCcdKey, -1)
        record.setI(self.task.ccdGoodPixKey, nGoodPix)
        if calExp is not None:
            self._setExposureInfoInRecord(exposure=calExp, record=record)
            if self.task.config.saveCcdWeights:
                record.setD(self.task.ccdWeightKey, 1.0)  # No weighting or overlap when warping
            record.set(self.task.ccdFilterKey, calExp.getFilter().physicalLabel)

    def finish(self, coaddTempExp, nGoodPix=None):
        """Finish creating the CoaddInputs for a CoaddTempExp.

        Parameters
        ----------
        coaddTempExp : `lsst.afw.image.Exposure`
            Exposure object from which to obtain the PSF, WCS, and bounding
            box for the entry in the 'visits' table.  On return, the completed
            CoaddInputs object will be attached to it.
        nGoodPix : `int`
            Total number of good pixels in the CoaddTempExp; ignored unless
            saveVisitGoodPix is true.
        """
        self._setExposureInfoInRecord(exposure=coaddTempExp, record=self.visitRecord)
        if self.task.config.saveVisitGoodPix:
            self.visitRecord.setI(self.task.visitGoodPixKey, nGoodPix)
        coaddTempExp.getInfo().setCoaddInputs(self.coaddInputs)
        wcs = coaddTempExp.getWcs()
        if False:
            # This causes a test failure, pending fix in issue HSC-802
            coaddTempExp.setPsf(CoaddPsf(self.coaddInputs.ccds, wcs))
        apCorrMap = makeCoaddApCorrMap(self.coaddInputs.ccds, coaddTempExp.getBBox(afwImage.PARENT), wcs)
        coaddTempExp.getInfo().setApCorrMap(apCorrMap)

    def _setExposureInfoInRecord(self, exposure, record):
        """Set exposure info and bbox in an ExposureTable record.

        Parameters
        ----------
        exposure : `lsst.afw.image.ExposureF`
            Exposure whose info is to be recorded.
        record : `Unknown`
            Record of an ExposureTable to set.
        """
        info = exposure.getInfo()
        record.setPsf(info.getPsf())
        record.setWcs(info.getWcs())
        record.setPhotoCalib(info.getPhotoCalib())
        record.setApCorrMap(info.getApCorrMap())
        record.setValidPolygon(info.getValidPolygon())
        record.setVisitInfo(info.getVisitInfo())
        record.setBBox(exposure.getBBox())
        record.setTransmissionCurve(info.getTransmissionCurve())


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
            self.visitGoodPixKey = self.visitSchema.addField("goodpix", type=numpy.int32,
                                                             doc="Number of good pixels in the coaddTempExp")
        self.visitWeightKey = self.visitSchema.addField("weight", type=float,
                                                        doc="Weight for this visit in the coadd")
        self.ccdSchema = afwTable.ExposureTable.makeMinimalSchema()
        self.ccdCcdKey = self.ccdSchema.addField("ccd", type=numpy.int32, doc="cameraGeom CCD serial number")
        self.ccdVisitKey = self.ccdSchema.addField("visit", type=numpy.int64,
                                                   doc="Foreign key for the visits (coaddTempExp) catalog")
        self.ccdGoodPixKey = self.ccdSchema.addField("goodpix", type=numpy.int32,
                                                     doc="Number of good pixels in this CCD")
        if self.config.saveCcdWeights:
            self.ccdWeightKey = self.ccdSchema.addField("weight", type=float,
                                                        doc="Weight for this visit in the coadd")
        self.visitFilterKey = self.visitSchema.addField("filter", type=str, size=32,
                                                        doc="Physical filter associated with this visit.")
        self.ccdFilterKey = self.ccdSchema.addField("filter", type=str, size=32,
                                                    doc="Physical filter associated with this visit.")

    def makeCoaddTempExpRecorder(self, visitId, num=0):
        """Return a CoaddTempExpInputRecorder instance to help with saving a CoaddTempExp's inputs.

        Parameters
        ----------
        visitId : `Unknown`
        num : `int`, optional
            Number of CCDs for this visit that overlap this patch (for reserving memory).

        Notes
        -----
        The visitId may be any number that is unique for each :that goes into a coadd,
        but ideally should be something more meaningful that can be used to reconstruct a data ID.
        """
        return CoaddTempExpInputRecorder(self, visitId, num=num)

    def makeCoaddInputs(self):
        """Create a CoaddInputs object with schemas defined by the task configuration."""
        return afwImage.CoaddInputs(self.visitSchema, self.ccdSchema)

    def addVisitToCoadd(self, coaddInputs, coaddTempExp, weight):
        """Called by AssembleCoaddTask when adding (a subset of) a coaddTempExp to a coadd.  The
        base class impementation extracts the CoaddInputs from the coaddTempExp and appends
        them to the given coaddInputs, filling in the weight column(s).

        Parameters
        ----------
        coaddInputs : `lsst.afw.Image.CoaddInputs`
            A record of the observations that are included in the coadd.
        coaddTempExp : `lsst.afw.image.Exposure`
            Exposure object from which to obtain the PSF, WCS, and bounding
            box for the entry in the 'visits' table.  On return, the completed
            CoaddInputs object will be attached to it.
        weight : `Unknown`

        Returns
        -------
        inputVisitRecord : `Unknown`
            The record for the visit to allow subclasses to fill in additional fields or
            None if the inputRecorder catalogs for the coaddTempExp are not usable.

        Notes
        -----
        Note that the passed coaddTempExp may be a subimage, but that this method will only be
        called for the first subimage.
        """
        tempExpInputs = coaddTempExp.getInfo().getCoaddInputs()
        if len(tempExpInputs.visits) != 1:
            self.log.warning("CoaddInputs for coaddTempExp should have exactly one record in visits table "
                             "(found %d).  CoaddInputs for this visit will not be saved.",
                             len(tempExpInputs.visits))
            return None
        inputVisitRecord = tempExpInputs.visits[0]
        outputVisitRecord = coaddInputs.visits.addNew()
        outputVisitRecord.assign(inputVisitRecord)
        outputVisitRecord.setD(self.visitWeightKey, weight)
        outputVisitRecord.set(self.visitFilterKey, coaddTempExp.getFilter().physicalLabel)
        for inputCcdRecord in tempExpInputs.ccds:
            if inputCcdRecord.getL(self.ccdVisitKey) != inputVisitRecord.getId():
                self.log.warning("CoaddInputs for coaddTempExp with id %d contains CCDs with visit=%d. "
                                 "CoaddInputs may be unreliable.",
                                 inputVisitRecord.getId(), inputCcdRecord.getL(self.ccdVisitKey))
            outputCcdRecord = coaddInputs.ccds.addNew()
            outputCcdRecord.assign(inputCcdRecord)
            if self.config.saveCcdWeights:
                outputCcdRecord.setD(self.ccdWeightKey, weight)
            outputCcdRecord.set(self.ccdFilterKey, coaddTempExp.getFilter().physicalLabel)
        return inputVisitRecord
