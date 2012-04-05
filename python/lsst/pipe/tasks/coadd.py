#!/usr/bin/env python
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
import math
import os
import sys

import lsst.pex.config as pexConfig
import lsst.afw.detection as afwDetection
import lsst.afw.geom as afwGeom
import lsst.afw.math as afwMath
import lsst.coadd.utils as coaddUtils
import lsst.ip.diffim as ipDiffIm
import lsst.pipe.base as pipeBase
import lsst.pipe.tasks.calibrate as pipeCal
import lsst.meas.algorithms as measAlg
from lsst.pipe.tasks.coaddArgumentParser import CoaddArgumentParser

FWHMPerSigma = 2 * math.sqrt(2 * math.log(2))

class CoaddConfig(pexConfig.Config):
    """Config for CoaddTask
    """
    coadd    = pexConfig.ConfigField(dtype = coaddUtils.Coadd.ConfigClass, doc = "")
    warp     = pexConfig.ConfigField(dtype = afwMath.Warper.ConfigClass, doc = "")
    psfMatch = pexConfig.ConfigField(dtype = ipDiffIm.ModelPsfMatchTask.ConfigClass, doc = "a hack!")
    calibrate = pexConfig.ConfigField(dtype=pipeCal.CalibrateTask.ConfigClass,
                                      doc="Configuration for calibration")
    detection = pexConfig.ConfigField(dtype=measAlg.SourceDetectionTask.ConfigClass,
                                      doc="Configuration for source detection")
    measurement = pexConfig.ConfigField(dtype=measAlg.SourceMeasurementTask.ConfigClass,
                                        doc="Configuration for source measurement")
    doCalibrate = pexConfig.Field(doc="Calibrate coadd?", dtype=bool, default=False)
    doDetection = pexConfig.Field(doc="Detect sources?", dtype=bool, default=False)
    doMeasurement = pexConfig.Field(doc="Measure sources?", dtype=bool, default=False)

class CoaddTask(pipeBase.CmdLineTask):
    """Coadd images by PSF-matching (optional), warping and computing a weighted sum
    """
    ConfigClass = CoaddConfig
    _DefaultName = "coadd"
    
    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)
        self.makeSubtask("psfMatch", ipDiffIm.ModelPsfMatchTask)
        self.warper = afwMath.Warper.fromConfig(self.config.warp)
        self._prevKernelDim = afwGeom.Extent2I(0, 0)
        self._modelPsf = None

        self.makeSubtask("calibrate", pipeCal.CalibrateTask)
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        self.algMetadata = dafBase.PropertyList()
        if self.config.doDetection:
            self.makeSubtask("detection", measAlg.SourceDetectionTask, schema=self.schema)
        if self.config.doMeasurement:
            self.makeSubtask("measurement", measAlg.SourceMeasurementTask,
                             schema=self.schema, algMetadata=self.algMetadata)

    @classmethod
    def parseAndRun(cls, args=None, config=None, log=None):
        """Parse an argument list and run the command

        @param args: list of command-line arguments; if None use sys.arv
        @param config: config for task (instance of pex_config Config); if None use cls.ConfigClass()
        @param log: log (instance of pex_logging Log); if None use the default log
        """
        argumentParser = cls._makeArgumentParser()
        if config is None:
            config = cls.ConfigClass()
        parsedCmd = argumentParser.parse_args(config=config, args=args, log=log)
        task = cls(name = cls._DefaultName, config = parsedCmd.config, log = parsedCmd.log)

        taskRes = task.run(
            dataRefList = parsedCmd.dataRefList,
            bbox = parsedCmd.bbox,
            wcs = parsedCmd.wcs,
            desFwhm = parsedCmd.fwhm,
        )
        
        coadd = taskRes.coadd
        coaddExposure = coadd.getCoadd()
        weightMap = coadd.getWeightMap()
    
        filterName = coaddExposure.getFilter().getName()
        if filterName == "_unknown_":
            filterStr = "unk"
        coaddBaseName = "%s_filter_%s_fwhm_%s" % (task.getName(), filterName, parsedCmd.fwhm)
        coaddPath = coaddBaseName + ".fits"
        weightPath = coaddBaseName + "weight.fits"
        print "Saving coadd as %s" % (coaddPath,)
        coaddExposure.writeFits(coaddPath)
        print "saving weight map as %s" % (weightPath,)
        weightMap.writeFits(weightPath)
        # XXX persist additional results: sources, psf, aperture correction
    
    def getCalexp(self, dataRef, getPsf=True):
        """Return one "calexp" calibrated exposure, perhaps with psf
        
        @param dataRef: a sensor-level data reference
        @param getPsf: include the PSF?
        @return calibrated exposure with psf
        """
        exposure = dataRef.get("calexp")
        if getPsf:
            psf = dataRef.get("psf")
            exposure.setPsf(psf)
        return exposure
    
    def makeCoadd(self, bbox, wcs):
        """Make a coadd object, e.g. lsst.coadd.utils.Coadd
        
        @param[in] bbox: bounding box for coadd
        @param[in] wcs: WCS for coadd
        """
        return coaddUtils.Coadd.fromConfig(bbox=bbox, wcs=wcs, config=self.config.coadd)
    
    def makeModelPsf(self, exposure, desFwhm):
        """Construct a model PSF, or reuse the prior model, if possible
        
        The model PSF is a double Gaussian with core FWHM = desFwhm
        and wings of amplitude 1/10 of core and FWHM = 2.5 * core.
        
        @param exposure: exposure containing a psf; the model PSF will have the same dimensions
        @param desFwhm: desired FWHM of PSF, in pixels
        @return model PSF
        
        @raise RuntimeError if desFwhm <= 0
        """
        if desFwhm <= 0:
            raise RuntimeError("desFwhm = %s; must be positive" % (desFwhm,))
        psfKernel = exposure.getPsf().getKernel()
        kernelDim = psfKernel.getDimensions()
        if self._modelPsf is None or kernelDim != self._prevKernelDim:
            self._prevKernelDim = kernelDim
            self.log.log(self.log.INFO,
                "Create double Gaussian PSF model with core fwhm %0.1f and size %dx%d" % \
                (desFwhm, kernelDim[0], kernelDim[1]))
            coreSigma = desFwhm / FWHMPerSigma
            self._modelPsf = afwDetection.createPsf("DoubleGaussian", kernelDim[0], kernelDim[1],
                coreSigma, coreSigma * 2.5, 0.1)
        return self._modelPsf

    def run(self, dataRefList, bbox, wcs, desFwhm):
        """Coadd images by PSF-matching (optional), warping and computing a weighted sum
        
        PSF matching is to a double gaussian model with core FWHM = desFwhm
        and wings of amplitude 1/10 of core and FWHM = 2.5 * core.
        The size of the PSF matching kernel is the same as the size of the kernel
        found in the first calibrated science exposure, since there is no benefit
        to making it any other size.
        
        PSF-matching is performed before warping so the code can use the PSF models
        associated with the calibrated science exposures (without having to warp those models).
        
        Coaddition is performed as a weighted sum. See lsst.coadd.utils.Coadd for details.
    
        @param dataRefList: list of data identity dictionaries
        @param bbox: bounding box of coadd
        @param wcs: WCS of coadd
        @param desFwhm: desired FWHM of PSF, in science exposure pixels
                (note that the coadd often has a different scale than the science images);
                if 0 then no PSF matching is performed.
        @return: a pipeBase.Struct with fields:
        - coadd: a coaddUtils.Coadd object; call coadd.getCoadd() to get the coadd exposure
        """
        numExp = len(dataRefList)
        if numExp < 1:
            raise RuntimeError("No exposures to coadd")
        self.log.log(self.log.INFO, "Coadd %s calexp" % (numExp,))

        doPsfMatch = desFwhm > 0
    
        if not doPsfMatch:
            self.log.log(self.log.INFO, "No PSF matching will be done (desFwhm <= 0)")
    
        coadd = self.makeCoadd(bbox, wcs)
        for ind, dataRef in enumerate(dataRefList):
            self.log.log(self.log.INFO, "Processing exposure %d of %d: id=%s" % (ind+1, numExp, dataRef.dataId))
            exposure = self.getCalexp(dataRef, getPsf=doPsfMatch)
            if desFwhm > 0:
                modelPsf = self.makeModelPsf(exposure, desFwhm)
                self.log.log(self.log.INFO, "PSF-match exposure")
                psfRes = self.psfMatch.run(exposure, modelPsf)
                exposure = psfRes.psfMatchedExposure
            self.log.log(self.log.INFO, "Warp exposure")
            exposure = self.warper.warpExposure(wcs, exposure, maxBBox = bbox)
            coadd.addExposure(exposure)

        if self.config.doCalibrate:
            calib = self.calibrate.run(coadd.getCoadd())
            exposure = calib.exposure

        if self.config.doDetection:
            if exposure is None or calib is None:
                raise RuntimeError("Attempting source detection without calibration")
            table = afwTable.SourceTable.make(self.schema)
            table.setMetadata(self.algMetadata)
            detRet = self.detection.makeSourceCatalog(table, exposure)
            sources = detRet.sources

        if self.config.doMeasurement:
            if exposure is None or calib is None or sources is None:
                raise RuntimeError("Attempting source measurement without calibration and detection")
            self.measurement.run(exposure, sources, calib.apCorr)

        
        return pipeBase.Struct(
            coadd = coadd,
            calib = calib,
            sources = sources,
        )

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser
        """
        return CoaddArgumentParser(name=cls._DefaultName)
