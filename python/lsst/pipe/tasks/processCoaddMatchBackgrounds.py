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
# MERCHANTABILIY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the LSST License Statement and 
# the GNU General Public License along with this program.  If not, 
# see <http://www.lsstcorp.org/LegalNotices/>.
import sys, os, re
import numpy as num
import numpy.linalg as la
import datetime
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.geom as afwGeom
import lsst.afw.detection as afwDetect
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.ip.diffim as ipDiffim
import lsst.coadd.utils as coaddUtils
#uses coadd_utils v5_1:
from lsst.coadd.utils import ZeroPointScaler

from lsst.pipe.tasks.coadd import CoaddTask
from lsst.obs.sdss.convertfpM import convertfpM
from lsst.obs.sdss.convertasTrans import convertasTrans
from lsst.obs.sdss.convertpsField import convertpsField
from lsst.obs.sdss.converttsField import converttsField
import matplotlib.pyplot as plt
#Yes, I know scipy is a no no. TODO: replace with numpy function
from scipy.interpolate import SmoothBivariateSpline
try:
    import pymssql 
except:
    print "You need pymssql installed to access the DB"
    sys.exit(1)

#This script is very UW specific in general too
#TODO:  Needs to be refactored to use the butler and databases on ncsa
#adapted from Becker's obs_sdss/matchBackground2.py

class FieldMatch(object):
    def __init__(self, args):
        self.rootdir =  "/astro/net/pogo1/stripe82/imaging"
        self.run    = args[0]
        self.rerun  = args[1]
        self.filt   = args[2]
        self.camcol = args[3]
        self.field  = args[4]
        self.strip  = args[5]
        self.psfWidth = args[6]
        self.fluxMag0 = args[7]
        self.fluxMag0err = args[8]
        self.overlaps = args[9]
        self.fpC    = None
        self.fpM    = None
        self.wcs    = None
        self.psf    = None
        self.gain   = None
        self.calib  = None

    def loadfpC(self):
        self.fpC = getfpC(self.run, self.rerun, self.filt, self.camcol, self.field, self.rootdir)

    def loadfpM(self):
        self.fpM = getfpM(self.run, self.rerun, self.filt, self.camcol, self.field, self.rootdir)

    def loadWcs(self):
        asTrans = getasTrans(self.run, self.rerun, self.rootdir)
        if asTrans:
            self.wcs = convertasTrans(asTrans, self.filt, self.camcol, self.field)

    def loadPsf(self):
        self.psf = getpsField(self.run, self.rerun, self.filt, self.camcol, self.field, self.rootdir)

    def loadCalib(self):
        self.calib, self.gain = gettsField(self.run, self.rerun, self.filt, self.camcol, self.field, self.rootdir)

    def createExp(self):
        var  = afwImage.ImageF(self.fpC, True)
        var /= self.gain
        mi   = afwImage.MaskedImageF(self.fpC, self.fpM, var)
        exp  = afwImage.ExposureF(mi, self.wcs)
        exp.setPsf(self.psf)
        exp.setCalib(self.calib)
        return exp

class SigmaClippedCoaddConfig(CoaddTask.ConfigClass):
    sigmaClip = pexConfig.Field(
        dtype = float,
        doc = "sigma for outlier rejection",
        default = 3.0,
        optional = None,
    )
    clipIter = pexConfig.Field(
        dtype = int,
        doc = "number of iterations of outlier rejection",
        default = 2,
        optional = False,
    )

class MatchBackgroundsConfig(pexConfig.Config):
    warpingKernelName = pexConfig.Field(
        dtype = str,
        doc = """Type of kernel for remapping""",
        default = "lanczos3"
    )
    backgroundOrder = pexConfig.Field(
        dtype = int,
        doc = """Order of background Chebyshev""",
        default = 4
    )
    backgroundBinsize = pexConfig.Field(
        dtype = int,
        doc = """Bin size for background matching""",
        default = 128 #256
    )
    writeFits = pexConfig.Field(
        dtype = bool,
        doc = """Write output fits files""",
        default = False
    )
    outputPath = pexConfig.Field(
        dtype = str,
        doc = """Location of output files""",
        default = "/astro/net/pogo3/yusra/fits/testTimes4145/"
    )
    
    psfMatch = pexConfig.Field(
        dtype = bool,
        doc = """Psf match all images to the model Psf""",
        default = True
    )
    refPsfSize = pexConfig.Field(
        dtype = int,
        doc = """Size of reference Psf matrix; must be same size as SDSS Psfs""",
        default = 31
    )
    refPsfSigma = pexConfig.Field(
        dtype = float,
        doc = """Gaussian sigma for Psf FWHM (pixels)""",
        default = 3.0
    )
    useNN2 = pexConfig.Field(
        dtype = bool,
        doc = """Use NN2 to estimate difference image backgrounds.""",
        default = True
    )
    
    commonMask = pexConfig.Field(
        dtype = bool,
        doc = """True -  uses sum(all masks) for a common mask for all images in background estimate
                 False - uses only sum(2 mask) appropriate for each pair of images being matched""",
        default = True
    )
    
    useMean = pexConfig.Field(
        dtype = bool,
        doc = """True -  estimates difference image background as MEAN of unmasked pixels per bin
                 False - uses MEDIAN""",
        default = False
    )    
    useDetectionBackground = pexConfig.Field(
        dtype = bool,
        doc = """True -  True uses lsst.meas.detection.getBackground()
                False - masks, grids and fits Chebyshev polynomials to the backgrounds """,
        default = False
    )
   
    # With linear background model, this should fail
    # /astro/net/pogo1/stripe82/imaging/6447/40/corr/1/fpC-006447-r1-0718.fit.gz
    maxBgRms = pexConfig.Field(
        dtype = float,
        doc = """Maximum RMS of matched background differences, in counts""",
        default = 5.0
    )

    # Clouds
    # /astro/net/pogo1/stripe82/imaging/7071/40/corr/1/fpC-007071-r1-0190.fit.gz
    minFluxMag0 = pexConfig.Field(
        dtype = float,
        doc = """Minimum flux for star of mag 0""",
        default = 1.0e+10
    )

class SigmaClippedCoaddTask(CoaddTask):
    ConfigClass = SigmaClippedCoaddConfig
    _DefaultName = "SigmaClippedCoadd"
    def __init__(self, *args, **kwargs):
        CoaddTask.__init__(self, *args, **kwargs)

        # Stats object for sigma clipping
        self.statsCtrl = afwMath.StatisticsControl()
        self.statsCtrl.setNumSigmaClip(self.config.sigmaClip)
        self.statsCtrl.setNumIter(self.config.clipIter)
        self.statsCtrl.setAndMask(afwImage.MaskU.getPlaneBitMask(self.config.badMaskPlanes))

    @pipeBase.timeMethod   
    def normalizeForCoadd(self, exp):
        # WARNING: IF YOU HAVE BACKGROUND MATCHED IMAGES THIS WILL
        # DESTROY THEIR MATCHING.  RUN THIS BEFORE BACKGROUND MATCHING
        print  "CoaddZeroPoint is: ", self.config.coaddZeroPoint
        calib = exp.getCalib()
        scaleFac = 1.0 / calib.getFlux(self.config.coaddZeroPoint)
        mi  = afwImage.MaskedImageF(exp.getMaskedImage(), True)
        mi *= scaleFac
        self.log.log(self.log.INFO, "Normalized using scaleFac=%0.3g" % (scaleFac))

        # Copy Psf, Wcs, Calib (bad idea?)
        normExp = afwImage.ExposureF(exp, True)
        normExp.setMaskedImage(mi)
        return normExp

    @pipeBase.timeMethod   
    def weightForCoadd(self, exp, weightFactor = 1.0):
        statObj = afwMath.makeStatistics(exp.getMaskedImage().getVariance(), exp.getMaskedImage().getMask(),
                                         afwMath.MEANCLIP, self.statsCtrl)
        meanVar, meanVarErr = statObj.getResult(afwMath.MEANCLIP)
        weight = weightFactor / float(meanVar)
        return weight

    @pipeBase.timeMethod   
    def run(self, refExp, expList):
        #import pdb; pdb.set_trace()
        # Calib object for coadd (i.e. zeropoint)
        #print coaddUtils.zeroPointScaler
        print self.config.coaddZeroPoint
        zpScaler =  ZeroPointScaler(zeroPoint = self.config.coaddZeroPoint)
        coaddCalib  = zpScaler.getCalib()

        # Destination for the coadd
        refMi    = refExp.getMaskedImage()
        coaddMi  = refMi.Factory(refMi.getBBox(afwImage.PARENT))

        # Vectors for images to coadd and their weights
        maskedImageList = afwImage.vectorMaskedImageF()
        weightList = []
        
        # Add reference image first
        maskedImageList.append(refExp.getMaskedImage())
        weightList.append(self.weightForCoadd(refExp))
        
        # All the other matched images
        for exp in expList:
            maskedImageList.append(exp.getMaskedImage())
            weightList.append(self.weightForCoadd(exp))

        print "weightList = ", weightList
        try:
            coaddMi = afwMath.statisticsStack(maskedImageList, afwMath.MEANCLIP, self.statsCtrl, weightList)
        except Exception, e:
            self.log.log(self.log.ERR, "Outlier rejected coadd failed: %s" % (e,))
            sys.exit(1)

        # Post processing
        coaddUtils.setCoaddEdgeBits(coaddMi.getMask(), coaddMi.getVariance())

        coaddExp = afwImage.ExposureF(coaddMi, refExp.getWcs())
        coaddExp.setCalib(zpScaler.getCalib())
        self.log.log(self.log.INFO, "COADD with %d images" % (len(expList) + 1))
        self.log.log(self.log.INFO, "")
        self.log.log(self.log.INFO, self.metadata.toString())
        return coaddExp

            
class MatchBackgrounds(pipeBase.Task):
    ConfigClass = MatchBackgroundsConfig
    _DefaultName = "MatchBackgrounds" 
    def __init__(self, refrun, rerun, camcol, filt, field, seeingCutoff, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)
        self.refrun  = refrun
        self.rerun   = rerun
        self.camcol  = camcol
        self.filt    = filt
        self.field   = field
        self.seeingCutoff = seeingCutoff
        self.rootdir =  "/astro/net/pogo1/stripe82/imaging"
        #how big is your reference image (how many fields):
        self.refLen  = 1
        #initialize:        
        self.numimages = 0
        self.asTrans = getasTrans(self.refrun, self.rerun, self.rootdir)
        self.warper  = afwMath.Warper(self.config.warpingKernelName)
        self.refPsf  = afwDetect.createPsf("DoubleGaussian", self.config.refPsfSize, self.config.refPsfSize, self.config.refPsfSigma)
        
        config          = ipDiffim.ModelPsfMatchTask.ConfigClass()
        config.kernel.active.kernelSize = self.config.refPsfSize // 2
        self.psfMatcher = ipDiffim.ModelPsfMatchTask(config=config)
        self.coadder    = SigmaClippedCoaddTask()



    @pipeBase.timeMethod
    def run(self, nMax = 10, **kwargs):
        expArr = []
        for i in range(self.refLen):
            fpCRef = getfpC(self.refrun, self.rerun, self.filt, self.camcol, self.field + i, self.rootdir)
            if not fpCRef:
                return

            fpMRef = getfpM(self.refrun, self.rerun, self.filt, self.camcol, self.field + i, self.rootdir)
            if not fpMRef: 
                return

            wcsRef = convertasTrans(self.asTrans, self.filt, self.camcol, self.field + i)
            if not wcsRef:
                return

            psfRef = getpsField(self.refrun, self.rerun, self.filt, self.camcol, self.field + i, self.rootdir)
            if not psfRef:
                return

            calib, gain = gettsField(self.refrun, self.rerun, self.filt, self.camcol, self.field + i, self.rootdir)
            if (not calib) or (not gain):
                return
            
            print psfRef, calib, gain
            # Assemble an exposure out of this
            varRef  = afwImage.ImageF(fpCRef, True)
            varRef /= gain
            miRef   = afwImage.MaskedImageF(fpCRef, fpMRef, varRef)
            expArr.append(afwImage.ExposureF(miRef, wcsRef))
            expArr[i].setPsf(psfRef)
            expArr[i].setCalib(calib)

            
        exp = self.stitchExposures(expArr)
        exp.setPsf(psfRef)
        exp.setCalib(calib)
        # Ref exposure for this field
        if self.config.psfMatch:
            self.refExp = self.psfMatcher.run(exp, self.refPsf).psfMatchedExposure
        else:
            self.refExp = exp
        print "sizes:", self.refExp.getWidth(),self.refExp.getHeight()
        
        # Matches per runs
        
        matches = self.queryClueFiltered(self.refExp.getBBox(), self.refExp.getWcs(), self.filt, self.camcol)
        self.warpedExp       = {}
        #self.warpedExpCalibCenters tracks the centers of the images so that
        #the original Calibs can be interpolated over the stitched images. 
        self.warpedExpCalibCenters = {} 
        self.bgMatchedExp    = {}
        
        timeStart =  datetime.datetime.now()
        self.loadMatches(matches, nMax = nMax)
        timeMatch =  datetime.datetime.now()
        
        if self.config.useDetectionBackground:
            self.matchBackgroundsBkgd()
        else:
            self.matchBackgroundsNew()
            
        timeCoadd = datetime.datetime.now()
        self.createCoadd()
        self.loadTimesToDatabase(timeStart, timeMatch, timeCoadd)
        self.log.log(self.log.INFO, "")
        self.log.log(self.log.INFO, self.metadata.toString())

    def loadTimesToDatabase(self,timeStart, timeMatch, timeCoadd):
        endtime =  datetime.datetime.now()
        secondsLoad = (timeMatch - timeStart).seconds
        secondsMatch = (timeCoadd - timeMatch).seconds
        secondsCoadd = (endtime - timeCoadd).seconds
        DBconnection = pymssql.connect(user="LSST-2",password="L$$TUser",host="fatboy",database="[dev-yusra]")
        db = DBconnection.cursor()
        sql  = "INSERT INTO backgroundMatchTimes ("
        sql += "refrun, field, filter, camcol, psfMatch, useNN2, commonMask,backgroundOrder, numImages, timeLoad, timeMatch, timeCoadd) "
        sql += "VALUES (%d, %d, '%s', %d, %d, %d, %d, %d, %d, %d, %d, %d)"% (self.refrun, self.field, self.filt, self.camcol, self.config.psfMatch, self.config.useNN2, self.config.commonMask, self.config.backgroundOrder, self.numImages,  secondsLoad, secondsMatch, secondsCoadd)
        print sql 
        db.execute(sql)
        DBconnection.commit()     
        DBconnection.close()
        
    def queryClueFiltered(self, bbox, wcs, filt, camcol):
        #setup for UW database
        db      = pymssql.connect(user="clue-1", password="wlZH2xWy", host="fatboy", database="clue")
        cursor  = db.cursor()
        #add RA direction 0.5*Height buffers so that we are ensured
        #  to get surrounding calibs
        #This will let us consistently interpolate the zeropoint
        buffer = bbox.getHeight()/2.
        LLC = wcs.pixelToSky(bbox.getMinX(), bbox.getMinY())
        ULC = wcs.pixelToSky(bbox.getMinX(), bbox.getMaxY())
        URC = wcs.pixelToSky(bbox.getMaxX(), bbox.getMaxY())
        LRC = wcs.pixelToSky(bbox.getMaxX(), bbox.getMinY())
        #Assemble sql query
        sql  = "select run,rerun,filter,camCol,field,strip, psfWidth, fluxMag0, fluxMag0err, "
        sql += "bbox.STIntersects(geography::STGeomFromText('POLYGON (("
        sql += " %f %f," % (LLC[0].asDegrees(), LLC[1].asDegrees())
        sql += " %f %f," % (ULC[0].asDegrees(), ULC[1].asDegrees())
        sql += " %f %f," % (URC[0].asDegrees(), URC[1].asDegrees())
        sql += " %f %f," % (LRC[0].asDegrees(), LRC[1].asDegrees())
        sql += " %f %f ))', 4326)) "% (LLC[0].asDegrees(), LLC[1].asDegrees())
        
        LLC = wcs.pixelToSky(bbox.getMinX(), bbox.getMinY()-buffer)
        ULC = wcs.pixelToSky(bbox.getMinX(), bbox.getMaxY()+buffer)
        URC = wcs.pixelToSky(bbox.getMaxX(), bbox.getMaxY()+buffer)
        LRC = wcs.pixelToSky(bbox.getMaxX(), bbox.getMinY()-buffer)
        
        sql += " from [clue].[dbo].[viewStripe82JoinAll] WITH(INDEX(idx_bbox))"
        sql += " where bbox.STIntersects(geography::STGeomFromText('POLYGON (("
        sql += " %f %f," % (LLC[0].asDegrees(), LLC[1].asDegrees())
        sql += " %f %f," % (ULC[0].asDegrees(), ULC[1].asDegrees())
        sql += " %f %f," % (URC[0].asDegrees(), URC[1].asDegrees())
        sql += " %f %f," % (LRC[0].asDegrees(), LRC[1].asDegrees())
        sql += " %f %f ))', 4326))=1" % (LLC[0].asDegrees(), LLC[1].asDegrees())
        sql += " and filter='%s' and camcol=%i" % (filt, camcol)
#        sql += " and psfWidth < %0.3f " % (seeingCutoff)        
        sql += " and blacklist = 0 and quality between 2 and 3"
        sql += " and fluxmag0 is not null "  #ensures that we have it on disk at UW
#        sql += " and (run =  4145 OR run between 6000  and 7000)" #was used for debugging
        sql += " order by run asc, field asc;"   
        
        self.log.log(self.log.INFO, "SQL: %s" % (sql))
        cursor.execute(sql)
        results = cursor.fetchall()
        db.close()
        # Note: lets just add each strip up and we'll use the N/S overlap
        # to match coadd backgroudns
    
        amatches = []
        for result in results:
            amatches.append(FieldMatch(result))
    
        strips = num.array([x.strip for x in amatches])
        runs   = num.array([x.run for x in amatches])
        idx    = num.where(runs == self.refrun)[0]
        strip  = list(set(strips[idx]))
        if len(strip) != 1:
            print "ERROR in strips"
            sys.exit(1)
    
        idxs     = num.where(strips == strip)[0]
        smatches = []
        for idx in idxs:
            smatches.append(amatches[idx])
        return smatches

    @pipeBase.timeMethod
    def loadMatches(self, matches, nMax = None):
        """Finds matches on disk; Psf matches them to common Psf;
        stitches them together; remaps to fiducial field; stores in
        self.warpedExp"""

        runs  = num.array([x.run for x in matches])
        uruns = list(set(runs))
        uruns.sort()

        #loop through runs in the matches
        nProc = 0
        for run in uruns:
            if nMax and nProc >= nMax:
                break

            #skip it if it's the ref-run
            if run == self.refrun:
                continue

            self.log.log(self.log.INFO, "") # spacer
            self.log.log(self.log.INFO, "RUNNING %d vs. %d" % (run, self.refrun))

            #find indexes of the matches with run iteration
            idxs = num.where(runs == run)[0]
            #quick check that the image covers the reference:
            #should replace with a geometry test
            if len(idxs) < (self.refLen +2): 
                continue 

            #also want to move on to next run if any one of the matchs has a psfWidth > seeing limit
            badSeeingFlag = 0
            
            #create an array of matches for this run iteration
            runMatches = []
            runImageMatches = []
            for idx in idxs:
                runMatches.append(matches[idx])
                if matches[idx].overlaps == 1:
                    runImageMatches.append(matches[idx])
                    print "seeing ", matches[idx].psfWidth, self.seeingCutoff
                    if matches[idx].psfWidth > self.seeingCutoff:
                        badSeeingFlag = 1
                else:
                    print "not overlapping", matches[idx].psfWidth
                    
            if badSeeingFlag > 0:
                print "run over seeing limit"
                continue
            
            calibList = {}
            nloaded = 0
            calibloaded = 0
            for match in runMatches:
                match.loadWcs()
                match.loadCalib()
                if (match.wcs and match.calib and match.calib.getFluxMag0() > self.config.minFluxMag0):
                    calibloaded +=1
                    #list of key: calibs and value: their positions
                    #position is set to the center of each field
                    calibList[match.calib] = match.wcs.pixelToSky(self.refExp.getWidth()/2., self.refExp.getHeight()/2.)
                if match.overlaps == 1:
                    match.loadfpC()   
                    match.loadfpM()
                    match.loadPsf()
                    if (match.fpC and match.fpM and 
                        match.wcs and match.psf and 
                        match.gain and match.calib and 
                        match.calib.getFluxMag0() > self.config.minFluxMag0):
                        nloaded += 1


            if calibloaded != len(runMatches):
                self.log.log(self.log.INFO, "Not able to load all images, skipping to next run")
                continue
            else:
                self.log.log(self.log.INFO, "OK")

            #save this calib list    
            self.warpedExpCalibCenters[run] = calibList           
            stitch = self.stitchMatches(runImageMatches)
            if stitch is None:
                print "Does not cover!!"
                continue
            
            # Keep Wcs and calib of first image,
            # but the flux scaling will be done with the calibList 
            exp = afwImage.ExposureF(stitch, runImageMatches[0].wcs)
            exp.setPsf(self.refPsf)
            exp.setCalib(runImageMatches[0].calib)


#            if self.config.writeFits:
            if 0:
                exp.writeFits(os.path.join(self.config.outputPath, 
                                           "psfmatch-%06d-%s%d-%04d-r%06d.fits" % 
                                           (self.refrun, self.filt, self.camcol, self.field, run)))

            # Do need to keep this
            self.warpedExp[run] = self.warper.warpExposure(self.refExp.getWcs(), 
                                                           exp, 
                                                           destBBox = self.refExp.getBBox(afwImage.PARENT))

            # Do after warping, since it loses it in warping
            self.warpedExp[run].setPsf(self.refPsf)

            if self.config.writeFits:
                self.warpedExp[run].writeFits(os.path.join(self.config.outputPath, 
                                                           "warp-%06d-%s%d-%04d-r%06d.fits" % 
                                                           (self.refrun, self.filt, self.camcol, self.field, run)))

            nProc += 1
     
        self.numImages = nProc


    @pipeBase.timeMethod
    def stitchExposures(self, exposures, overlap = 128):
        # Stitching together neighboring images from the matching run
        nloaded = len(exposures)
        width   = exposures[0].getWidth()
        height  = exposures[0].getHeight() * nloaded - overlap * (nloaded - 1) 
        stitch  = afwImage.MaskedImageF(width, height)

        for i in range(len(exposures)):
            Exp = exposures[i]       
            symin  = (i + 0) * Exp.getHeight() 
            symax  = (i + 1) * Exp.getHeight()
            iymin  = 0
            iymax  = Exp.getHeight()
            
            if i > 0:
                iymin   = overlap//2
                symin  -= (i - 1) * overlap + overlap//2
                symax  -= (i - 0) * overlap 

            print stitch.getImage().getArray()[symin:symax,:].shape,  Exp.getMaskedImage().getImage().getArray()[iymin:iymax,:].shape
            # Note transpose of getArray()
            try:
                stitch.getImage().getArray()[symin:symax,:]    = Exp.getMaskedImage().getImage().getArray()[iymin:iymax,:]
                stitch.getMask().getArray()[symin:symax,:]     = Exp.getMaskedImage().getMask().getArray()[iymin:iymax,:]
                stitch.getVariance().getArray()[symin:symax,:] = Exp.getMaskedImage().getVariance().getArray()[iymin:iymax,:]
            except:
                import pdb; pdb.set_trace()
     
        return afwImage.ExposureF(stitch,exposures[0].getWcs())


    @pipeBase.timeMethod
    def stitchMatches(self, runMatches, overlap = 128, testme = 0):
        # Stitching together neighboring images from the matching run
        nloaded = len(runMatches)
        print nloaded
        width   = runMatches[0].fpC.getWidth()
        height  = runMatches[0].fpC.getHeight() * nloaded - overlap * (nloaded - 1) 
        stitch  = afwImage.MaskedImageF(width, height)

        for i in range(len(runMatches)):
            match  = runMatches[i]
            matchExp = match.createExp()
            # Psf match before stitching!
            if self.config.psfMatch:
                psfmatchedExp = self.psfMatcher.run(matchExp, self.refPsf).psfMatchedExposure
            else: # no psf matching:
                psfmatchedExp = matchExp
            
            symin  = (i + 0) * match.fpC.getHeight() 
            symax  = (i + 1) * match.fpC.getHeight() 
            iymin  = 0
            iymax  = match.fpC.getHeight()
            
            if i > 0:
                iymin   = overlap//2
                symin  -= (i - 1) * overlap + overlap//2
                symax  -= (i - 0) * overlap
    
            # Note transpose of getArray()
            try:
                stitch.getImage().getArray()[symin:symax,:]    = psfmatchedExp.getMaskedImage().getImage().getArray()[iymin:iymax,:]
                stitch.getMask().getArray()[symin:symax,:]     = psfmatchedExp.getMaskedImage().getMask().getArray()[iymin:iymax,:]
                stitch.getVariance().getArray()[symin:symax,:] = psfmatchedExp.getMaskedImage().getVariance().getArray()[iymin:iymax,:]
            except:
                import pdb; pdb.set_trace()              
            # Clear up memory
            match.fpC = None
            match.fpM = None

        return stitch

    def normalizeByInterpZeroPoint(self,run):
        #calibFlux = self.warpedExp[run].getCalib().getFlux(self.config.coaddZeroPoint)
        calibs = self.warpedExpCalibCenters[run].keys()
        positions = self.warpedExpCalibCenters[run].values()
        #hold positions in first column and flux's in second colum
        fluxArr = num.empty([len(self.warpedExpCalibCenters[run]),2])
        i=0
        for calib in calibs:
            fluxArr[i][1]= calib.getFlux(self.coadder.config.coaddZeroPoint)
            fluxArr[i][0]  = self.refExp.getWcs().skyToPixel(positions[i])[1] #only care about the RA position
            i +=1

        #find sort by RA so that the interpolator will be happy
        ind= num.lexsort((fluxArr[:,1],fluxArr[:,0]))
        #add in extra points on the boundaries of the fields defined by linear interp
        #this will ensure that if we want to mosiac the strip:
        # when we coadd the neighboring field the flux scaling in the 128pix overlap region will be roughly the same
        # Only way to get them EXACTLY the same is to do a linear fit, but the drawback is discontinuities in slope
        # at the center points
        boundaries = num.empty([len(self.warpedExpCalibCenters[run])-1,2])
        boundaries[:,0] = (fluxArr[ind,0][1:] + fluxArr[ind,0][0:-1])/2.
        boundaries[:,1] = (fluxArr[ind,1][1:] + fluxArr[ind,1][0:-1])/2.
        fluxArr =num.vstack((fluxArr, boundaries))
        #shuffle in the new points:
        ind = num.lexsort((fluxArr[:,1],fluxArr[:,0]))
        
        degree = min(4,len(fluxArr-1))
        #change this to numpy polyfit!
        from scipy.interpolate import splrep, splev
        #create an "image" to hold the flux scale factor
        mi  = afwImage.MaskedImageF(self.warpedExp[run].getMaskedImage(), True)
        # decisions: In order to have the refExp boundary solutions to be identical there are two choices:
        # 1) use linear interpolation
        # 2) use a d=4 polynomial/spline to the 3 real points + 2 extra points on the refExp boundaries
        # defined by the linear solution (we have no additional info to make it otherwise) (describe above)
        height = mi.getBBox().getHeight()
        width = mi.getBBox().getWidth()
        fluxArr[:,1] = 1./fluxArr[:,1]
        spline = splrep(fluxArr[ind,0], fluxArr[ind,1],k=degree, s=0)
        eval = splev(range(height),spline)               
        evalGrid, _ =   num.meshgrid(eval,range(width))
        mi.getImage().getArray()[:,:] *=  evalGrid.T
        mi.getVariance().getArray()[:,:] *=  evalGrid.T**2
        #no change to the mask
        print "Normalized using scaleFac: ",    fluxArr[ind,1], "At pixel values: ", fluxArr[ind,0]
        
        # Copy Psf, Wcs, Calib (bad idea?)
        normExp = afwImage.ExposureF(self.warpedExp[run], True)
        normExp.setMaskedImage(mi)
        return normExp

    def getChebFitPoly(self, bbox, degree, X, Y, Z, dZ):
        poly  = afwMath.Chebyshev1Function2D(int(degree), bbox)          
        terms = list(poly.getParameters())
        Ncell = num.sum(num.isfinite(Z))
        Nterm = len(terms)               
        m  = num.zeros((Ncell, Nterm))
        b  = num.zeros((Ncell))
        iv = num.zeros((Ncell))
        nc = 0
        #Would be nice if the afwMath.ChebyshevFunction2D could make the matrix for fitting:
        #so that looping wasn't necessary
        for na in range(Ncell):
            for nt in range(Nterm):
                terms[nt] = 1.0
                poly.setParameters(terms)
                m[nc, nt] = poly(X[na], Y[na])
                terms[nt] = 0.0
            b[nc]  = Z[na]
            iv[nc] = 1/(dZ[na]*dZ[na])
            nc += 1
        M    = num.dot(num.dot(m.T, num.diag(iv)), m)       
        B    = num.dot(num.dot(m.T, num.diag(iv)), b)
        Soln = num.linalg.solve(M,B)
        poly.setParameters(Soln)
        return poly 

    @pipeBase.timeMethod   
    def matchBackgroundsNew(self):
        """Puts images on a common zeropoint by interpolating across matches;
        then background matches them;
        saves results in self.bgMatchedExp after checking some
        quality flags
        """
        # IMPORTANT DETAIL : match zeropoints before matching backgrounds!
        self.refExp = self.coadder.normalizeForCoadd(self.refExp)
        for run in self.warpedExp.keys():
            self.warpedExp[run] = self.normalizeByInterpZeroPoint(run)
        # IMPORTANT DETAIL 

        if self.config.writeFits:
            self.refExp.writeFits(os.path.join(self.config.outputPath, 
                                               "exp-%06d-%s%d-%04d.fits" % 
                                               (self.refrun, self.filt, self.camcol, self.field)))

        refMask = self.refExp.getMaskedImage().getMask().getArray()
        refArr = self.refExp.getMaskedImage().getImage().getArray()
        
        runsToMatch = self.warpedExp.keys()
        expsToMatch = self.warpedExp.values()

        skyMask = num.sum(num.array([x.getMaskedImage().getMask().getArray() for x in expsToMatch]), 0)
        maskArr  = num.array([x.getMaskedImage().getMask().getArray() for x in expsToMatch])
        skyArr = num.array([x.getMaskedImage().getImage().getArray() for x in expsToMatch])
        Nim = len(skyArr)

        # Find all unmasked (sky) pixels
        idx = num.where((refMask + skyMask) == 0)

        width  = self.refExp.getMaskedImage().getWidth()
        height = self.refExp.getMaskedImage().getHeight()
        nbinx  = width  // self.config.backgroundBinsize
        nbiny  = height // self.config.backgroundBinsize
        if (width  % self.config.backgroundBinsize != 0):
            nbinx += 1
            
        if (height % self.config.backgroundBinsize != 0):
            nbiny += 1


        #used lists/append to protect against the case where
        #    a bin has NO valid pixels and should not be included in the fit
        bgX = []
        bgY = []
        bgZ = []
        bgdZ = []      
        for  i in range(Nim):
            bgX.append(num.array([]))
            bgY.append(num.array([]))
            bgZ.append(num.array([]))
            bgdZ.append(num.array([]))

        #COMMON MASK
        #Pixel masked in ANY of the N images, will be masked
        #Are the various options confusing?
        if self.config.commonMask:    
            for biny in range(nbiny):          
                ymin = biny * self.config.backgroundBinsize
                ymax = min((biny + 1) * self.config.backgroundBinsize, height)
                idxy = num.where( (idx[0] >= ymin) & (idx[0] < ymax) )[0]
                for binx in range(nbinx):
                    xmin   = binx * self.config.backgroundBinsize
                    xmax   = min((binx + 1) * self.config.backgroundBinsize, width)
                    idxx   = num.where( (idx[1] >= xmin) & (idx[1] < xmax) )[0]
                    inreg  = num.intersect1d(idxx, idxy)
                    if len(inreg) > 1:
                        area0 = refArr[idx[0][inreg],idx[1][inreg]]
                        if self.config.useNN2:
                            A, E = self.calcAECommonMask(area0, Nim, skyArr, idx, inreg)
                            V, Verr, Vint, Vext, chi   = self.NN2(A,E)
                        for i in range(Nim):
                            areai     = skyArr[i][idx[0][inreg],idx[1][inreg]]
                            area      = area0 - areai
                            bgX[i]    =  num.append(bgX[i], 0.5 * (xmin + xmax))
                            bgY[i]    =  num.append(bgY[i], 0.5 * (ymin + ymax))
                            if self.config.useMean:
                                bgZ[i]    =  num.append(bgZ[i], num.mean(area[num.where(~num.isnan(area))]))
                                bgdZ[i]   =  num.append(bgdZ[i],
                                                    num.std(area[num.where(~num.isnan(area))])/
                                                    num.sqrt(num.size(area[num.where(~num.isnan(area))])))
                                if self.config.useNN2:
                                    """commmon mask + mean + nn2 will give same results as
                                       common mask + mean. Not using nn2"""
                            else:
                                if self.config.useNN2:
                                    bgZ[i] = num.append(bgZ[i], V[i+1]-V[0])
                                    bgdZ[i] = num.append(bgdZ[i],Verr[i+1]) 
                                else:                                    
                                    bgZ[i]    =  num.append(bgZ[i],
                                                            num.median(area[num.where(~num.isnan(area))]))
                                    bgdZ[i]   =  num.append(bgdZ[i],
                                                    num.std(area[num.where(~num.isnan(area))])/
                                                    num.sqrt(num.size(area[num.where(~num.isnan(area))])))

        #Pairwise masks (different mask for each pair of images)
        #This loops over bins and finds 
        else:
            for biny in range(nbiny):
                ymin = biny * self.config.backgroundBinsize
                ymax = min((biny + 1) * self.config.backgroundBinsize, height)
                for binx in range(nbinx):
                    xmin   = binx * self.config.backgroundBinsize
                    xmax   = min((binx + 1) * self.config.backgroundBinsize, width)
                    #############################################################
                    area0 =     refArr[ymin:ymax][:,xmin:xmax]
                    area0Mask = refMask[ymin:ymax][:,xmin:xmax]
                    areaSky =      skyArr[:][:,ymin:ymax][:,:,xmin:xmax]
                    areaSkyMask =  maskArr[:][:,ymin:ymax][:,:,xmin:xmax]
                    #zeros = num.where(areaSkyMask ==0)
                    #if len(num.unique(zeros[0]))<Nim:
                        #at least one image has no unmasked pixels
                        #do somethign about it?
                    if self.config.useNN2:
                        A, E = self.calcAEMaskPairs(area0, area0Mask, areaSky, areaSkyMask, Nim)
                        V, Verr, Vint, Vext, chi   = self.NN2(A,E)
                    #fill arrays
                    for i in range(Nim):              
                        bgX[i]    =  num.append(bgX[i], 0.5 * (xmin + xmax))
                        bgY[i]    =  num.append(bgY[i], 0.5 * (ymin + ymax))
                        if self.config.useNN2:    
                            bgZ[i] = num.append(bgZ[i],V[i+1]-V[0])                 
                            bgdZ[i] = num.append(bgdZ[i],Verr[i+1])
                        else:
                            mask = area0Mask + areaSkyMask[i]
                            ix, iy = num.where(mask == 0)
                            #print area0[ix,iy], areaSky[i][ix,iy]
                            area =  area0[ix,iy] - areaSky[i][ix,iy]
                            if self.config.useMean:
                                bgZ[i]    =  num.append(bgZ[i],
                                                    num.mean(area[num.where(~num.isnan(area))])
                                                    )
                            #median
                            else:
                                bgZ[i]    =  num.append(bgZ[i],
                                                    num.median(area[num.where(~num.isnan(area))])
                                                    )
                            #same for both mean and median
                            bgdZ[i]   =  num.append(bgdZ[i],
                                                    num.std(area[num.where(~num.isnan(area))])
                                                    /num.sqrt(num.size(area[num.where(~num.isnan(area))]))      
                                                    )
                
        #import pdb; pdb.set_trace()
        # Fit Polynomial for each image grid
        for i in range(Nim):
            bbox  = afwGeom.Box2D(self.refExp.getMaskedImage().getBBox())
            poly = self.getChebFitPoly(bbox, self.config.backgroundOrder, bgX[i],bgY[i],bgZ[i],bgdZ[i])  
            run = runsToMatch[i]
            exp = expsToMatch[i]
            im  = exp.getMaskedImage()
            im += poly

            if self.config.writeFits:
                exp.writeFits(os.path.join(self.config.outputPath, 
                                           "match-%06d-%s%d-%04d-r%06d.fits" % 
                                           (self.refrun, self.filt, self.camcol, self.field, run)))
                
            # DEBUGGING INFO
            tmp  = afwImage.MaskedImageF(im, True)
            tmp -= self.refExp.getMaskedImage()

            if self.config.writeFits:
                tmp.writeFits(os.path.join(self.config.outputPath, 
                                           "diff-%06d-%s%d-%04d-r%06d.fits" % 
                                           (self.refrun, self.filt, self.camcol, self.field, run)))
            #import pdb; pdb.set_trace()
            # Lets see some stats!
            area = tmp.getImage().getArray()[idx]
            self.log.log(self.log.INFO, "Diff BG %06d: mean=%0.3g med=%0.3g std=%0.3g npts=%d" % (
                    run, num.mean(area), num.median(area), num.std(area), len(area))
            )
            
            if num.std(area) < self.config.maxBgRms:
                self.bgMatchedExp[run] = exp
            
            self.warpedExp[run] = None # Clear memory

    @pipeBase.timeMethod   
    def matchBackgroundsBkgd(self):
        """Uses lsst.meas.algorithms.detection to calculate background
               for comparison with polynomial fitting method
        
        Puts images on a common zeropoint by interpolating across matches;
        then background matches them;
        saves results in self.bgMatchedExp after checking some
        quality flags"""
        #from matplotlib.backends.backend_pdf import PdfPages
        #pdf = PdfPages('BackgroundMatching.pdf')
        # IMPORTANT DETAIL : match zeropoints before matching backgrounds!
        self.refExp = self.coadder.normalizeForCoadd(self.refExp)
        for run in self.warpedExp.keys():
            self.warpedExp[run] = self.normalizeByInterpZeroPoint(run)
        # IMPORTANT DETAIL : match zeropoints before matching b0ackgrounds!

        if self.config.writeFits:
            self.refExp.writeFits(os.path.join(self.config.outputPath, 
                                               "exp-%06d-%s%d-%04d.fits" % 
                                               (self.refrun, self.filt, self.camcol, self.field)))


        refMask   = self.refExp.getMaskedImage().getMask().getArray()
        refArr    = self.refExp.getMaskedImage().getImage().getArray()        
        runsToMatch = self.warpedExp.keys()
        expsToMatch = self.warpedExp.values()
        skyMask  = num.sum(num.array([x.getMaskedImage().getMask().getArray() for x in expsToMatch]), 0)
        skyArr   = num.array([x.getMaskedImage().getImage().getArray() for x in expsToMatch])
        Nim      = len(skyArr)
        for i in range(Nim):
            if self.config.writeFits:
                expsToMatch[i].writeFits(os.path.join(self.config.outputPath, 
                                           "scaled-%06d-%s%d-%04d-r%06d.fits" % 
                                           (self.refrun, self.filt, self.camcol, self.field, runsToMatch[i])))
        import lsst.meas.algorithms.detection as detection
        dconfig = detection.BackgroundConfig()
        dconfig.statisticsProperty = dconfig.statisticsProperty
        dconfig.undersampleStyle = dconfig.undersampleStyle
        dconfig.binSize = 512                 
        # Function for each image
        for i in range(Nim):
            run = runsToMatch[i]
            exp = expsToMatch[i]
            im  = exp.getMaskedImage()
            diff = im.Factory(im,True)
            diff -= self.refExp.getMaskedImage()
            try:
                bkgd = detection.getBackground(diff, dconfig)
            except Exception, e:
                print >> sys.stderr, "Failed to fit background for %d: %s" % (run, e)
                continue
            
            im -= bkgd.getImageF()
            if self.config.writeFits:
                exp.writeFits(os.path.join(self.config.outputPath, 
                                           "match-%06d-%s%d-%04d-r%06d.fits" % 
                                           (self.refrun, self.filt, self.camcol, self.field, run)))
                
            # DEBUGGING INFO
            tmp  = afwImage.MaskedImageF(im, True)
            tmp -= self.refExp.getMaskedImage()

            if self.config.writeFits:
                tmp.writeFits(os.path.join(self.config.outputPath, 
                                           "diff-%06d-%s%d-%04d-r%06d.fits" % 
                                           (self.refrun, self.filt, self.camcol, self.field, run)))            
            # Lets see some stats!
            idx = num.where((refMask + skyMask) == 0)
            area = tmp.getImage().getArray()[idx]
            self.log.log(self.log.INFO, "Diff BG %06d: mean=%0.3g med=%0.3g std=%0.3g npts=%d" % (
                    run, num.mean(area), num.median(area), num.std(area), len(area))
            )
            
            if num.std(area) < self.config.maxBgRms:
                self.bgMatchedExp[run] = exp
            
            self.warpedExp[run] = None # Clear memory


    @pipeBase.timeMethod
    def createCoadd(self):
        coaddExp = self.coadder.run(self.refExp, self.bgMatchedExp.values())
        coaddExp.setPsf(self.refPsf)
        #put options in the coadd file name:
        if self.config.commonMask:
            nameStr = "oneMask"
        else:
            nameStr = "maskPairs"            
        if self.config.useNN2:
            nameStr += "NN2"
        if self.config.psfMatch:
            nameStr += "psf%d"%(self.config.refPsfSigma)
            
        coaddExp.writeFits(os.path.join(self.config.outputPath, 
                                        "coadd-%06d-%s%d-%04d-%03d-%s-Cheb%d.fits" % 
                                        (self.refrun, self.filt, self.camcol, self.field,self.numImages,nameStr,self.config.backgroundOrder)))


    def calcAECommonMask(self, area0, Nim, skyArr, idx, inreg):
        n = Nim +1
        Aij = num.zeros((n, n))
        Eij = num.zeros((n,n))
        nRange = range(n) #The plus 1 because we're adding the reference image
        #do zero first (reference image)
        i=0
        for j in range(i+1, n):
            areaj  = skyArr[j-1][idx[0][inreg],idx[1][inreg]]
            area = area0 - areaj
            if self.config.useMean:
                #this option is pointless because it give same results as non-NN2
                Aij[i][j] = num.mean(area[num.where(~num.isnan(area))])
            else:
                Aij[i][j] = num.median(area[num.where(~num.isnan(area))])
            Aij[j][i] = -Aij[i][j]
            Eij[i][j] = num.std(area[num.where(~num.isnan(area))])/num.sqrt(num.size(area[num.where(~num.isnan(area))]))
            Eij[j][i] = Eij[i][j]
        for i in range(1,n):
            areai  = skyArr[i-1][idx[0][inreg],idx[1][inreg]]
            for j in range(i+1, Nim+1):
                areaj  = skyArr[j-1][idx[0][inreg],idx[1][inreg]]
                area = areai - areaj
                if self.config.useMean:
                    #this option is pointless because it give same results as non-NN2
                    Aij[i][j] = num.mean(area[num.where(~num.isnan(area))])
                else:
                    Aij[i][j] = num.median(area[num.where(~num.isnan(area))])
                Aij[j][i] = -Aij[i][j]
                Eij[i][j] = num.std(area[num.where(~num.isnan(area))])/num.sqrt(num.size(area[num.where(~num.isnan(area))]))
                Eij[j][i] = Eij[i][j]
        return Aij, Eij

    def calcAEMaskPairs(self, ref, refMask, sArr, sMaskArr, Nim):
        """Calculates the A, E matrices for input to NN2.
           takes 2-d arrays for a single bin. sArr and sMaskArr are 3-d arrays: (Nim,ypixel, xpixels)
           Very sloooooow"""
        n = Nim +1
        Aij = num.zeros((n,n))
        Eij = num.zeros((n,n))
        nRange = range(n) #The plus 1 because we're adding the reference image
        #do zero first (reference image)
        i=0
        for j in range(i+1, n):
            #get mask for pair
            mask = refMask + sMaskArr[j-1]
            #find indexes where mask is zero
            idx, idy = num.where(mask == 0)
            areaj  = sArr[j-1][idx,idy]
            area0  = ref[idx,idy]
            area = area0 - areaj
            if self.config.useMean:
                Aij[i][j] = num.mean(area[num.where(~num.isnan(area))])
            else:    
                Aij[i][j] = num.median(area[num.where(~num.isnan(area))])
            Aij[j][i] = -Aij[i][j]
            Eij[i][j] = num.std(area[num.where(~num.isnan(area))])/num.sqrt(num.size(area[num.where(~num.isnan(area))]))
            Eij[j][i] = Eij[i][j]
        for i in range(1,n):
            for j in range(i+1, Nim+1):
                mask = sMaskArr[i-1] + sMaskArr[j-1]
                idx, idy = num.where(mask == 0)
                areai  = sArr[i-1][idx,idy]
                areaj  = sArr[j-1][idx,idy]
                area = areai - areaj
                if self.config.useMean:
                    Aij[i][j] = num.mean(area[num.where(~num.isnan(area))])    
                else:
                    Aij[i][j] = num.median(area[num.where(~num.isnan(area))])
                Aij[j][i] = -Aij[i][j]
                Eij[i][j] = num.std(area[num.where(~num.isnan(area))])/num.sqrt(num.size(area[num.where(~num.isnan(area))]))
                Eij[j][i] = Eij[i][j]
        return Aij, Eij

    def NN2(self, A,E):
        ##########################################################
        #Implemented from Barris et al. 2005
        #and A.B. Newman and Rest:  http://www.ctio.noao.edu/supermacho/NN2/
        ###########################################################
        #TODO :Currently contains loops to protect against dividing by zero.
        #      Could certainly be made faster using numpy triangles and num.where !=0 
        n, m = A.shape
        if n!=m:
            print "A is not square"
            return None
        V = num.zeros(n)
        Vint = num.zeros(n) #Barris internal err
        Vext = num.zeros(n) #Barris External err
        Verr = num.zeros(n) #Total error in quadrature. 
        y = num.zeros(n)
        nRange = range(n)
        dof = (n*(n-1))/2 - (n-1)
        cov = num.zeros((n, n))
        wgtave = 0.
        #Calculate Barris et al: 1/<E>^2 (eq. 4) 
        #take average of 1/E^2 for non-diagonal components for half of matrix
        #Looping not the best way to do it, but it's safe for zeros
        for j in range(n):      
            for i in range(j):
                if E[i,j] != 0:
                    wgtave += 1.0/(E[i,j]*E[i,j])
        wgtave /= (n*(n-1))/2   
        #Calculate Cik in Barris et al eq(9).
        #This implementation has a sign flip compared with the paper
        cov[:,:] = wgtave 
        y[:] = 0.
        for j in range(n):
            for i in range(n):
                if E[i,j] != 0 and i != j: 
                    wgt = 1.0/(E[i,j]*E[i,j])
                else:
                    wgt = 0.0
                cov[i,j] += wgt
                cov[j,j] -= wgt    # Add and sub cancel when i==j
                y[j] += A[j,i] * wgt 
        #Solve for V:
        cov = la.inv(cov)        
        V = num.dot(y,cov)
        #Calculate Internal Error
        #Barris says  "The inverse of C yields uncertainties for V from the square root of the diagonal elements" 
        try:
            Vint = num.sqrt(num.diag(-cov))
        except:
            print "WARNING: Tried to take sqrt of negative number in interr"
        #Barris says: "The covariances [can be calculated] from normalizeing the off-diagonal elements by the two diagonal terms...
        #This assums error matrix truly does represent a Gaussian"
        #YA: We don't need this for now
            # for i in range(j):
            #     try: cov[i,j] = cov[j,i] = -cov[i,j] / (cov[i,i]*cov[j,j])
            #     except:
            #         print "WARNING: Tried to divide by zero"
            #         return None
        #Calculate Chi2
        chi = 0.0
        for j in range(n):
            for i in range(j+1,n):
                if E[i,j] != 0:
                    #Newman's code didn't square the nominator. Why not??
                    chi += ((A[i,j] - (V[j]-V[i]))**2 / E[i,j])**2
        Vint *= num.sqrt(chi/dof)
        ########Barris way to calculate Vext###########
        #As opposed to the Tonry way to calculate Vext. I'll will add that as well. 
        #Barris Eq 16:
        Dik = num.ones((n,n))    
        num.fill_diagonal(Dik, n-1)
        Dik = la.inv(Dik)
        # Create y vector so that [Dik]*[sigma**2] = [y] is the equation to solve.
        #Waring: diagonal elements of Eij need to be zero (Sum for i!=j)
        y = num.apply_over_axes(num.sum, E**2, 0).flatten()  
        Vext = num.dot(y, Dik)
        if min(Vext) < 0:
            print "WARNING: Tried to take sqrt of negative number in exterr_barris_et_al (min. is %d)" % min(Vext)
            print Vext
        Vext = num.sqrt(abs(Vext))
        #############################################
        #calculate total error:
        Verr = num.sqrt(Vint**2 + Vext**2)
        return V, Verr, Vint, Vext, chi


        
def getfpC(run, rerun, filt, camcol, field,rootdir):
    fname = os.path.join(rootdir, str(run), str(rerun), "corr", str(camcol), "fpC-%06d-%s%d-%04d.fit.gz" % (run, filt, camcol, field))
    print fname
    if os.path.isfile(fname):
        im  = afwImage.ImageF(fname)
        im -= 1000 
        return im
    return None

def getfpM(run, rerun, filt, camcol, field,rootdir):
    fname1 = os.path.join(rootdir, str(run), str(rerun), "objcs", str(camcol), "fpM-%06d-%s%d-%04d.fit" % (run, filt, camcol, field))
    fname2 = os.path.join(rootdir, str(run), str(rerun), "objcs", str(camcol), "fpM-%06d-%s%d-%04d.fit.gz" % (run, filt, camcol, field))
    for fname in (fname1, fname2):
        print fname
        if os.path.isfile(fname):
            try:
                return convertfpM(fname, allPlanes = True)
            except:
                return None
    return None

def getasTrans(run, rerun,rootdir):
    fname = os.path.join(rootdir, str(run), str(rerun), "astrom", "asTrans-%06d.fit" % (run))
    print fname
    if os.path.isfile(fname):
        return fname
    fname = os.path.join(rootdir, str(run), str(rerun), "astrom", "asTrans-%06d.fit.gz" % (run))
    print fname
    if os.path.isfile(fname):
        return fname
    return None

def getpsField(run, rerun, filt, camcol, field,rootdir):
    fname = os.path.join(rootdir, str(run), str(rerun), "objcs", str(camcol), "psField-%06d-%d-%04d.fit" % (run, camcol, field))
    print fname
    if os.path.isfile(fname):
        return convertpsField(fname, filt)
    return None

def gettsField(run, rerun, filter, camcol, field,rootdir):
    fname1 = os.path.join(rootdir, str(run), str(rerun), "calibChunks", str(camcol), "tsField-%06d-%d-%d-%04d.fit" % (run, camcol, rerun, field))
    fname2 = os.path.join(rootdir, str(run), str(rerun), "calibChunks", str(camcol), "tsField-%06d-%d-%d-%04d.fit.gz" % (run, camcol, rerun, field))
    shortname = 'tsField-%06d-%d-%d-%04d.fit' % (run, camcol, rerun, field)
    for fname in (fname1, fname2):
        print fname
        if os.path.isfile(fname):
            try:
                return converttsField(fname,filter)
            except:
                return None, None
            

def loopThroughRun(refrun, camcol, filt, fieldStart, fieldEnd, nMax, seeingCutoff, refLen):
    for field in range(fieldStart, fieldEnd, refLen):
        print "Creating Coadd for refrun %i, field %i"%(refrun,field)
        matcher = MatchBackgrounds(refrun, 40, camcol, filt, field, seeingCutoff)
        matcher.run(nMax = nMax)
        matcher = None


def doThese():
     loopThroughRun(4145, 1, 'g', 134, 137,1000, 5.,1)

def doOne(refrun, camcol, filt, field, seeingCutoff, nMax):
    print refrun, camcol, filt, field, seeingCutoff
    matcher = MatchBackgrounds(refrun, 40, camcol, filt, field, seeingCutoff)
    matcher.run(nMax = nMax)
    sys.exit(1)
        
        
if __name__ == '__main__':
    refrun  = int(sys.argv[1])
    camcol  = int(sys.argv[2])
    filt    = sys.argv[3]
    field   = int(sys.argv[4])
    nMax    = int(sys.argv[5])
    seeingCutoff = float(sys.argv[6])

    doOne(refrun, camcol, filt, field, seeingCutoff, nMax)

