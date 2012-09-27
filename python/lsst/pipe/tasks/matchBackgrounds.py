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
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base.argumentParser import IdValueAction

class MatchBackgroundsConfig(pexConfig.Config):
    
    useDetectionBackground = pexConfig.Field(
        dtype = bool,
        doc = """True -  True uses lsst.meas.detection.getBackground()
                 False - masks, grids and fits Chebyshev polynomials to the backgrounds """,
        default = False
    )
    
    #Cheb options
    backgroundOrder = pexConfig.Field(
        dtype = int,
        doc = """Order of background Chebyshev""",
        default = 4
    )
    
    backgroundBinsize = pexConfig.Field(
        dtype = int,
        doc = """Bin size for background matching""",
        default = 128 
    )
      
    gridStat = pexConfig.ChoiceField(
        dtype = str,
        doc = """Type of statistic to use for the grid points""",     
        default = "MEAN",
        allowed = {
            "MEAN": "mean",
            "MEDIAN": "median"
            }
    )
    
    #Misc options 
    datasetType = pexConfig.Field(
        dtype = str,
        doc = """Name of data product to fetch (calexp, etc)""",
        default = "coaddTempExp"
    )        
      
 
class MatchBackgroundsTask(pipeBase.CmdLineTask):
    ConfigClass = MatchBackgroundsConfig
    _DefaultName = "matchBackgrounds"
    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)

    @pipeBase.timeMethod
    def run(self, refDataRef, toMatchDataRef):
        self.log.log(self.log.INFO, "Matching background of %s to %s" % (toMatchDataRef.dataId, refDataRef.dataId))
        
        if not refDataRef.datasetExists(self.config.datasetType):
            raise pipeBase.TaskError("Data id %s does not exist" % (refDataRef.dataId))
        refExposure = refDataRef.get(self.config.datasetType)

        if not toMatchDataRef.datasetExists(self.config.datasetType):
            raise pipeBase.TaskError("Data id %s does not exist" % (toMatchDataRef.dataId))
        sciExposure = toMatchDataRef.get(self.config.datasetType)

        matchBackgroundModel, matchedExposure = self.matchBackgrounds(refExposure, sciExposure)

        return pipeBase.Struct(
            matchBackgroundModel = matchBackgroundModel,
            matchedExposure = matchedExposure
        )

    @pipeBase.timeMethod
    def matchBackgrounds(self, refExposure, sciExposure):
        """
        Matches a science exposure's background level to that of a reference image.
        sciExposure's image is overwritten in memory, mask preserved

        Potential TO DOs:
            check to make sure they aren't the same image?
        """
        
        #Check that exps are the same shape. Return if they aren't
        if (sciExposure.getDimensions() != refExposure.getDimensions()):
            wSci, hSci = sciExposure.getDimensions()
            wRef, hRef = refExposure.getDimensions()
            self.log.log(self.log.WARN,
                         "Cannot Match. Exposures different shapes. sci:(%i, %i) vs. ref:(%i, %i)" %
                         (wSci,hSci,wRef,hRef))
            self.log.log(self.log.WARN, "Returning None")
            #Could return None, sciExposure
            #but it would have to be caught by the coadder
            return None, None
        
        mask  = refExposure.getMaskedImage().getMask().getArray()
        mask += sciExposure.getMaskedImage().getMask().getArray()
        #get indicies of masked pixels
        #  Currently gets all non-zero pixels,
        #  but this can be tweaked to get whatver types you want. 
        ix,iy = num.where((mask) > 0)
        
        #make difference image array
        diffArr  = refExposure.getMaskedImage().getImage().getArray()
        diffArr -= sciExposure.getMaskedImage().getImage().getArray()

        #set image array pixels to nan if masked
        diffArr[ix, iy] = num.nan

        #bin
        width, height  = refExposure.getDimensions()
        xedges = num.arange(0, width, self.config.backgroundBinsize)
        yedges = num.arange(0, height, self.config.backgroundBinsize)
        xedges = num.hstack(( xedges, width ))  #add final edge
        yedges = num.hstack(( yedges, height )) #add final edge
        #Initialize lists to hold grid.
        #Use lists/append to protect against the case where
        #a bin has no valid pixels and should not be included in the fit
        bgX = []
        bgY = []
        bgZ = []
        bgdZ = []

        for ymin, ymax in zip(yedges[0:-1],yedges[1:]):
            for xmin, xmax in zip(xedges[0:-1],xedges[1:]):
                print ymin, ymax, xmin,xmax
                area = diffArr[ymin:ymax][:,xmin:xmax]
                # if there are less than 2 pixels with non-nan,non-masked values:
                #TODO: num.where is expensive and perhaps
                #      we can do it once above the loop and just compare indices here
                idxNotNan = num.where(~num.isnan(area))
                if len(idxNotNan[0]) >= 2:
                    bgX.append(0.5 * (xmin + xmax))
                    bgY.append(0.5 * (ymin + ymax))
                    bgdZ.append( num.std(area[idxNotNan])
                                                /num.sqrt(num.size(area[idxNotNan])))
                    if self.config.gridStat == "MEDIAN":
                       bgZ.append(num.median(area[idxNotNan])) 
                    elif self.config.gridStat == 'MEAN':
                       bgZ.append(num.mean(area[idxNotNan]))
                    else:
                        print "Unspecified grid statistic. Using mean"
                        bgZ.append(num.mean(area[idxNotNan]))

        #Fit grid with polynomial           
        bbox  = afwGeom.Box2D(refExposure.getMaskedImage().getBBox())
        matchBackgroundModel = self.getChebFitPoly(bbox, self.config.backgroundOrder, bgX,bgY,bgZ,bgdZ)  
        im  = sciExposure.getMaskedImage()
        #matches sciExposure in place in memory
        im += matchBackgroundModel
       
        #To Do: Perform RMS check here to make sure new sciExposure is matched well enough?

        #returns the background Model, and the matched science exposure
        return matchBackgroundModel, sciExposure    

    def getChebFitPoly(self, bbox, degree, X, Y, Z, dZ):
        poly  = afwMath.Chebyshev1Function2D(int(degree), bbox)          
        terms = list(poly.getParameters())
        Ncell = num.sum(num.isfinite(Z)) #number of bins to fit: usually nbinx*nbiny
        Nterm = len(terms)               
        m  = num.zeros((Ncell, Nterm))
        b  = num.zeros((Ncell))
        iv = num.zeros((Ncell))
        nc = 0
        #Would be nice if the afwMath.ChebyshevFunction2D could make the matrix for fitting:
        #so that looping here wasn't necessary
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


    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser
        """
        return MatchBackgroundsParser(name=cls._DefaultName, datasetType=cls.ConfigClass.datasetType)

class MatchBackgroundsParser(pipeBase.ArgumentParser):
    """A version of lsst.pipe.base.ArgumentParser specialized for background matching
    """
    def __init__(self, *args, **kwargs):
        pipeBase.ArgumentParser.__init__(self, *args, **kwargs)
        self.add_argument("--refid", nargs="*", action=IdValueAction,
                          help="unique, full reference background data ID, e.g. --refid visit=865833781 raft=2,2 sensor=1,0 filter=3 patch=3 tract=77,69", metavar="KEY=VALUE") 

#    def _makeDataRefList(self, namespace):
#        """Make namespace.dataRefList from namespace.dataIdList
#        """
#        import pdb; pdb.set_trace()
#        datasetType = namespace.config.datasetType
#        validKeys = namespace.butler.getKeys(datasetType=datasetType, level=self._dataRefLevel)
#
#        namespace.dataRefList = []
#        for dataId in namespace.dataIdList:
#            # tract and patch are required
#            for key in validKeys:
#                if key not in dataId:
#                    self.error("--id must include " + key)
#            dataRef = namespace.butler.dataRef(
#                datasetType = datasetType,
#                dataId = dataId,
#            )
#            namespace.dataRefList.append(dataRef)
        
