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
#
"""Read preprocessed bright stars and stack them to build an extended
PSF model."""

__all__ = ["MeasureExtendedPsfTask"]

import traceback
import sys
import numpy as np

from lsst.pipe import base as pipeBase
from lsst.pipe.tasks.assembleCoadd import AssembleCoaddTask
import lsst.pex.config as pexConfig
from lsst.afw import math as afwMath
from lsst.afw import image as afwImage
from lsst.geom import Extent2I
from lsst.daf.persistence import butlerExceptions as bE


class MeasureExtendedPsfConfig(pexConfig.Config):
    """Configuration parameters for MeasureExtendedPsfTask
    """
    subregionSize = pexConfig.ListField(
        dtype=int,
        doc="Size, in pixels, of the subregions over which the stacking be "
            "iteratively performed.",
        default=(20, 20)
    )
    stackingStatistic = pexConfig.ChoiceField(
        dtype=str,
        doc="Type of statistic to use for stacking.",
        default="MEANCLIP",
        allowed={
            "MEAN": "mean",
            "MEDIAN": "median",
            "MEANCLIP": "clipped mean",
        }
    )
    numSigmaClip = pexConfig.Field(
        dtype=float,
        doc="Sigma for outlier rejection; ignored if stackingStatistic != 'MEANCLIP'.",
        default=4
    )
    numIter = pexConfig.Field(
        dtype=int,
        doc="Number of iterations of outlier rejection; ignored if atackingStatistic != 'MEANCLIP'.",
        default=3
    )
    badMaskPlanes = pexConfig.ListField(
        dtype=str,
        doc="Mask planes that, if set, lead to associated pixels not being included in the stacking of the "
            "bright star stamps.",
        default=('BAD', 'CR', 'CROSSTALK', 'EDGE', 'NO_DATA', 'SAT', 'SUSPECT', 'UNMASKEDNAN')
    )
    modelFilename = pexConfig.Field(
        dtype=str,
        doc="Full path to where the extended PSF model fits file should be saved.",
        default="extendedPsf.fits"
    )
    doMagCut = pexConfig.Field(
        dtype=bool,
        doc="Reapply mag cut before stacking?",
        default=False
    )
    magLimit = pexConfig.Field(
        dtype=float,
        doc="Magnitude limit, in Gaia G; all stars brighter than this value will be processed",
        default=18
    )


class ExtendedPsfTaskRunner(pipeBase.TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        return [(parsedCmd.butler, parsedCmd.selectId.idList)]

    def __call__(self, parsedCmd):
        butler, selectDataList = parsedCmd
        task = self.TaskClass(config=self.config, log=self.log)
        if self.doRaise:
            results = task.runDataRef(butler, selectDataList)
        else:
            try:
                results = task.runDataRef(butler, selectDataList)
            except Exception as e:
                task.log.fatal("Failed: %s" % e)
                if not isinstance(e, pipeBase.TaskError):
                    traceback.print_exc(file=sys.stderr)
        # task.writeMetadata(butler)
        if self.doReturnResults:
            return results


class MeasureExtendedPsfTask(pipeBase.CmdLineTask):
    """TODO: write docstring
    """
    ConfigClass = MeasureExtendedPsfConfig
    RunnerClass = ExtendedPsfTaskRunner
    _DefaultName = "measureExtendedPsf"

    def __init__(self, initInputs=None, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser.
        """
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--selectId", "brightStarStamps",
                               help="data ID, e.g. --selectId visit=6789 ccd=0..9",
                               ContainerClass=pipeBase.DataIdContainer)
        return parser

    @pipeBase.timeMethod
    def runDataRef(self, butler, selectDataList=None):
        self.log.info("Stacking bright star stamps from %i different exposures." % (len(selectDataList)))
        # read in example set of full stamps
        dataId = {'visit': selectDataList[0]["visit"], 'ccd': selectDataList[0]["ccd"]}
        bss = butler.get("brightStarStamps", dataId)
        exampleStamp = bss[0].starStamp
        # create model image
        extPsf = afwImage.MaskedImageF(exampleStamp.getBBox())
        # divide model image into smaller subregions
        subregionSize = Extent2I(*self.config.subregionSize)
        subBBoxes = AssembleCoaddTask._subBBoxIter(extPsf.getBBox(), subregionSize)
        # compute approximate number of subregions
        nbSubregions = int(extPsf.getDimensions()[0]/subregionSize[0] + 1)*int(
            extPsf.getDimensions()[1]/subregionSize[1] + 1)
        # set up stacking statistic
        statsControl = afwMath.StatisticsControl()
        statsControl.setNumSigmaClip(self.config.numSigmaClip)
        statsControl.setNumIter(self.config.numIter)
        badMasks = self.config.badMaskPlanes
        statsControl.setWeighted(True)
        if badMasks:
            andMask = exampleStamp.mask.getPlaneBitMask(badMasks[0])
            for bm in badMasks[1:]:
                andMask = andMask | exampleStamp.mask.getPlaneBitMask(bm)
            statsControl.setAndMask(andMask)
        statsFlags = afwMath.stringToStatisticsProperty(self.config.stackingStatistic)
        # iteratively stack over small subregions
        for jbbox, bbox in enumerate(subBBoxes):
            if not jbbox % 50:
                self.log.info("Stacking subregion %i out of %i" % (jbbox+1, nbSubregions))
            allStars = None
            weights = None
            for stampId in selectDataList:
                dataId = {'visit': stampId["visit"], 'ccd': stampId["ccd"]}
                try:
                    readStars = butler.get("brightStarStamps_sub", dataId, bbox=bbox)
                    if self.config.doMagCut:
                        readStars = readStars.selectByMag(magMax=self.config.magLimit)
                    readWeights = 18. - np.array(readStars.getMagnitudes())
                    if allStars:
                        allStars.extend(readStars)
                        weights = np.hstack((weights, readWeights))
                    else:
                        allStars = readStars
                        weights = readWeights
                except bE.NoResults:
                    self.log.info(f"No BrightStarStamps found for dataId {dataId}; skipping it")
                    continue
            weights /= np.sum(weights)
            coaddSubregion = afwMath.statisticsStack(allStars.getMaskedImages(), statsFlags, statsControl,
                                                     wvector=weights)
            extPsf.assign(coaddSubregion, bbox)
        extPsf.writeFits(self.config.modelFilename)
