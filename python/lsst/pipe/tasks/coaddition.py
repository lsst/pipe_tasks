from __future__ import absolute_import, division, print_function
#
# LSST Data Management System
# Copyright 2008-2016 AURA/LSST.
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
# see <https://www.lsstcorp.org/LegalNotices/>.
#
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.tasks.assembleCoadd import SafeClipAssembleCoaddTask, AssembleCoaddTask, \
    AssembleCoaddDataIdContainer, RobustAssembleCoaddTask
from .coaddBase import CoaddBaseTask, SelectDataIdContainer, WarpType

__all__ = ["CoadditionTask"]


class CoadditionConfig(CoaddBaseTask.ConfigClass):
    assembleCoaddDirect = pexConfig.ConfigurableField(
        target=AssembleCoaddTask,
        doc="Task to assemble direct Coadd",
    )
    assembleCoaddPsfMatched = pexConfig.ConfigurableField(
        target=AssembleCoaddTask,
        doc="Task to assemble psfMatched Coadd",
    )
    doMatchBackgrounds = pexConfig.Field(
        doc="Match backgrounds of coadd temp exposures before coadding them? "
        "If False, the coadd temp expsosures must already have been background subtracted or matched",
        dtype=bool,
        default=True,
    )
    autoReference = pexConfig.Field(
        doc="Automatically select the coadd temp exposure to use as a reference for background matching? "
        "Ignored if doMatchBackgrounds false. "
        "If False you must specify the reference temp exposure as the data Id",
        dtype=bool,
        default=True,
    )

    def setDefaults(self):
        self.assembleCoaddDirect.warpName = 'direct'
        self.assembleCoaddPsfMatched.warpName = 'psfMatched'


class CoadditionTask(CoaddBaseTask):
    """!Task to do all coaddition needed for a DRP
    """
    ConfigClass = CoadditionConfig
    _DefaultName = "coadditionTask"

    def __init__(self, *args, **kwargs):
        """!
        \brief Initialize the task. Create the \ref InterpImageTask "interpImage",
        \ref MatchBackgroundsTask "matchBackgrounds", & \ref ScaleZeroPointTask "scaleZeroPoint" subtasks.
        """
        CoaddBaseTask.__init__(self, *args, **kwargs)
        self.makeSubtask("assembleCoaddDirect")
        self.makeSubtask("assembleCoaddPsfMatched")

    @pipeBase.timeMethod
    def run(self, dataRef, selectDataList=[]):
        """!
        """
        warpTypeList = self.getWarpTypeList()
        taskDict = {WarpType.DIRECT: self.assembleCoaddDirect,
                    WarpType.PSF_MATCHED: self.assembleCoaddPsfMatched}

        for warpType in warpTypeList:
            taskDict[warpType].run(dataRef, selectDataList)


class CoadditionTask(CoaddBaseTask):
    """!Task to do all coaddition needed for a DRP
    """
    ConfigClass = CoadditionConfig
    _DefaultName = "coadditionTask"

    def __init__(self, *args, **kwargs):
        """!
        \brief Initialize the task. Create the \ref InterpImageTask "interpImage",
        \ref MatchBackgroundsTask "matchBackgrounds", & \ref ScaleZeroPointTask "scaleZeroPoint" subtasks.
        """
        CoaddBaseTask.__init__(self, *args, **kwargs)
        self.makeSubtask("assembleCoaddDirect")
        self.makeSubtask("assembleCoaddPsfMatched")

    @pipeBase.timeMethod
    def run(self, dataRef, selectDataList=[]):
        """!
        """
        warpTypeList = self.getWarpTypeList()
        taskDict = {WarpType.DIRECT: self.assembleCoaddDirect,
                    WarpType.PSF_MATCHED: self.assembleCoaddPsfMatched}

        for warpType in warpTypeList:
            taskDict[warpType].run(dataRef, selectDataList)


class RobustCoadditionConfig(CoaddBaseTask.ConfigClass):
    assembleCoaddDirect = pexConfig.ConfigurableField(
        target=RobustAssembleCoaddTask,
        doc="Task to assemble direct Coadd",
    )
    assembleCoaddPsfMatched = pexConfig.ConfigurableField(
        target=AssembleCoaddTask,
        doc="Task to assemble psfMatched Coadd",
    )
    doMatchBackgrounds = pexConfig.Field(
        doc="Match backgrounds of coadd temp exposures before coadding them? "
        "If False, the coadd temp expsosures must already have been background subtracted or matched",
        dtype=bool,
        default=True,
    )
    autoReference = pexConfig.Field(
        doc="Automatically select the coadd temp exposure to use as a reference for background matching? "
        "Ignored if doMatchBackgrounds false. "
        "If False you must specify the reference temp exposure as the data Id",
        dtype=bool,
        default=True,
    )

    def setDefaults(self):
        self.assembleCoaddDirect.warpName = 'direct'
        self.assembleCoaddPsfMatched.warpName = 'psfMatched'
        self.assembleCoaddPsfMatched.doClip = True


    # @classmethod
    # def _makeArgumentParser(cls):
    #     """!
    #     \brief Create an argument parser
    #     """
    #     parser = pipeBase.ArgumentParser(name=cls._DefaultName)
    #     parser.add_id_argument("--id", cls.ConfigClass().coaddName + "Coadd_directWarp",
    #                            help="data ID, e.g. --id tract=12345 patch=1,2",
    #                            ContainerClass=AssembleCoaddDataIdContainer)
    #     parser.add_id_argument("--selectId", "calexp", help="data ID, e.g. --selectId visit=6789 ccd=0..9",
    #                            ContainerClass=SelectDataIdContainer)
    #     return parser

class RobustCoadditionTask(CoaddBaseTask):
    """!Task to do all coaddition needed for a DRP
    """
    ConfigClass = CoadditionConfig
    _DefaultName = "coadditionTask"

    def __init__(self, *args, **kwargs):
        """!
        \brief Initialize the task. Create the \ref InterpImageTask "interpImage",
        \ref MatchBackgroundsTask "matchBackgrounds", & \ref ScaleZeroPointTask "scaleZeroPoint" subtasks.
        """
        CoaddBaseTask.__init__(self, *args, **kwargs)
        self.makeSubtask("assembleCoaddPsfMatched")
        self.makeSubtask("assembleCoaddDirect")


    @pipeBase.timeMethod
    def run(self, dataRef, selectDataList=[]):
        """!
        """
        clippedCoadd = self.assembleCoaddPsfMatched.run(dataRef, selectDataList).coaddExp
        self.assembleCoaddDirect.run()

        warpTypeList = self.getWarpTypeList()
        taskDict = {WarpType.DIRECT: self.assembleCoaddDirect,
                    WarpType.PSF_MATCHED: self.assembleCoaddPsfMatched}

        for warpType in warpTypeList:
            taskDict[warpType].run(dataRef, selectDataList)



