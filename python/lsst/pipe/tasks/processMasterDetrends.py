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
import sys
import traceback
import numpy
import argparse
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from .detrends import DetrendTask
from lsst.pipe.base.argumentParser import ArgumentParser

class MakeRefListDictTask(pipeBase.Task):
    ConfigClass = pexConfig.Config
    
    def run(self, dataRefList, masterName, calibType):
        retDict = {}
        mybutler = dataRefList[0].butlerSubset.butler
        calibKeys = mybutler.getKeys(datasetType=masterName)
        for ref in dataRefList:
            groupbykey = self.getDictKey(calibKeys, ref.dataId)
            if retDict.has_key(groupbykey):
                retDict[groupbykey].append(ref)
            else:
                retDict[groupbykey] = [ref,]
        return retDict

    def getDictKey(self, keys, dataId):
        return tuple([el for el in [dataId[key] for key in keys]])

class ProcessMasterDetrendsConfig(pexConfig.Config):
    """Config for ProcessMasterDetrendsTask"""
    detrend = pexConfig.ConfigurableField(target=DetrendTask, doc="Task containing detrending tasks",)
    getRefDict = pexConfig.ConfigurableField(target=MakeRefListDictTask, doc="Task for grouping data ids to be combined for master calibration production",)
    doNormalize = pexConfig.Field(dtype=bool, default=False, doc="Normalize master calibration frames over the entire mosaic",)

class ProcessMasterDetrendsTask(pipeBase.CmdLineTask):
    """
    """
    ConfigClass = ProcessMasterDetrendsConfig
    _DefaultName = "processMasterDetrends"

    def __init__(self, **kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)
        self.makeSubtask("detrend")
        self.makeSubtask("getRefDict")

    @pipeBase.timeMethod
    def run(self, dataRefList, calibType):
        """Process an Image
        
        @param dataRef: data reference that corresponds to the input image
        @param inputExposure:  exposure to process

        @return pipe_base Struct containing these fields:
        """

        # initialize outputs
        isrKey = "%sIsr"%(calibType)
        masterName = "%sOut"%(calibType)
        groupDict = self.getRefDict.run(dataRefList, masterName, calibType)
        resultDict = {}
        for key in groupDict.keys(): 
            resultDict[key] = []
            for ref in groupDict[key]:
                result = self.detrend.process.run(calibType, isrKey, ref)
                resultDict[key].append((ref, result.background))
                ref.put(result.exposure, "postISR")
            if not self.config.doNormalize:
                #Safe to combine this group right now
                refList = [el[0] for el in resultDict[key]]
                combined = self.detrend.combine.run(refList)
        else:
            backgrounds = []
            refList = []
            #I don't understand Paul's scaling task well enough to use it...
            #This seems like it should be about as good.
            refBackground = []
            for k in resultDict.keys():
                refList.append([el[0] for el in resultDict[k]])
                backgrounds.append([el[1] for el in resultDict[k]])
                refBackground.append(backgrounds[-1][0])
            refScale = numpy.max(refBackground)
            #scaleRes = self.detrend.scale.run(numpy.array(backgrounds, dtype=float))
            #for ref, scale in zip(refList, scaleRes.components):
            for ref, scale in zip(refList, backgrounds):
                #combined = self.detrend.combine.run(ref, expScales=scaleRes.exposures, finalScale=scale)
                combined = self.detrend.simpleCombine.run(ref, scale, refScale)
                mybutler.put(combined, masterName, dataId=self.getDataId(calibKeys, ref[0].dataId))
    
        return pipeBase.Struct(
        )


    def getDataId(self, keys, dataId):
        ret = dict()
        for k in keys:
            ret[k] = dataId[k]
        return ret

    @classmethod
    def parseAndRun(cls, args=None, config=None, log=None):
        """Parse an argument list and run the command

        @param args: list of command-line arguments; if None use sys.argv
        @param config: config for task (instance of pex_config Config); if None use cls.ConfigClass()
        @param log: log (instance of pex_logging Log); if None use the default log

        @return a Struct containing:
        - argumentParser: the argument parser
        - parsedCmd: the parsed command returned by argumentParser.parse_args
        - task: the instantiated task
        The return values are primarily for testing and debugging
        """
        argumentParser = cls._makeArgumentParser()
        if config is None:
            config = cls.ConfigClass()
        parsedCmd = argumentParser.parse_args(config=config, args=args, log=log, override=cls.applyOverrides)
        cls.runParsedCmd(parsedCmd)
        return Struct(
            argumentParser = argumentParser,
            parsedCmd = parsedCmd,
            )

    def runDataRef(self, dataRef, calibType, doraise=False):
        """Execute the task on the data reference

        If you want to override this method with different inputs, you're
        also going to want to override getRunInfo and also have that provide
        a different 'func' that calls into this method.  That's three
        different places to override this one method: one for the
        functionality (runDataRef), one to provide different input
        (getRunInfo) and an (unfortunate) extra layer of indirection
        required for multiprocessing support (runTask).

        @param dataRef   Data reference to process
        @param doraise   Allow exceptions to float up?
        """
        if doraise:
            self.run(dataRef, calibType)
        else:
            try:
                self.run(dataRef, calibType)
            except Exception, e:
                self.log.fatal("Failed on dataId=%s: %s" % (dataRef.dataId, e))
                if not isinstance(e, pipeBase.TaskError):
                    traceback.print_exc(file=sys.stderr)

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser

        Subclasses may wish to override, e.g. to change the dataset type or data ref level
        """
        argParser = ArgumentParser(name=cls._DefaultName, usage="%(prog)s input calibType [options]")
        argParser.add_argument("calibType", type=str, action=CalibTypeValueAction, help="Name of the type of master calibration to create (i.e. flat, bias, fringe, ...)")
        return argParser

    @classmethod
    def getRunInfo(cls, parsedCmd):
        """Construct information necessary to run the task from the command-line arguments

        For multiprocessing to work, the 'func' returned must be picklable
        (i.e., typically a named function rather than anonymous function or
        method).  Thus, an extra level of indirection is typically required,
        so that the 'func' will create the Task from the 'inputs', and run.
        Because the 'func' is executed using 'map', it should not return any
        large data structures (which will require transmission between
        processes, and long-term memory storage).

        @param parsedCmd   Results of the argument parser
        @return Struct(func: Function to receive 'inputs';
                       inputs: List of Structs to be passed to the 'func')
        """
        log = parsedCmd.log if not useMultiProcessing(parsedCmd) else None# XXX pexLogging is not yet picklable
        inputs = [pipeBase.Struct(cls=cls, config=parsedCmd.config, log=log, doraise=parsedCmd.doraise, calibType=parsedCmd.calibType, dataRef=parsedCmd.dataRefList)]
        return pipeBase.Struct(func=runTask, inputs=inputs)

def useMultiProcessing(args):
    """Determine whether we're using multiprocessing,
    based on the parsed command-line arguments."""
    return hasattr(args, 'processes') and args.processes > 1

def runTask(args):
    """Run task, by forwarding to CmdLineTask._runDataRef.

    This forwarding is necessary because multiprocessing requires
    that the function used is picklable, which means it must be a
    named function, rather than an anonymous function (lambda) or
    method.
    """
    task = args.cls(name = args.cls._DefaultName, config=args.config, log=args.log)
    ###HOWTO write config of data ref list
    #task.writeConfig(args.dataRef)
    task.runDataRef(args.dataRef, args.calibType, doraise=args.doraise)
    ###HOWTO write metadata of data ref list
    #task.writeMetadata(args.dataRef)

class CalibTypeValueAction(argparse.Action):
    """argparse action to set the active calib type in the registry based on the command line parameter
    """
    def __call__(self, parser, namespace, values, option_string=None):
        """Load one or more files of config overrides
        """
        try:
            namespace.config.detrend.process.isr.name = "%sIsr" % (values)
            namespace.calibType = values
        except Exception, e:
            parser.error("Cannot apply calibration type to the config: %s"%(e))
