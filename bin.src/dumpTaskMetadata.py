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
"""Select images and report which tracts and patches they are in
"""
from __future__ import print_function

import lsst.daf.base as dafBase
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

__all__ = ["DumpTaskMetadataTask", ]


class DumpTaskMetadataConfig(pexConfig.Config):
    """Config for DumpTaskMetadataTask
    """
    showTimingData = pexConfig.Field(dtype=bool, default=True,
                                     doc="Show timing data?")


class DumpTaskMetadataTask(pipeBase.CmdLineTask):
    """Report which tracts and patches are needed for coaddition
    """
    ConfigClass = DumpTaskMetadataConfig
    _DefaultName = "DumpTaskMetadata"

    @pipeBase.timeMethod
    def run(self, dataRef):
        """Report task metadata
        """
        print("%s for dataId=%s" % (dataRef.butlerSubset.datasetType, dataRef.dataId))
        TimerSuffixList = ("CpuTime", "InBlock", "MajFlt", "MaxRss",
                           "MinFlt", "NIvCsw", "NVCsw", "OuBlock", "STime", "UTime", "Utc")

        taskMetadata = dataRef.get()
        nameList = list(taskMetadata.names(False))  # hierarchical names
        nameList.sort()
        for name in nameList:
            if not self.config.showTimingData:
                if any(name.endswith(timerSuffix) for timerSuffix in TimerSuffixList):
                    continue

            data = taskMetadata.get(name)
            if isinstance(data, dafBase.PropertySet):
                # this same data will come up again in nameList
                continue
            print("%s    %s" % (name, data))

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser
        """
        return pipeBase.InputOnlyArgumentParser(name=cls._DefaultName,
                                                datasetType=pipeBase.DatasetArgument(
                                                    help="dataset type for task metadata"))

    def _getConfigName(self):
        """Don't persist config, so return None
        """
        return None

    def _getMetadataName(self):
        """Don't persist metadata, so return None
        """
        return None


if __name__ == "__main__":
    DumpTaskMetadataTask.parseAndRun()
