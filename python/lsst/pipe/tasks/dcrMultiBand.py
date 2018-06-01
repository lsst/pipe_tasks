#!/usr/bin/env python
# This file is part of pipe_tasks.
#
# LSST Data Management System
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
# See COPYRIGHT file at the top of the source tree.
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
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#

from lsst.coadd.utils.coaddDataIdContainer import ExistingCoaddDataIdContainer
from lsst.pipe.base import ArgumentParser
from lsst.pipe.tasks.multiBand import MergeDetectionsTask
from lsst.pipe.tasks.multiBand import DetectCoaddSourcesTask
from lsst.pipe.tasks.multiBand import MergeSourcesRunner
from lsst.pipe.tasks.multiBand import MeasureMergedCoaddSourcesTask
from lsst.pipe.tasks.multiBand import MergeMeasurementsTask

__all__ = ["DetectDcrCoaddSources", "MergeDcrDetectionsTask",
           "MeasureMergedDcrCoaddSourcesTask", "MergeDcrMeasurementsTask"]

"""Measure sources and their sub-filter spectrum from a DCR model."""


class MergeDcrSourcesRunner(MergeSourcesRunner):
    """Task runner for the MergeSourcesTask.

    Required because the run method requires a list of dataRefs
    rather than a single dataRef.
    """

    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        """Provide a list of patch and filter references for each tract.

        The filter references within the list will have different subfilters.

        Parameters
        ----------
        parsedCmd : `dict`
            The parsed command
        **kwargs
            Key word arguments (unused)

        Returns
        -------
        list of `patchRef`
            List of all matching data references for a given patch.

        Raises
        ------
        RuntimeError
            if multiple references are provided for the same combination of
            tract, patch, filter, and subfilter
        """
        refList = {}  # Will index this as refList[tract][patch][filter][subfilter] = ref
        for ref in parsedCmd.id.refList:
            tract = ref.dataId["tract"]
            patch = ref.dataId["patch"]
            filter = ref.dataId["filter"]
            subfilter = ref.dataId["subfilter"]
            if tract not in refList:
                refList[tract] = {}
            if patch not in refList[tract]:
                refList[tract][patch] = {}
            if filter not in refList[tract][patch]:
                refList[tract][patch][filter] = {}
            if subfilter in refList[tract][patch][filter]:
                raise RuntimeError("Multiple versions of %s" % (ref.dataId,))
            refList[tract][patch][filter][subfilter] = ref
        return [(list(f.values()), kwargs)
                for t in list(refList.values())
                for p in list(t.values())
                for f in list(p.values())]


class DetectDcrCoaddSources(DetectCoaddSourcesTask):
    """Detect sources on a DCR coadd."""

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(name="--id",
                               datasetType="dcrCoadd",
                               help="data ID, e.g. --id tract=12345 patch=1,2 filter=g, subfilter=0",
                               ContainerClass=ExistingCoaddDataIdContainer)
        return parser


class MergeDcrDetectionsTask(MergeDetectionsTask):
    """Merge dcrCoadd detections from multiple subfilters."""

    RunnerClass = MergeDcrSourcesRunner

    @classmethod
    def _makeArgumentParser(cls):
        """Create a suitable ArgumentParser.

        We will use the ArgumentParser to get a provide a list of data
        references for patches; the RunnerClass will sort them into lists
        of data references for the same patch
        """
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(name="--id",
                               datasetType="dcrCoadd_" + cls.inputDataset,
                               ContainerClass=ExistingCoaddDataIdContainer,
                               help="data ID, e.g. --id tract=12345 patch=1,2 filter=g, subfilter=0^1^2")
        return parser


class MeasureMergedDcrCoaddSourcesTask(MeasureMergedCoaddSourcesTask):
    """Deblend sources from master catalog in each coadd seperately and measure."""

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(name="--id",
                               datasetType="dcrCoadd_calexp",
                               help="data ID, e.g. --id tract=12345 patch=1,2 filter=g, subfilter=0",
                               ContainerClass=ExistingCoaddDataIdContainer)
        return parser


class MergeDcrMeasurementsTask(MergeMeasurementsTask):
    """Merge measurements from multiple subfilters."""

    RunnerClass = MergeDcrSourcesRunner

    @classmethod
    def _makeArgumentParser(cls):
        """Create a suitable ArgumentParser.

        We will use the ArgumentParser to get a provide a list of data
        references for patches; the RunnerClass will sort them into lists
        of data references for the same patch
        """
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(name="--id",
                               datasetType="dcrCoadd_" + cls.inputDataset,
                               ContainerClass=ExistingCoaddDataIdContainer,
                               help="data ID, e.g. --id tract=12345 patch=1,2 filter=g, subfilter=0^1^2")
        return parser
