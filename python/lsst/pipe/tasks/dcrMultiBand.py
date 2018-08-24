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
from lsst.pipe.tasks.multiBand import DeblendCoaddSourcesTask
from lsst.pipe.tasks.multiBand import DetectCoaddSourcesTask
from lsst.pipe.tasks.multiBand import MergeSourcesRunner
from lsst.pipe.tasks.multiBand import MeasureMergedCoaddSourcesTask
from lsst.pipe.tasks.multiBand import MergeDetectionsTask
from lsst.pipe.tasks.multiBand import MergeMeasurementsTask

__all__ = ["DetectDcrCoaddSources", "DeblendDcrCoaddSourcesTask", "MergeDcrDetectionsTask",
           "MeasureMergedDcrCoaddSourcesTask", "MergeDcrMeasurementsTask"]

"""Measure sources and their sub-filter spectrum from a DCR model."""


class MergeDcrSourcesRunner(MergeSourcesRunner):
    """Task runner for the MergeSourcesTask.

    Required because the run method requires a list of dataRefs
    rather than a single dataRef.
    """

    @staticmethod
    def buildRefDict(parsedCmd):
        """Build a hierarchical dictionary of patch references

        The filter references within the list will have different subfilters.

        Parameters
        ----------
        parsedCmd : `dict`
            The parsed command
        **kwargs
            Key word arguments (unused)

        Returns
        -------
        refDict: dict
            A reference dictionary of the form {patch: {tract: {filter: {subfilter: dataRef}}}}

        Raises
        ------
        RuntimeError
            Thrown when multiple references are provided for the same combination of
            tract, patch, filter, and subfilter
        """
        refDict = {}  # Will index this as refDict[tract][patch][filter][subfilter] = ref
        for ref in parsedCmd.id.refList:
            tract = ref.dataId["tract"]
            patch = ref.dataId["patch"]
            filter = ref.dataId["filter"]
            subfilter = ref.dataId["subfilter"]
            if tract not in refDict:
                refDict[tract] = {}
            if patch not in refDict[tract]:
                refDict[tract][patch] = {}
            if filter not in refDict[tract][patch]:
                refDict[tract][patch][filter] = {}
            if subfilter in refDict[tract][patch][filter]:
                raise RuntimeError("Multiple versions of %s" % (ref.dataId,))
            refDict[tract][patch][filter][subfilter] = ref
        return refDict

    @classmethod
    def getTargetList(cls, parsedCmd, **kwargs):
        """Provide a list of patch references for each patch, tract, filter combo.

        Parameters
        ----------
        parsedCmd:
            The parsed command
        kwargs:
            Keyword arguments passed to the task

        Returns
        -------
        targetList: list
            List of tuples, where each tuple is a (dataRef, kwargs) pair.
        """
        refDict = cls.buildRefDict(parsedCmd)
        return [(list(f.values()), kwargs)
                for t in list(refDict.values())
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


class DeblendDcrCoaddSourcesRunner(MergeDcrSourcesRunner):
    """Task runner for the `MergeSourcesTask`.

    Required because the run method requires a list of
    dataRefs rather than a single dataRef.
    """

    @classmethod
    def getTargetList(cls, parsedCmd, **kwargs):
        """Provide a list of patch references for each patch, tract, filter combo.

        Parameters
        ----------
        parsedCmd:
            The parsed command
        kwargs:
            Keyword arguments passed to the task

        Returns
        -------
        targetList: list
            List of tuples, where each tuple is a (dataRef, kwargs) pair.
        """
        refDict = cls.buildRefDict(parsedCmd)
        kwargs["psfCache"] = parsedCmd.psfCache
        return [(list(f.values()), kwargs)
                for t in list(refDict.values())
                for p in list(t.values())
                for f in list(p.values())]


class DeblendDcrCoaddSourcesTask(DeblendCoaddSourcesTask):
    """Deblend the sources in a merged catalog."""

    RunnerClass = DeblendDcrCoaddSourcesRunner

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(name="--id",
                               datasetType="dcrCoadd_calexp",
                               help="data ID, e.g. --id tract=12345 patch=1,2 filter=g, subfilter=0",
                               ContainerClass=ExistingCoaddDataIdContainer)
        parser.add_argument("--psfCache", type=int, default=100, help="Size of CoaddPsf cache")
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
        parser.add_argument("--psfCache", type=int, default=100, help="Size of CoaddPsf cache")
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
