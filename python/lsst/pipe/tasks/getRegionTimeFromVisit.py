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


__all__ = ["GetRegionTimeFromVisitTask", "GetRegionTimeFromVisitConfig"]


import lsst.pipe.base as pipeBase
from lsst.pipe.base.utils import RegionTimeInfo
from lsst.utils.timer import timeMethod


class GetRegionTimeFromVisitConnections(pipeBase.PipelineTaskConnections,
                                        dimensions={"instrument", "group", "detector"},
                                        defaultTemplates={"coaddName": "goodSeeing", "fakesType": ""}):

    dummy_visit = pipeBase.connectionTypes.Input(
        doc="Placeholder connection to provide visit-detector records and "
            "constrain data IDs to images we're processing.",
        name="initial_stars_footprints_detector",
        storageClass="SourceCatalog",
        dimensions=["instrument", "visit", "detector"],
    )
    dummy_exposure = pipeBase.connectionTypes.Output(
        doc="Placeholder connection to guarantee visit-exposure-group mapping. "
            "This output is never produced and need not be registered.",
        name="getRegionTimeFromVisit_dummy2",  # Unique because it gets registered anyway.
        storageClass="int",
        dimensions={"instrument", "exposure", "detector"},
        multiple=True,
    )
    output = pipeBase.connectionTypes.Output(
        doc="The region and time associated with this group's visit.",
        name="regionTimeInfo",
        storageClass="RegionTimeInfo",
        dimensions={"instrument", "group", "detector"},
    )


class GetRegionTimeFromVisitConfig(pipeBase.PipelineTaskConfig,
                                   pipelineConnections=GetRegionTimeFromVisitConnections):
    pass


class GetRegionTimeFromVisitTask(pipeBase.PipelineTask):
    """A converter that reads metadata from visit-dimension records and writes
    it to a Butler dataset.
    """
    _DefaultName = "getRegionTimeFromVisit"
    ConfigClass = GetRegionTimeFromVisitConfig

    @timeMethod
    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        """Convert the passed dataset refs to persistable metadata.
        """
        # Input datasetRefs guaranteed to be expanded.
        times = inputRefs.dummy_visit.dataId.records["visit"].timespan
        region = inputRefs.dummy_visit.dataId.records["visit_detector_region"].region
        outputs = pipeBase.Struct(
            output=RegionTimeInfo(region=region, timespan=times),
            dummy_exposure=None,
        )
        butlerQC.put(outputs, outputRefs)

    # All work is done in runQuantum
    run = None
