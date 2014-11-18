#
# LSST Data Management System
# Copyright 2008-2014 LSST Corporation.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

import lsst.pex.config
import lsst.pipe.base
import lsst.afw.image

class FakeSourcesConfig(lsst.pex.config.Config):
    maskPlaneName = lsst.pex.config.Field(
        dtype=str, default="FAKE",
        doc="Mask plane to set on pixels affected by fakes.  Will be added if not already present."
        )

class FakeSourcesTask(lsst.pipe.base.Task):
    """An abstract base class for subtasks that inject fake sources into images to test completeness and
    other aspects of the processing.

    This class simply adds a mask plane that subclasses should use to mark pixels that have been touched.

    To disable the injection of fake sources, use DummyFakeSourcesTask.
    """

    ConfigClass = FakeSourcesConfig

    def __init__(self, **kwargs):
        """Initialize the Task.

        Subclasses that define their own __init__ should simply forward all arguments to the base
        class constructor.  They can then assume self.config is an instance of their ConfigClass.

        If an external catalog is used to add sources consistently to multiple overlapping images,
        that catalog should generally be loaded and attached to self here, so it can be used
        multiple times by the run() method.
        """
        lsst.pipe.base.Task.__init__(self, **kwargs)
        lsst.afw.image.MaskU.addMaskPlane(self.config.maskPlaneName)
        self.bitmask = lsst.afw.image.MaskU.getPlaneBitMask(self.config.maskPlaneName)

    def run(self, exposure, background):
        """Add fake sources to the given Exposure, making use of the given BackgroundList if desired.

        If pixels in the Exposure are replaced, not added to, extra care should be taken with the background,
        mask, and variance planes.  The Exposure as given is background-subtracted (using the supplied
        background model) and should be returned in the same state.
        """
        raise NotImplementedError("FakeSourcesTask is abstract; did you want DummyFakeSourcesTask?")


class DummyFakeSourcesConfig(lsst.pex.config.Config):
    pass

class DummyFakeSourcesTask(lsst.pipe.base.Task):
    """A stand-in for FakeSourcesTask that doesn't do anything, to be used as the default (to disable
    fake injection) anywhere FakeSourcesTask could be used.
    """

    ConfigClass = DummyFakeSourcesConfig

    def __init__(self, **kwargs):
        lsst.pipe.base.Task.__init__(self, **kwargs)

    def run(self, exposure, background):
        pass
