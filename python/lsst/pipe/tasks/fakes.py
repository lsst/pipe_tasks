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

__all__ = ["BaseFakeSourcesConfig", "BaseFakeSourcesTask"]

import abc

from deprecated.sphinx import deprecated

import lsst.pex.config
import lsst.pipe.base
import lsst.afw.image


class BaseFakeSourcesConfig(lsst.pex.config.Config):
    maskPlaneName = lsst.pex.config.Field(
        dtype=str, default="FAKE",
        doc="Mask plane to set on pixels affected by fakes.  Will be added if not already present."
    )


@deprecated(reason=("The fakes code has been deprecated in favor of ProcessCcdWithFakes. "
                    "This will be removed after v24."),
            category=FutureWarning,
            version="24.0")
class BaseFakeSourcesTask(lsst.pipe.base.Task, metaclass=abc.ABCMeta):
    """An abstract base class for subtasks that inject fake sources into images to test completeness and
    other aspects of the processing.

    This class simply adds a mask plane that subclasses should use to mark pixels that have been touched.

    This is an abstract base class (abc) and is not intended to be directly used. To create a fake sources
    injector, create a child class and re-implement the required methods.
    """

    ConfigClass = BaseFakeSourcesConfig
    _DefaultName = "baseFakeSources"

    def __init__(self, **kwargs):
        """Initialize the Task.

        Subclasses that define their own __init__ should simply forward all arguments to the base
        class constructor.  They can then assume self.config is an instance of their ConfigClass.

        If an external catalog is used to add sources consistently to multiple overlapping images,
        that catalog should generally be loaded and attached to self here, so it can be used
        multiple times by the run() method.
        """
        lsst.pipe.base.Task.__init__(self, **kwargs)
        lsst.afw.image.Mask[lsst.afw.image.MaskPixel].addMaskPlane(self.config.maskPlaneName)
        self.bitmask = lsst.afw.image.Mask[lsst.afw.image.MaskPixel]\
            .getPlaneBitMask(self.config.maskPlaneName)

    @abc.abstractmethod
    def run(self, exposure, background):
        """Add fake sources to the given Exposure, making use of the given BackgroundList if desired.

        If pixels in the Exposure are replaced, not added to, extra care should be taken with the background,
        mask, and variance planes.  The Exposure as given is background-subtracted (using the supplied
        background model) and should be returned in the same state.
        """
        raise NotImplementedError("FakeSourcesTask is abstract, create a child class to use this method")
