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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the LSST License Statement and 
# the GNU General Public License along with this program.  If not, 
# see <http://www.lsstcorp.org/LegalNotices/>.
#

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

class CrosstalkConfig(pexConfig.Config):
    """An empty placeholder/base class for crosstalk correction configuration.
    
    See CrosstalkTask for more information.
    """
    pass

class CrosstalkTask(pipeBase.Task):
    """An empty placeholder/base class for crosstalk correction.

    This class does nothing, but its existence allows us to have a ConfigurableField
    slot for camera-specific crosstalk correction subtasks in RepairTask.  Maybe someday
    we'll have camera-generic crosstalk code, which could then go here instead.  A useful
    starting point may be the Subaru-specific crosstalk task in obs_subaru.
    """

    ConfigClass = CrosstalkConfig

    def run(self):
        pass
