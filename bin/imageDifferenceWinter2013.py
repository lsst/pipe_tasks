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

# Workaround for a problem in the lsstimport module.  This module sets
# RTLD_GLOBAL for all LSST imports, which is necessary for RTTI, but
# causes scipy to fail when loading numpy.  Importing scipy beforehand
# avoids this problem.
try:
    import scipy.stats
except ImportError:
    pass
from lsst.pipe.tasks.imageDifference import Winter2013ImageDifferenceTask

Winter2013ImageDifferenceTask.parseAndRun()
