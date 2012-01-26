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
import lsst.pex.logging as pexLog
from lsst.pipe.tasks.coaddArgumentParser import CoaddArgumentParser
from lsst.pipe.tasks.outlierRejectedCoadd import OutlierRejectedCoaddTask

if __name__ == "__main__":
    TaskClass = OutlierRejectedCoaddTask
    algName = "outlierRejectedCoadd"
    pexLog.Trace.setVerbosity('lsst.coadd', 3)
    pexLog.Trace.setVerbosity('lsst.ip.diffim', 1)

    parser = CoaddArgumentParser()
    cmd = parser.parse_args(config=TaskClass.ConfigClass())
    task = TaskClass(cmd.config)
    taskRes = task.run(
        butler = cmd.butler,
        idList = cmd.idList,
        bbox = cmd.bbox,
        wcs = cmd.wcs,
        desFwhm = cmd.fwhm,
    )
    
    coaddExposure = taskRes.coaddExposure

    filterName = coaddExposure.getFilter().getName()
    if filterName == "_unknown_":
        filterStr = "unk"
    coaddBasePath = cmd.rerun
    coaddBaseName = "%s_%s_filter_%s_fwhm_%s" % (coaddBasePath, algName, filterName, cmd.fwhm)
    coaddPath = coaddBaseName + ".fits"
    print "Saving coadd as %s" % (coaddPath,)
    coaddExposure.writeFits(coaddPath)
