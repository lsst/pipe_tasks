#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2014 LSST Corporation.
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
import os
import sys

from eups import productDir
from lsst.afw.image import MaskedImageF
from lsst.pipe.tasks.exampleTask import ExampleSimpleStatsTask, ExampleSigmaClippedStatsTask

if len(sys.argv) < 2:
    afwDataDir = productDir("afwdata")
    if not afwDataDir:
        print """Usage: exampleSigmaClippedStatsTask.py [fitsImage]
fitsImage is a path to a masked image or exposure
To use the default fitsImage you must have afwdata setup
"""
        sys.exit(1)
    maskedImagePath = os.path.join(afwDataDir, "data", "med.fits")
else:
    maskedImagePath = sys.argv[1]
print "computing statistics on %r\n" % (maskedImagePath,)

maskedImage = MaskedImageF(maskedImagePath)

print "running ExampleSimpleStatsTask"
config1 = ExampleSimpleStatsTask.ConfigClass()
task1 = ExampleSimpleStatsTask(config=config1)
res1 = task1.run(maskedImage)
print "result  =", res1
print

print "running ExampleSigmaClippedStatsTask"
config2 = ExampleSigmaClippedStatsTask.ConfigClass()
task2 = ExampleSigmaClippedStatsTask(config=config2)
res2 = task2.run(maskedImage)
print "result  =", res2