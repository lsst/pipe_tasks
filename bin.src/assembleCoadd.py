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
from __future__ import print_function
from lsst.pipe.tasks.assembleCoadd import AssembleCoaddTask, SafeClipAssembleCoaddTask
import sys
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--legacyCoadd', action='store_true', default=False)
parser.add_argument('-h', '--help', action='store_true', default=False)
result, extra = parser.parse_known_args(sys.argv)

legacy_message = '''
Meta Arguments:
  --legacyCoadd         An option to run with the original assembleCoadd task which will
                        not attempt to do any safe clipping.  If run, the config option
                        'config.measurement.plugins["base_PixelFlags"].masksFpAnywhere = [] 
                        must be set.
'''

if result.help:
    print(legacy_message)
    # As AssembleCoaddTask and SafeClipAssembleCoaddTask may have different help messages,
    # The appropriate one should be printed
    if result.legacyCoadd:
        AssembleCoaddTask.parseAndRun(args=['--help']+extra[1:])
    else:
        SafeClipAssembleCoaddTask.parseAndRun(args=['--help']+extra[1:])

# If no arguments are passed the default help message will be triggered in the argument parser.
# In order to ensure the  additional information is printed, zero argument lengths should trigger
# the message to be printed.
if len(extra) == 0:
    print(legacy_message)

if result.legacyCoadd:
    legacy_message = '''
    Remember, If you use the legacy adder you must set
    config.measurement.plugins['base_PixelFlags'].any = []
    so that pixel flags will not look for a non existent mask plane
    '''
    print(legacy_message)
    AssembleCoaddTask.parseAndRun(args=extra[1:])
else:
    SafeClipAssembleCoaddTask.parseAndRun(args=extra[1:])
