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

from lsst.pipe.base.argumentParser import ArgumentParser

def runTask(TaskClass, ArgumentParserClass=ArgumentParser, argList=['butler', 'idList'], argMap=None):
    """Run a Task after parsing the command-line arguments.
    This is intended to be used as a common front-end for scripts.

    Arguments to the TaskClass.run() method are pulled from the
    argument parser.  These may be specified with either of both
    of a list (for no translation required) or a dict (for when
    translation is required, e.g., {'TaskArg': 'ParserVar'})/

    @param TaskClass             Class of the particular Task to be run
    @param ArgumentParserClass   Class of ArgumentParser for command-line parsing
    @param argList               List of ArgumentParser instance variables to pass to Task
    @param argMap                Mapping from ArgumentParser instance variables to Task inputs
    @return Results of running the Task
    """
    
    parser = ArgumentParser()
    cmd = parser.parse_args(config=TaskClass.ConfigClass())
    task = TaskClass(cmd.config)

    kwargs = dict()
    if argList is not None:
        kwargs.update([(arg, getattr(cmd, arg)) for arg in argList])
    if argMap is not None:
        kwargs.update([(argTarget, getattr(cmd, argSource)) for argTarget,argSource in argMap])

    return task.run(**kwargs)

