.. lsst-task-topic:: lsst.pipe.tasks.calibrate.CalibrateTask

#############
CalibrateTask
#############

.. _lsst.pipe.tasks.calibrate.CalibrateTask-summary:

Processing summary
==================

Given an exposure with a good PSF model and aperture correction map
(e.g. as provided by ``CharacterizeImageTask``), perform the following
operations:

- Run detection and measurement
- Run astrometry subtask to fit an improved WCS
- Run photoCal subtask to fit the exposure's photometric zero-point

Invoking the Task

If you want this task to unpersist inputs or persist outputs, then call
the `runDataRef` method (a wrapper around the `run` method).

If you already have the inputs unpersisted and do not want to persist the
output then it is more direct to call the `run` method:

Quantities set in exposure Metadata

Exposure metadata

.. code-block:: none

    MAGZERO_RMS  MAGZERO's RMS == sigma reported by photoCal task
    ERO_NOBJ Number of stars used == ngood reported by photoCal
    task
    COLORTERM1    (always 0.0)
    COLORTERM2    (always 0.0)
    COLORTERM3    (always 0.0)

.. _lsst.pipe.tasks.calibrate.CalibrateTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.pipe.tasks.calibrate.CalibrateTask

.. _lsst.pipe.tasks.calibrate.CalibrateTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.pipe.tasks.calibrate.CalibrateTask

.. _lsst.pipe.tasks.calibrate.CalibrateTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.pipe.tasks.calibrate.CalibrateTask

.. _lsst.pipe.tasks.calibrate.CalibrateTask-debug:

Debugging
=========

The `lsst.pipe.base.cmdLineTask.CmdLineTask` command line task
interface supports a flag
`--debug` to import `debug.py` from your `$PYTHONPATH`; see ``baseDebug``
for more about `debug.py`.

CalibrateTask has a debug dictionary containing one key:

.. code-block:: none

    calibrate
    frame (an int; <= 0 to not display) in which to display the exposure,
    sources and matches. See ``lsst.meas.astrom.displayAstrometry`` for
    the meaning of the various symbols.

For example, put something like:

.. code-block:: py

    import lsstDebug
    def DebugInfo(name):
        di = lsstDebug.getInfo(name)  # N.b. lsstDebug.Info(name) would
                                      # call us recursively
        if name == "lsst.pipe.tasks.calibrate":
            di.display = dict(
                calibrate = 1,
            )

        return di

    lsstDebug.Info = DebugInfo

into your `debug.py` file and run `calibrateTask.py` with the `--debug`
flag.

Some subtasks may have their own debug variables; see individual Task
documentation.
