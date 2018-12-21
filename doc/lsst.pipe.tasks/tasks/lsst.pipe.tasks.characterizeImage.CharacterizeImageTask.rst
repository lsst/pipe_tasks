.. lsst-task-topic:: lsst.pipe.tasks.characterizeImage.CharacterizeImageTask

#####################
CharacterizeImageTask
#####################

.. _lsst.pipe.tasks.characterizeImage.CharacterizeImageTask-summary:

Processing summary
==================

Given an exposure (typically, e.g., as output by IsrTask):
    (1) Iterate over the following config.psfIteration times, or once if
    config.doMeasurePsf is False:
        - detect and measure bright sources
        - do an initial repair of cosmic rays (no interpolation yet)
        - measure and subtract background
        - do an initial PSF measurement estimate
    (2) Update or set final PSF
    (3) Do final cosmic ray repair, including interpolation
    (4) Perform final measurement with final PSF, including measuring and
        applying aperture correction, if applicable

.. _lsst.pipe.tasks.characterizeImage.CharacterizeImageTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.pipe.tasks.characterizeImage.CharacterizeImageTask

.. _lsst.pipe.tasks.characterizeImage.CharacterizeImageTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.pipe.tasks.characterizeImage.CharacterizeImageTask

.. _lsst.pipe.tasks.characterizeImage.CharacterizeImageTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.pipe.tasks.characterizeImage.CharacterizeImageTask

.. _lsst.pipe.tasks.characterizeImage.CharacterizeImageTask-debug:

Debugging
=========

The lsst.pipe.base.cmdLineTask.CmdLineTask command line task interface supports a flag
`--debug` to import `debug.py` from your `$PYTHONPATH`; see baseDebug for more about `debug.py`.

CharacterizeImageTask has a debug dictionary with the following keys

frame:
    - `int`: if specified, the frame of first debug image displayed (defaults to 1)

repair_iter:
    - `bool`; if True display image after each repair in the measure PSF loop

background_iter:
    - `bool`; if True display image after each background subtraction in the measure PSF loop

measure_iter:
    - `bool`; if True display image and sources at the end of each iteration of the measure PSF loop

See lsst.meas.astrom.displayAstrometry for the meaning of the various symbols.

psf:
    - `bool`; if True display image and sources after PSF is measured;

this will be identical to the final image displayed by measure_iter if measure_iter is true

repair:
    - `bool`; if True display image and sources after final repair

measure:
    - `bool`; if True display image and sources after final measurement

For example, put something like:

.. code-block:: none

    import lsstDebug
    def DebugInfo(name):
        di = lsstDebug.getInfo(name)  # N.b. lsstDebug.Info(name) would call us recursively
        if name == "lsst.pipe.tasks.characterizeImage":
            di.display = dict(
                repair = True,
            )

        return di

    lsstDebug.Info = DebugInfo

into your `debug.py` file and run `calibrateTask.py` with the `--debug` flag.
Some subtasks may have their own debug variables; see individual Task documentation.