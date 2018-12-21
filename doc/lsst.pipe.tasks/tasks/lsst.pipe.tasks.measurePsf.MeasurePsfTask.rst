.. lsst-task-topic:: lsst.pipe.tasks.measurePsf.MeasurePsfTask

##############
MeasurePsfTask
##############

.. _lsst.pipe.tasks.measurePsf.MeasurePsfTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.pipe.tasks.measurePsf.MeasurePsfTask

.. _lsst.pipe.tasks.measurePsf.MeasurePsfTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.pipe.tasks.measurePsf.MeasurePsfTask

.. _lsst.pipe.tasks.measurePsf.MeasurePsfTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.pipe.tasks.measurePsf.MeasurePsfTask

.. _lsst.pipe.tasks.measurePsf.MeasurePsfTask-examples:

Examples
========

A complete example of using MeasurePsfTask is in measurePsfTask.py in the
examples directory, and can be run as, e.g.,

.. code-block:: bash

    examples/measurePsfTask.py --ds9

The example also runs SourceDetectionTask and SourceMeasurementTask.
Import the tasks (there are some other standard imports; read the file to see them all):

.. code-block:: python

    from lsst.meas.algorithms.detection import SourceDetectionTask
    from lsst.meas.base import SingleFrameMeasurementTask
    from lsst.pipe.tasks.measurePsf import MeasurePsfTask

We need to create the tasks before processing any data as the task constructor
can add an extra column to the schema, but first we need an almost-empty
Schema:

.. code-block:: python

    schema = afwTable.SourceTable.makeMinimalSchema()

We can now call the constructors for the tasks we need to find and characterize candidate
PSF stars:

.. code-block:: python

    config = SourceDetectionTask.ConfigClass()
    config.reEstimateBackground = False
    detectionTask = SourceDetectionTask(config=config, schema=schema)
    config = SingleFrameMeasurementTask.ConfigClass()
    # Use the minimum set of plugins required.
    config.plugins.names.clear()
    for plugin in ["base_SdssCentroid", "base_SdssShape", "base_CircularApertureFlux", "base_PixelFlags"]:
        config.plugins.names.add(plugin)
    config.plugins["base_CircularApertureFlux"].radii = [7.0]
    config.slots.psfFlux = "base_CircularApertureFlux_7_0" # Use of the PSF flux is hardcoded in secondMomentStarSelector
    measureTask = SingleFrameMeasurementTask(schema, config=config)

Note that we've chosen a minimal set of measurement plugins: we need the
outputs of ``base_SdssCentroid``, ``base_SdssShape``, and ``base_CircularApertureFlux``
as inputs to the PSF measurement algorithm, while ``base_PixelFlags`` identifies
and flags bad sources (e.g. with pixels too close to the edge) so they can be
removed later.

Now we can create and configure the task that we're interested in:

.. code-block:: python

    config = MeasurePsfTask.ConfigClass()
    psfDeterminer = config.psfDeterminer.apply()
    psfDeterminer.config.sizeCellX = 128
    psfDeterminer.config.sizeCellY = 128
    psfDeterminer.config.spatialOrder = 1
    psfDeterminer.config.nEigenComponents = 3
    measurePsfTask = MeasurePsfTask(config=config, schema=schema)

We're now ready to process the data (we could loop over multiple exposures/catalogues using the same
task objects).  First create the output table:

.. code-block:: python

    tab = afwTable.SourceTable.make(schema)

And process the image:

.. code-block:: python

    sources = detectionTask.run(tab, exposure, sigma=2).sources
    measureTask.measure(exposure, sources)
    result = measurePsfTask.run(exposure, sources)

We can then unpack and use the results:

.. code-block:: python

    psf = result.psf
    cellSet = result.cellSet

.. _lsst.pipe.tasks.measurePsf.MeasurePsfTask-debug:

Debugging
=========

The  ``lsst.pipe.base.cmdLineTask.CmdLineTask`` command line task interface supports a
flag -d to import debug.py from your PYTHONPATH; see baseDebug for more about debug.py files.

.. code-block:: none

    display
    If True, display debugging plots
    displayExposure
    display the Exposure + spatialCells
    displayPsfCandidates
    show mosaic of candidates
    showBadCandidates
    Include bad candidates
    displayPsfMosaic
    show mosaic of reconstructed PSF(xy)
    displayResiduals
    show residuals
    normalizeResiduals
    Normalise residuals by object amplitude


Additionally you can enable any debug outputs that your chosen star selector and psf determiner support.

To investigate the pipe_tasks_measurePsf_Debug, put something like

.. code-block :: none

    import lsstDebug
    def DebugInfo(name):
        di = lsstDebug.getInfo(name)        # N.b. lsstDebug.Info(name) would call us recursively

        if name == "lsst.pipe.tasks.measurePsf" :
            di.display = True
            di.displayExposure = False          # display the Exposure + spatialCells
            di.displayPsfCandidates = True      # show mosaic of candidates
            di.displayPsfMosaic = True          # show mosaic of reconstructed PSF(xy)
            di.displayResiduals = True          # show residuals
            di.showBadCandidates = True         # Include bad candidates
            di.normalizeResiduals = False       # Normalise residuals by object amplitude

        return di

    lsstDebug.Info = DebugInfo

into your debug.py file and run measurePsfTask.py with the --debug flag.
