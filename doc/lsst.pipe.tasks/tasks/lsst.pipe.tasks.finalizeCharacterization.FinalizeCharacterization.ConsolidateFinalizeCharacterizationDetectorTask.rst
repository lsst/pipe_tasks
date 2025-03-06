.. lsst-task-topic:: lsst.pipe.tasks.finalizeCharacterization.ConsolidateFinalizeCharacterizationDetectorTask

###############################################
ConsolidateFinalizeCharacterizationDetectorTask
###############################################

`ConsolidateFinalizeCharacterizationDetectorTask` is a task to consolidate all the outputs from the per-detector :doc:`lsst.pipe.tasks.finalizeCharacterization.FinalizeCharacterizationDetectorTask` to make an output compatible with the output from the per-visit :doc:`lsst.pipe.tasks.finalizeCharacterization.FinalizeCharacterizationTask`.

.. _lsst.pipe.tasks.finalizeCharacterization.ConsolidateFinalizeCharacterizationDetectorTask-summary:

Processing summary
==================

This task reads in the individual per-detector exposure catalogs with the psf and aperture correction map, as well as the per-detector sources used for psf and aperture correction estimation, and consolidates them into single catalogs.

There are no subtasks for configurations for this task.
