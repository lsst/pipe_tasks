.. lsst-task-topic:: lsst.pipe.tasks.finalizeCharacterization.FinalizeCharacterizationDetectorTask

####################################
FinalizeCharacterizationDetectorTask
####################################

`FinalizeCharacterizationDetectorTask` is a task to rerun image characterization (determine the PSF model and measure aperture corrections) after initial characterization, calibration, and isolated star association.
Currently, this allows PSF stars to be reserved consistently for all overlapping visits/detectors.
This task differs from :doc:`lsst.pipe.tasks.finalizeCharacterization.FinalizeCharacterizationTask` in that it runs per-detector and thus allows for greater parallelization.
However, it means that this task cannot allow PSF and aperture correction modeling to utilize full visits.

Running this task should be followed by running :doc:`lsst.pipe.tasks.finalizeCharacterization.ConsolidateFinalizeCharacterizationDetectorTask` to make the full exposure catalog containing all the detectors in the visit for downstream consumption.

.. _lsst.pipe.tasks.finalizeCharacterization.FinalizeCharacterizationDetectorTask-summary:

Processing summary
==================

`FinalizeCharacterizationDetectorTask` first reads the isolated source association table to get a list of PSF modeling and reserve stars that is consistent across overlapping visits, and then runs PSF modeling and aperture correction modeling per detector.
In the first stage, the task will:

- Read in ``isolated_star_cat`` and ``isolated_star_sources`` for all tracts that overlap the given visit.
- Select isolated sources that have been detected in the band of the visit.
- Reserve a configurable fraction of these isolated sources using the task configured in ``config.reserve_selection``.

In the second stage, the task will run per-detector:

- Select sources (``src``) from the calibrated exposure (``calexp``) above a configurable signal-to-noise.
- Match by source id to the ``isolated_star_sources`` and mark PSF candidate and reserve stars.
- Make PSF candidates using the task configured in ``config.make_psf_candidates``.
- Determine the PSF using the configured PSF determination task configured in ``config.psf_determiner``.
- Run measurement on the PSF candidate, used, and reserved stars using the task configured in ``config.measurement``.
- Run the aperture correction measurement task configured in ``config.measure_ap_corr`` using the PSF stars.
- Run the aperture correction application task configured in ``config.apply_ap_corr`` on the PSF stars.

The task returns an exposure catalog containing the PSF model for the detector and the aperture correction map, and an astropy table with all the measurements and flags are returned for persistence.

.. _lsst.pipe.tasks.finalizeCharacterization.FinalizeCharacterizationDetectorTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.pipe.tasks.finalizeCharacterization.FinalizeCharacterizationDetectorTask

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.pipe.tasks.finalizeCharacterization.FinalizeCharacterizationDetectorTask

.. _lsst.pipe.tasks.finalizeCharacterization.FinalizeCharacterizationDetectorTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.pipe.tasks.finalizeCharacterization.FinalizeCharacterizationDetectorTask
