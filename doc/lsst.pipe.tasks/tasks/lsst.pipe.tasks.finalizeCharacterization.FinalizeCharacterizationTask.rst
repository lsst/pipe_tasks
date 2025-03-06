.. lsst-task-topic:: lsst.pipe.tasks.finalizeCharacterization.FinalizeCharacterizationTask

############################
FinalizeCharacterizationTask
############################

`FinalizeCharacterizationTask` is a task to rerun image characterization (determine the PSF model and measure aperture corrections) after initial characterization, calibration, and isolated star association.
Currently, this allows PSF stars to be reserved consistently for all overlapping visits/detectors.
In the future, this task will allow PSF and aperture correction modeling to take advantage of full visits (utilizing all the detectors) as well as the refined astrometric model.

.. _lsst.pipe.tasks.finalizeCharacterization.FinalizeCharacterizationTask-summary:

Processing summary
==================

`FinalizeCharacterizationTask` first reads the isolated source association table to get a consistent list of PSF modeling and reserve stars, and then runs PSF modeling and aperture correction modeling per detector.
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

The task returns an exposure catalog containing all the PSF models in the visit and all the aperture correction maps in the visit, and an astropy table with all the measurements and flags are returned for persistence.

.. _lsst.pipe.tasks.finalizeCharacterization.FinalizeCharacterizationTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.pipe.tasks.finalizeCharacterization.FinalizeCharacterizationTask

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.pipe.tasks.finalizeCharacterization.FinalizeCharacterizationTask

.. _lsst.pipe.tasks.finalizeCharacterization.FinalizeCharacterizationTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.pipe.tasks.finalizeCharacterization.FinalizeCharacterizationTask
