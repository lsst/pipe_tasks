.. lsst-task-topic:: lsst.pipe.tasks.propagateSourceFlags.PropagateSourceFlagsTask

########################
PropagateSourceFlagsTask
########################

`PropagateSourceFlagsTask` is a task to associate flags from source catalogs (e.g. which objects were used for astrometry, photometry, and psf fitting) with coadd object catalogs.
Flagged sources may come from a mix of two different types of source catalogs.
The `sourceTable_visit` catalogs from `CalibrateTask` contain flags for the first round of astrometry/photometry/psf fits.
The `finalized_src_table` catalogs from `FinalizeCalibrationTask` contain flags from the second round of psf fitting.
Which table should be used for the source of which flags depends on the configuration of coadds (in particular ``makeWarpConfig.useVisitSummaryPsf``).

.. _lsst.pipe.tasks.propagateSourceFlags.PropagateSourceFlagsTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.pipe.tasks.propagateSourceFlags.PropagateSourceFlagsTask

.. _lsst.pipe.tasks.propagateSourceFlags.PropagateSourceFlagsTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.pipe.tasks.propagateSourceFlags.PropagateSourceFlagsTask

.. _lsst.pipe.tasks.propagateSourceFlags.PropagateSourceFlagsTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.pipe.tasks.propagateSourceFlags.PropagateSourceFlagsTask
