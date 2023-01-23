.. lsst-task-topic:: lsst.pipe.tasks.update_visit_summary.UpdateVisitSummaryTask

######################
UpdateVisitSummaryTask
######################

``UpdateVisitSummaryTask`` combines updated versions of the various `~lsst.afw.image.Exposure` component objects used to characterize and calibrate a single-epoch image into a single per-visit exposure catalog (``finalVisitSummary``).
It also recomputes summary statistics to reflect these updates.

.. _lsst.pipe.tasks.update_visit_summary.UpdateVisitSummary-summary:

Processing summary
==================

``UpdateVisitSummaryTask`` reads in the initial summary dataset of essentially the same form (``visitSummary``), then replaces the object fields of each record with objects loaded from other (often optional) input datasets that contain newer (often final) versions of those objects.
When an object field is replaced, any related summary statistics in the catalog's non-object columns are also recomputed.

See connection and ``run`` argument documentation for details.

.. _lsst.pipe.tasks.update_visit_summary.UpdateVisitSummaryTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.pipe.tasks.update_visit_summary.UpdateVisitSummaryTask

.. _lsst.pipe.tasks.update_visit_summary.UpdateVisitSummaryTask-butler:

Butler datasets
===============

When run through the `~lsst.pipe.tasks.update_visit_summary.UpdateVisitSummaryTask.runQuantum` method, ``UpdateVisitSummaryTask`` obtains datasets from the input Butler data repository and persists outputs to the output Butler data repository.

In this mode, the PSF and aperture correction map are always replaced (since at present all relevant pipelines do recompute these), even though they are optional when calling `lsst.pipe.tasks.update_visit_summary.UpdateVisitSummaryTask.run` directly.

The main output dataset, ``finalVisitSummary`` (by default), can typically be
used to provide all downstream tasks with the best versions of all calibrations
for each detector.

.. _lsst.pipe.tasks.update_visit_summary.UpdateVisitSummaryTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.pipe.tasks.update_visit_summary.UpdateVisitSummaryTask

.. _lsst.pipe.tasks.update_visit_summary.UpdateVisitSummaryTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.pipe.tasks.update_visit_summary.UpdateVisitSummaryTask
