.. lsst-task-topic:: lsst.pipe.tasks.postprocess.ConsolidateVisitSummaryTask

###########################
ConsolidateVisitSummaryTask
###########################

``ConsolidateVisitSummaryTask`` combines the non-trivial metadata, including the wcs, detector id, psf size and shape, filter, and bounding box corners into one per-visit exposure catalog (dataset `visitSummary`).

.. _lsst.pipe.tasks.postprocess.ConsolidateVisitSummary-summary:

Processing summary
==================

``ConsolidateVisitSummaryTask`` reads in detector-level processed exposure metadata tables (dataset `calexp`) for a given visit, combines these data into an exposure catalog, and writes the result out as a visit-level summary catalog (dataset `visitSummary`).
The metadata from each exposure/detector includes:

- The ``visitInfo``.
- The ``wcs``.
- The ``photoCalib``.
- The ``physical_filter`` and ``band`` (if available).
- The psf size, shape, and effective area at the center of the detector.
- The corners of the bounding box in right ascension/declination.

Other quantities such as detector, PSF, aperture correction map, and
transmission curve are not persisted because of storage concerns, and
because of their limited utility as summary statistics.

.. _lsst.pipe.tasks.postprocess.ConsolidateVisitSummaryTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.pipe.tasks.postprocess.ConsolidateVisitSummaryTask

.. _lsst.pipe.tasks.postprocess.ConsolidateVisitSummaryTask-butler:

Butler datasets
===============

When run through the `~lsst.pipe.tasks.postprocess.ConsolidateVisitSummaryTask.runQuantum` method, ``ConsolidateVisitSummaryTask`` obtains datasets from the input Butler data repository and persists outputs to the output Butler data repository.

.. _lsst.pipe.tasks.postprocess.ConsolidateVisitSummaryTask-butler-inputs:

Input datasets
--------------

``calexp``
    Per-detector, processed exposures with metadata (wcs, psf, etc.)

.. _lsst.pipe.tasks.postprocess.ConsolidateVisitSummaryTask-butler-outputs:

Output datasets
---------------

``visitSummary``
    Per-visit summary catalog of ccd/visit metadata.
``visitSummary_schema``
    Catalog with no rows with the same schema as ``visitSummary``.
