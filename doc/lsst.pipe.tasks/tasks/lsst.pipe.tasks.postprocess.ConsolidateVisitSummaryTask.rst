.. lsst-task-topic:: lsst.pipe.tasks.postprocess.ConsolidateVisitSummaryTask

##########################
ConsolidateVisitSummaryTask
##########################

``ConsolidateVisitSummaryTask`` combines the important metadata (wcs, detector, psf size, etc.) into one per-visit exposure catalog (dataset `visitSummary`).

``ConsolidateVisitSummaryTask`` is available as a :ref:`command-line task <lsst.pipe.tasks-command-line-tasks>`, :command:`consolidateVisitSummary.py`.

.. _lsst.pipe.tasks.postprocess.ConsolidateVisitSummary-summary:

Processing summary
==================

``ConsolidateVisitSummaryTask`` reads in all detector-level calibrated exposure metadata tables (dataset `calexp`) for a given visit, combines these data into an exposure catalog, and writes the result out as a visit-level summary catalog (dataset `visitSummary`).

.. lsst.pipe.tasks.postprocess.ConsolidateVisitSummaryTask-cli:

consolidateVisitSummary.py command-line interface
================================================

.. code-block:: text

   consolidateVisitSummary.py REPOPATH [@file [@file2 ...]] [--output OUTPUTREPO | --rerun RERUN] [--id] [other options]

Key arguments:

.. option:: REPOPATH

   The input Butler repository's URI or file path.

Key options:

.. option:: --id

   The data IDs to process.

.. seealso::

   See :ref:`command-line-task-argument-reference` for details and additional options.

.. _lsst.pipe.tasks.postprocess.ConsolidateVisitSummaryTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.pipe.tasks.postprocess.ConsolidateVisitSummaryTask

.. _lsst.pipe.tasks.postprocess.ConsolidateVisitSummaryTask-butler:

Butler datasets
===============

When run as the ``consolidateVisitSummary.py`` command-line task, or directly through the `~lsst.pipe.tasks.postprocess.ConsolidateVisitSummaryTask.runDataRef` method, ``ConsolidateVisitSummaryTask`` obtains datasets from the input Butler data repository and persists outputs to the output Butler data repository.
Note that configurations for ``ConsolidateVisitSummaryTask``, and its subtasks, affect what datasets are persisted and what their content is.

.. _lsst.pipe.tasks.postprocess.ConsolidateVisitSummaryTask-butler-inputs:

Input datasets
--------------

``calexp``
    Per-detector, calibrated exposures with metadata (wcs, psf, etc.)

.. _lsst.pipe.tasks.postprocess.ConsolidateVisitSummaryTask-butler-outputs:

Output datasets
---------------

``visitSummary``
    Per-visit summary catalog of ccd/visit metadata and other quantities.


.. _lsst.pipe.tasks.postprocess.ConsolidateSourceTableTask-subtasks:

Examples
========

The following command shows an example of how to run the task on an example HSC repository.

.. code-block:: bash

    consolidateVisitSummary.py /datasets/hsc/repo --rerun <rerun name> --id visit=30504

.. _lsst.pipe.tasks.postprocess.ConsolidateVisitSummaryTask-debug:
