.. lsst-task-topic:: lsst.pipe.tasks.postprocess.ConsolidateSourceTableTask

##########################
ConsolidateSourceTableTask
##########################

``ConsolidateSourceTableTask`` concatenates per-detector Source Tables (dataset `sourceTable`) into one per-visit Source Table (dataset `sourceTable_visit`).
It only does I/O, and therefore has no run method.
The inputs have already been transformed to the `DPDD <https://lse-163.lsst.io>`_-specified columns.
This task assumes that they are sufficiently narrow to fit all tables for a given visit in memory at once.

It is the third of three postprocessing tasks to convert a `src` table to a
per-visit Source Table that conforms to the standard data model. The first is
:doc:`lsst.pipe.tasks.postprocess.WriteSourceTableTask`. The second is :doc:`lsst.pipe.tasks.postprocess.TransformSourceTableTask`.

.. _lsst.pipe.tasks.postprocess.ConsolidateSourceTableTask-summary:

Processing summary
==================

``ConsolidateSourceTableTask`` reads in all detector-level Source Tables (dataset `sourceTable`) for a given visit, concatenates them, and writes the result out as a visit-level Source Table (dataset `sourceTable_visit`)

.. _lsst.pipe.tasks.postprocess.ConsolidateSourceTableTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.pipe.tasks.postprocess.ConsolidateSourceTableTask

.. _lsst.pipe.tasks.postprocess.ConsolidateSourceTableTask-butler:

Butler datasets
===============

When run through the `~lsst.pipe.tasks.postprocess.ConsolidateSourceTableTask.runQuantum` method, ``ConsolidateSourceTableTask`` obtains datasets from the input Butler data repository and persists outputs to the output Butler data repository.
Note that configurations for ``ConsolidateSourceTableTask``, and its subtasks, affect what datasets are persisted and what their content is.

.. _lsst.pipe.tasks.postprocess.ConsolidateSourceTableTask-butler-inputs:

Input datasets
--------------

``sourceTable``
    Per-detector, parquet-formatted Source Table that has been transformed to DPDD_-specification

.. _lsst.pipe.tasks.postprocess.ConsolidateSourceTableTask-butler-outputs:

Output datasets
---------------

``sourceTable_visit``
    Per-visit, parquet-formatted Source Table that has been transformed to DPDD_-specification


.. _lsst.pipe.tasks.postprocess.ConsolidateSourceTableTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.pipe.tasks.postprocess.ConsolidateSourceTableTask

.. _lsst.pipe.tasks.postprocess.ConsolidateSourceTableTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.pipe.tasks.postprocess.ConsolidateSourceTableTask
