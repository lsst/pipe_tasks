.. lsst-task-topic:: lsst.pipe.tasks.metrics.NumberDeblendedSourcesMetricTask

################################
NumberDeblendedSourcesMetricTask
################################

``NumberDeblendedSourcesMetricTask`` computes the number of science sources deblended when processing data through single-frame processing (as the ``pipe_tasks.numDeblendedSciSources`` metric).
It requires source catalogs (by default, a ``src`` dataset) as input, and operates at image-level granularity.

.. _lsst.pipe.tasks.metrics.NumberDeblendedSourcesMetricTask-summary:

Processing summary
==================

``NumberDeblendedSourcesMetricTask`` reads source catalogs (``src`` datasets, by default) and adds up the number of deblended top-level sources in those catalogs.
Sky sources, and sources that are themselves the result of deblending, are not counted.

.. _lsst.pipe.tasks.metrics.NumberDeblendedSourcesMetricTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.pipe.tasks.metrics.NumberDeblendedSourcesMetricTask

.. _lsst.pipe.tasks.metrics.NumberDeblendedSourcesMetricTask-butler:

Butler datasets
===============

Input datasets
--------------

``sources``
    The catalog type for science sources (default: ``src``).

.. _lsst.pipe.tasks.metrics.NumberDeblendedSourcesMetricTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.pipe.tasks.metrics.NumberDeblendedSourcesMetricTask

.. _lsst.pipe.tasks.metrics.NumberDeblendedSourcesMetricTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.pipe.tasks.metrics.NumberDeblendedSourcesMetricTask
