.. lsst-task-topic:: lsst.pipe.tasks.metrics.NumberDeblendChildSourcesMetricTask

###################################
NumberDeblendChildSourcesMetricTask
###################################

``NumberDeblendChildSourcesMetricTask`` computes the number of science sources created by deblending when processing data through single-frame processing (as the ``pipe_tasks.numDeblendChildSciSources`` metric).
It requires source catalogs (by default, a ``src`` dataset) as input, and operates at image-level granularity.

.. _lsst.pipe.tasks.metrics.NumberDeblendChildSourcesMetricTask-summary:

Processing summary
==================

``NumberDeblendChildSourcesMetricTask`` reads source catalogs (``src`` datasets, by default) and adds up the number of sources that have a deblending parent in those catalogs.
Sky sources, and sources that are themselves deblended, are not counted.

.. _lsst.pipe.tasks.metrics.NumberDeblendChildSourcesMetricTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.pipe.tasks.metrics.NumberDeblendChildSourcesMetricTask

.. _lsst.pipe.tasks.metrics.NumberDeblendChildSourcesMetricTask-butler:

Butler datasets
===============

Input datasets
--------------

``sources``
    The catalog type for science sources (default: ``src``).

.. _lsst.pipe.tasks.metrics.NumberDeblendChildSourcesMetricTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.pipe.tasks.metrics.NumberDeblendChildSourcesMetricTask

.. _lsst.pipe.tasks.metrics.NumberDeblendChildSourcesMetricTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.pipe.tasks.metrics.NumberDeblendChildSourcesMetricTask
