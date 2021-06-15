.. lsst-task-topic:: lsst.pipe.tasks.loadReferenceCatalog.LoadReferenceCatalogTask

########################
LoadReferenceCatalogTask
########################

``LoadReferenceCatalogTask`` is a convenience task that combines the reference object loader adapted for multiple filters; color term application; and reference selection in one task.
This task returns a numpy record array with the magnitude information for each of the physical filters requested by the caller.

If appropriate, all proper motion corrections are handled by the reference object loader as configured.
Mapping from physical filter to reference filter is handled by the ``filterMap`` configured with the reference object loader, as well as the color term library.

The format of the reference catalogs returned by this task will be a numpy record array with the following datatype:

.. code-block:: python

    import numpy as np
    dtype = [('ra', 'np.float64'),
             ('dec', 'np.float64'),
             ('refMag', 'np.float32', (len(filterList), )),
             ('refMagErr', 'np.float32', (len(filterList), ))]

Reference magnitudes (AB) and errors will be 99 for non-detections for a given reference filter.

.. _lsst.pipe.tasks.loadReferenceCatalog.LoadReferenceCatalogTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.pipe.tasks.loadReferenceCatalog.LoadReferenceCatalogTask

.. _lsst.pipe.tasks.loadReferenceCatalog.LoadReferenceCatalogTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.pipe.tasks.loadReferenceCatalog.LoadReferenceCatalogTask

Configuration fields
====================

.. lsst-task-config-fields:: lsst.pipe.tasks.loadReferenceCatalog.LoadReferenceCatalogTask
