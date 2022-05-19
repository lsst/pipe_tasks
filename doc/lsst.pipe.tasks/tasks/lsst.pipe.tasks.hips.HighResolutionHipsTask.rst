.. lsst-task-topic:: lsst.pipe.tasks.hips.HighResolutionHipsTask

######################
HighResolutionHipsTask
######################

`HighResolutionHipsTask` is a task to warp coadd images into (`HiPS <https://www.ivoa.net/documents/HiPS/>`_)-compatible HPX exposures.
Currently, the task includes a custom quantum-graph generation code, which is accessed from the command-line with ``build-high-resolution-hips-qg``.
Note that this task only generates HiPS-compatible exposures, it does not generate a HiPS directory tree.

.. _lsst.pipe.tasks.hips.HighResolutionHipsTask-summary:

Processing summary
==================

In order to use this task from ``pipetask``, you must generate a custom quantum graph.
There are two stages.
First one segments the survey with :code:`build-high-resolution-hips-qg segment`.
Second, one builds a quantum graph for a given segment with :code:`build-high-resolution-hips-qg build`.

.. code-block:: bash

    build-high-resolution-hips-qg segment -r REPO -p PIPELINE -c COLLECTIONS [COLLECTIONS] [-o HPIX_BUILD_ORDER]

The default for the build healpix order (that is, :code:`nside = 2**HPIX_BUILD_ORDER`) is 1, so that the generator segments the sky over large areas.
The ``PIPELINE`` is a pipeline yaml that must contain only one task (``HighResolutionHipsTask``) and should contain any configuration overrides.
Depending on the workflow engine, one may wish to build quantum graphs over a smaller region of the sky.
This will print out healpix pixels (nest ordering) over which to segment the sky.
For each healpix segment one can now build a quantum graph:

.. code-block:: bash

    build-high-resolution-hips-qg build -r REPO -p PIPELINE -c COLLECTIONS [COLLECTIONS] [-o HPIX_BUILD_ORDER] -f FILENAME -P PIXELS [PIXELS ...]

This will output a quantum graph covering ``PIXELS`` at order ``HPIX_BUILD_ORDER`` and output the graph to ``FILENAME``.
This quantum graph may now be used in a regular ``pipetask`` run.

.. _lsst.pipe.tasks.finalizeCharacterization.HighResolutionHipsTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.pipe.tasks.hips.HighResolutionHipsTask

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.pipe.tasks.hips.HighResolutionHipsTask

.. _lsst.pipe.tasks.hips.HighResolutionHipsTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.pipe.tasks.hips.HighResolutionHipsTask
