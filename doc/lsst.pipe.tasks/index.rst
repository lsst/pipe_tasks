.. _lsst.pipe.tasks:

###############
lsst.pipe.tasks
###############

``lsst.pipe.tasks`` provides many of the `~lsst.pipe.base.Task` classes that drive the LSST Science Pipelines.
The :ref:`pipeline tasks <lsst.pipe.tasks-pipeline-tasks>` listed here are useful data processing entry points for most users.

``lsst.pipe.tasks`` does not provide all the tasks and pipeline tasks in the LSST Science Pipelines.
For a complete list of all available tasks, see :doc:`/tasks` and for an introduction to processing data see :doc:`/getting-started/index`.


.. _lsst.pipe.tasks-using:

Using lsst.pipe.tasks
=====================

.. toctree linking to topics related to using the module's APIs.

.. toctree::
   :maxdepth: 1

   deblending-flags-overview

.. _lsst.pipe.tasks-contributing:

Contributing
============

``lsst.pipe.tasks`` is developed at https://github.com/lsst/pipe_tasks.
You can find Jira issues for this module under the `pipe_tasks <https://jira.lsstcorp.org/issues/?jql=project%20%3D%20DM%20AND%20component%20%3D%20pipe_tasks>`_ component.

.. If there are topics related to developing this module (rather than using it), link to this from a toctree placed here.

.. .. toctree::
..    :maxdepth: 1

.. _lsst.pipe.tasks-command-line-taskref:

Task reference
==============

.. _lsst.pipe.tasks-pipeline-tasks:

Pipeline tasks
--------------

.. lsst-pipelinetasks::
   :root: lsst.pipe.tasks

.. _lsst.pipe.tasks-tasks:

Tasks
-----

.. lsst-tasks::
   :root: lsst.pipe.tasks
   :toctree: tasks

.. _lsst.pipe.tasks-configs:

Configurations
--------------

.. lsst-configs::
   :root: lsst.pipe.tasks
   :toctree: configs

.. _lsst.pipe.tasks-pyapi:

Python API reference
====================

.. automodapi:: lsst.pipe.tasks.assembleCoadd
.. automodapi:: lsst.pipe.tasks.dcrAssembleCoadd

.. automodapi:: lsst.pipe.tasks.metrics
   :no-main-docstr:
   :no-inheritance-diagram:
