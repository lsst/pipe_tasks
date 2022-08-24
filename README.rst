##########
pipe_tasks
##########
``pipe_tasks`` is a package in the `LSST Science Pipelines <https://pipelines.lsst.io>`_.

``pipe_tasks`` provides many of the `Task <https://pipelines.lsst.io/py-api/lsst.pipe.base.Task.html#lsst.pipe.base.Task>`_ classes that drive the LSST Science Pipelines. The `pipeline tasks listed here <https://pipelines.lsst.io/modules/lsst.pipe.tasks/index.html#lsst-pipe-tasks-pipeline-tasks>`_ are useful data processing entry points for most users.

The repository also contains several Pipelines used for processing. You can also assemble your own pipelines by combining individual tasks through their Python APIs.

``pipe_tasks`` does not provide all the tasks and command-line tasks in the LSST Science Pipelines. Tasks can be found in many other high-level packages in the Science Pipelines, such as `ip_isr <https://github.com/lsst/ip_isr>`_.

To learn more about the task framework in general, see the `lsst.pipe.base <https://pipelines.lsst.io/modules/lsst.pipe.base/index.html#lsst-pipe-base>`_ module documentation.

The package namespace itself is mostly empty. Each specific processing tool must be imported directly from the module; for instance,

``from lsst.pipe.tasks.processCcd import ProcessCcdTask``
