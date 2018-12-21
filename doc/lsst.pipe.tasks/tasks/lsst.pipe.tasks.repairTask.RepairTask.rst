.. lsst-task-topic:: lsst.pipe.tasks.repair.RepairTask

##########
RepairTask
##########

.. _lsst.pipe.tasks.repair.RepairTask-summary:

Processing summary
==================

RepairTask repairs known defects in an exposure, and it also identifies and
repairs cosmic rays.

.. _lsst.pipe.tasks.repair.RepairTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.pipe.tasks.repair.RepairTask

.. _lsst.pipe.tasks.repair.RepairTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.pipe.tasks.repair.RepairTask

.. _lsst.pipe.tasks.repair.RepairTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.pipe.tasks.repair.RepairTask

.. _lsst.pipe.tasks.repair.RepairTask-debug:

Debugging
=========

The available debug variables in RepairTask are:

display : A dictionary containing debug point names as keys with frame number as value. Valid keys are:
repair.before : display image before any repair is done
repair.after : display image after cosmic ray and defect correction
displayCR : If True, display the exposure on ds9's frame 1 and overlay bounding boxes around detects CRs.

To investigate the pipe_tasks_repair_Debug, put something like

.. code-block :: none

    import lsstDebug
    def DebugInfo(name):
        di = lsstDebug.getInfo(name)        # N.b. lsstDebug.Info(name) would call us recursively
        if name == "lsst.pipe.tasks.repair":
            di.display = {'repair.before':2, 'repair.after':3}
            di.displayCR = True
        return di

lsstDebug.Info = DebugInfo
into your debug.py file and run runRepair.py with the --debug flag.

