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

.. automodapi:: lsst.pipe.tasks.associationUtils
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.background
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.calexpCutout
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.calibrateImage
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.calibrate
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.characterizeImage
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.coaddBase
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.coaddInputRecorder
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.colorterms
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.computeExposureSummaryStats
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.cosmicRayPostDiff
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.deblendCoaddSourcesPipeline
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.diff_matched_tract_catalog
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.drpAssociationPipe
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.drpDiaCalculationPipe
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.exampleStatsTasks
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.extended_psf
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.finalizeCharacterization
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.fit_multiband
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.functors
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.getRegionTimeFromVisit
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.healSparseMapping
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.healSparseMappingProperties
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.hips
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.insertFakes
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.interpImage
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.isolatedStarAssociation
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.loadReferenceCatalog
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.makeDiscreteSkyMap
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.make_direct_warp
   :no-inheritance-diagram:

   .. automodapi:: lsst.pipe.tasks.make_psf_matched_warp
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.makeWarp
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.matchBackgrounds
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.matchFakes
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.match_tract_catalog
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.match_tract_catalog_probabilistic
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.measurePsf
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.mergeDetections
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.mergeMeasurements
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.metrics
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.multiBand
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.multiBandUtils
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.objectMasks
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.photoCal
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.postprocess
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.processBrightStars
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.processCcdWithFakes
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.propagateSourceFlags
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.quickFrameMeasurement
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.read_curated_calibs
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.registerImage
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.repair
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.reserveIsolatedStars
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.scaleZeroPoint
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.selectImages
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.setPrimaryFlags
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.simpleAssociation
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.skyCorrection
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.snapCombine
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.statistic
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.visualizeVisit
   :no-inheritance-diagram:

.. automodapi:: lsst.pipe.tasks.warpAndPsfMatch
   :no-inheritance-diagram:

