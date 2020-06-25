.. lsst-task-topic:: lsst.pipe.tasks.processCcd.ProcessCcdTask

##############
ProcessCcdTask
##############

``ProcessCcdTask`` provides a preliminary astrometric and photometric calibration for a single frame (a ``raw`` dataset), yielding a ``calexp`` dataset.
``ProcessCcdTask`` is available as a :ref:`command-line task <lsst.pipe.tasks-command-line-tasks>`, ``processCcd.py``.

.. todo::

   Link to related tasks and direct users (if applicable)?

.. _lsst.pipe.tasks.processCcd.ProcessCcdTask-summary:

Processing summary
==================

``ProcessCcdTask`` runs this sequence of operations:

1. Removes instrumental signature from the ``raw`` dataset by calling the :lsst-config-field:`~lsst.pipe.tasks.processCcd.ProcessCcdConfig.isr` subtask (default: :lsst-task:`~lsst.ip.isr.isrTask.IsrTask`).

   This is the ISR step.
   In the LSST Science Pipelines, individual cameras tune the configurations for the :lsst-config-field:`~lsst.pipe.tasks.processCcd.ProcessCcdConfig.isr` subtask.
   By running this task, you automatically leverage expertise from the camera's builders and community.

2. Characterizes the post-ISR exposure by calling the :lsst-config-field:`~lsst.pipe.tasks.processCcd.ProcessCcdConfig.charImage` subtask (default: :lsst-task:`~lsst.pipe.tasks.characterizeImage.CharacterizeImageTask`):

   - Models the background.
   - Models the PSF.
   - Repairs cosmic ray hits.
   - Detects and measures bright sources.
   - Measures an aperture correction.

3. Calibrates the post-characterization exposure by calling the :lsst-config-field:`~lsst.pipe.tasks.processCcd.ProcessCcdConfig.calibrate` subtask (default: :lsst-task:`~lsst.pipe.tasks.calibrate.CalibrateTask`):

   - Detects sources more completely (using PSF and aperture corrections from the previous step).
   - Fits an improved WCS.
   - Fits a photometric zeropoint.

   **Note:** you can disable the calibration step with the :lsst-config-field:`~lsst.pipe.tasks.processCcd.ProcessCcdConfig.doCalibrate` configuration field.

.. _lsst.pipe.tasks.processCcd.ProcessCcdTask-cli:

processCcd.py command-line interface
====================================

.. code-block:: text

   processCcd.py REPOPATH [@file [@file2 ...]] [--output OUTPUTREPO | --rerun RERUN] [--id] [other named arguments]

Key arguments:

.. option:: REPOPATH

   The input Butler repository's URI or file path.

Key options:

.. option:: --id

   The data IDs to process.

.. seealso::

   See :ref:`command-line-task-argument-reference` for details and additional options.

.. _lsst.pipe.tasks.processCcd.ProcessCcdTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.pipe.tasks.processCcd.ProcessCcdTask

.. _lsst.pipe.tasks.processCcd.ProcessCcdTask-butler:

Butler datasets
===============

When run as the ``processCcd.py`` command-line task, or directly through the `~lsst.pipe.tasks.processCcd.ProcessCcdTask.runDataRef` method, ``ProcessCcdTask`` obtains datasets from the input Butler data repository and persists outputs to the output Butler data repository.
Note that configurations for ``ProcessCcdTask`` and its subtasks affect what datasets are persisted, and what their content is.

.. todo::

   Make each dataset a link to a canonical reference page.

.. _lsst.pipe.tasks.processCcd.ProcessCcdTask-butler-inputs:

Input datasets
--------------

``raw``
    Raw dataset from a camera, as ingested into the input Butler data repository.
    Unpersisted by the :lsst-config-field:`~lsst.pipe.tasks.processCcd.ProcessCcdConfig.isr` subtask.

.. _lsst.pipe.tasks.processCcd.ProcessCcdTask-butler-outputs:

Output datasets
---------------

``calexp``
    The calibrated exposure.
    Persisted by the :lsst-config-field:`~lsst.pipe.tasks.processCcd.ProcessCcdConfig.calibrate` subtask.

    The default subtask (:lsst-task:`~lsst.pipe.tasks.calibrate.CalibrateTask`) adds the following metadata:

    ``MAGZERO_RMS``
        The RMS (standard deviation) of ``MAGZERO``, measured by the :lsst-config-field:`~lsst.pipe.tasks.calibrate.CalibrateTask.photoCal` subtask.
    ``MAGZERO_NOBJ``: ``Number of stars used to estimate ``MAGZERO``.
        This is ``ngood`` reported by the :lsst-config-field:`~lsst.pipe.tasks.calibrate.CalibrateTask.photoCal` subtask.
    ``COLORTERM1``
        Always ``0.0``.
    ``COLORTERM2``
        Always ``0.0``.
    ``COLORTERM3``
        Always ``0.0``.

``calexpBackground``
    Background model for the ``calexp`` calibrated exposure.
    Persisted by the :lsst-config-field:`~lsst.pipe.tasks.processCcd.ProcessCcdConfig.calibrate` subtask.

``icExp``
    The characterized exposure.
    Persisted by the :lsst-config-field:`~lsst.pipe.tasks.ProcessCcdConfig.charImage` subtask.

``icExpBackground``
    Background model of the ``icExp`` exposure.

``icSrc``
    The source catalog of the characterized exposure, ``icExp``.
    Persisted by the :lsst-config-field:`~lsst.pipe.tasks.ProcessCcdConfig.charImage` subtask.

``postISRCCD``
    Post-ISR exposure.
    Persisted by the :lsst-config-field:`~lsst.pipe.tasks.ProcessCcdConfig.charImage` subtask.

``src``
    Table of sources measured in the calibrated exposure.
    Persisted by the :lsst-config-field:`~lsst.pipe.tasks.processCcd.ProcessCcdConfig.calibrate` subtask.

``srcMatch``
    Table of matches between the sources and reference objects, created by the astrometry solver.
    Persisted by the :lsst-config-field:`~lsst.pipe.tasks.processCcd.ProcessCcdConfig.calibrate` subtask.

``srcMatchFull``
    Denormalized version of ``srcMatch``.
    Persisted by the :lsst-config-field:`~lsst.pipe.tasks.processCcd.ProcessCcdConfig.calibrate` subtask.

.. _lsst.pipe.tasks.processCcd.ProcessCcdTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.pipe.tasks.processCcd.ProcessCcdTask

.. _lsst.pipe.tasks.processCcd.ProcessCcdTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.pipe.tasks.processCcd.ProcessCcdTask

.. _lsst.pipe.tasks.processCcd.ProcessCcdTask-examples:

Examples
========

.. _ProcessCcdTask-simple-example:

Simple command-line usage
-------------------------

``ProcessCcdTask`` is used as a command-line task that processes ``raw`` datasets into a Butler repository.

As an example, you can use ``raw`` datasets in the ``obs_test`` package.
First, set up the relevant packages on the command line:

.. code-block:: bash

   setup lsst_distrib
   setup -k obs_test

Then run the ``processCcd.py`` task:

.. code-block:: bash

   processCcd.py $OBS_TEST_DIR/data/input --output processCcdOut --id

Using the ``--id`` option without any data ID keys finds all available ``raw`` data in the Butler dataset for processing.
The output ``calexp`` and ``src`` datasets are written to the :file:`processCcdOut` directory.

.. important::

   If :file:`processCcdOut` already exists, you'll need to either delete the existing directory or give the :option:`--output` option a different directory name.

.. _lsst.pipe.tasks.processCcd.ProcessCcdTask-debug:

Debugging
=========

``ProcessCcdTask`` does not have debug output, though its subtasks may.
