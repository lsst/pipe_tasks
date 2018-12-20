.. lsst-task-topic:: lsst.pipe.tasks.assembleCoadd.AssembleCoaddTask

#################
AssembleCoaddTask
#################

Assemble a coadded image from a set of ``CoaddTempExps``.

.. _lsst.pipe.tasks.assembleCoadd.AssembleCoaddTask-summary:

Processing summary
==================

We want to assemble a coadded image from a set of Warps (also called
coadded temporary exposures or ``coaddTempExps``).
Each input Warp covers a patch on the sky and corresponds to a single
run/visit/exposure of the covered patch. We provide the task with a list
of Warps (``selectDataList``) from which it selects Warps that cover the
specified patch (pointed at by ``dataRef``).

Each Warp that goes into a coadd will typically have an independent
photometric zero-point. Therefore, we must scale each Warp to set it to
a common photometric zeropoint. WarpType may be one of 'direct' or
'psfMatched', and the boolean configs ``config.makeDirect`` and
``config.makePsfMatched`` set which of the warp types will be coadded.
The coadd is computed as a mean with optional outlier rejection.
Criteria for outlier rejection are set in ``AssembleCoaddConfig``.
Finally, Warps can have bad 'NaN' pixels which received no input from the
source calExps. We interpolate over these bad (NaN) pixels.

.. _lsst.pipe.tasks.assembleCoadd.AssembleCoaddTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.pipe.tasks.assembleCoadd.AssembleCoaddTask

.. _lsst.pipe.tasks.assembleCoadd.AssembleCoaddTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.pipe.tasks.assembleCoadd.AssembleCoaddTask

.. _lsst.pipe.tasks.assembleCoadd.AssembleCoaddTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.pipe.tasks.assembleCoadd.AssembleCoaddTask

.. _lsst.pipe.tasks.assembleCoadd.AssembleCoaddTask-examples:

Examples
========

`AssembleCoaddTask` assembles a set of warped images into a coadded image.
The `AssembleCoaddTask` can be invoked by running ``assembleCoadd.py``
with the flag '--legacyCoadd'. Usage of assembleCoadd.py expects two
inputs: a data reference to the tract patch and filter to be coadded, and
a list of Warps to attempt to coadd. These are specified using ``--id`` and
``--selectId``, respectively:

.. code-block:: none

   --id = [KEY=VALUE1[^VALUE2[^VALUE3...] [KEY=VALUE1[^VALUE2[^VALUE3...] ...]]
   --selectId [KEY=VALUE1[^VALUE2[^VALUE3...] [KEY=VALUE1[^VALUE2[^VALUE3...] ...]]

Only the Warps that cover the specified tract and patch will be coadded.
A list of the available optional arguments can be obtained by calling
``assembleCoadd.py`` with the ``--help`` command line argument:

.. code-block:: none

   assembleCoadd.py --help

To demonstrate usage of the `AssembleCoaddTask` in the larger context of
multi-band processing, we will generate the HSC-I & -R band coadds from
HSC engineering test data provided in the ``ci_hsc`` package. To begin,
assuming that the lsst stack has been already set up, we must set up the
obs_subaru and ``ci_hsc`` packages. This defines the environment variable
``$CI_HSC_DIR`` and points at the location of the package. The raw HSC
data live in the ``$CI_HSC_DIR/raw directory``. To begin assembling the
coadds, we must first

- processCcd
- process the individual ccds in $CI_HSC_RAW to produce calibrated exposures
- makeSkyMap
- create a skymap that covers the area of the sky present in the raw exposures
- makeCoaddTempExp
- warp the individual calibrated exposures to the tangent plane of the coadd

We can perform all of these steps by running

.. code-block:: none

   $CI_HSC_DIR scons warp-903986 warp-904014 warp-903990 warp-904010 warp-903988

This will produce warped exposures for each visit. To coadd the warped
data, we call assembleCoadd.py as follows:

.. code-block:: none

   assembleCoadd.py --legacyCoadd $CI_HSC_DIR/DATA --id patch=5,4 tract=0 filter=HSC-I \
   --selectId visit=903986 ccd=16 --selectId visit=903986 ccd=22 --selectId visit=903986 ccd=23 \
   --selectId visit=903986 ccd=100 --selectId visit=904014 ccd=1 --selectId visit=904014 ccd=6 \
   --selectId visit=904014 ccd=12 --selectId visit=903990 ccd=18 --selectId visit=903990 ccd=25 \
   --selectId visit=904010 ccd=4 --selectId visit=904010 ccd=10 --selectId visit=904010 ccd=100 \
   --selectId visit=903988 ccd=16 --selectId visit=903988 ccd=17 --selectId visit=903988 ccd=23 \
   --selectId visit=903988 ccd=24

that will process the HSC-I band data. The results are written in
``$CI_HSC_DIR/DATA/deepCoadd-results/HSC-I``.

You may also choose to run:

.. code-block:: none

   scons warp-903334 warp-903336 warp-903338 warp-903342 warp-903344 warp-903346
   assembleCoadd.py --legacyCoadd $CI_HSC_DIR/DATA --id patch=5,4 tract=0 filter=HSC-R \
   --selectId visit=903334 ccd=16 --selectId visit=903334 ccd=22 --selectId visit=903334 ccd=23 \
   --selectId visit=903334 ccd=100 --selectId visit=903336 ccd=17 --selectId visit=903336 ccd=24 \
   --selectId visit=903338 ccd=18 --selectId visit=903338 ccd=25 --selectId visit=903342 ccd=4 \
   --selectId visit=903342 ccd=10 --selectId visit=903342 ccd=100 --selectId visit=903344 ccd=0 \
   --selectId visit=903344 ccd=5 --selectId visit=903344 ccd=11 --selectId visit=903346 ccd=1 \
   --selectId visit=903346 ccd=6 --selectId visit=903346 ccd=12

to generate the coadd for the HSC-R band if you are interested in
following multiBand Coadd processing as discussed in `pipeTasks_multiBand`
(but note that normally, one would use the `SafeClipAssembleCoaddTask`
rather than `AssembleCoaddTask` to make the coadd.

.. _lsst.pipe.tasks.assembleCoadd.AssembleCoaddTask-debug:

Debugging
=========

The `lsst.pipe.base.cmdLineTask.CmdLineTask` interface supports a
flag ``-d`` to import ``debug.py`` from your ``PYTHONPATH``; see
`baseDebug` for more about ``debug.py`` files. `AssembleCoaddTask` has
no debug variables of its own. Some of the subtasks may support debug
variables. See the documentation for the subtasks for further information.
