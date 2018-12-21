.. lsst-task-topic:: lsst.pipe.tasks.assembleCoadd.SafeClipAssembleCoaddTask

#########################
SafeClipAssembleCoaddTask
#########################

.. _lsst.pipe.tasks.assembleCoadd.SafeClipAssembleCoaddTask-summary:

Processing summary
==================

In ``AssembleCoaddTask``, we compute the coadd as an clipped mean (i.e.,
we clip outliers). The problem with doing this is that when computing the
coadd PSF at a given location, individual visit PSFs from visits with
outlier pixels contribute to the coadd PSF and cannot be treated correctly.
In this task, we correct for this behavior by creating a new
``badMaskPlane`` 'CLIPPED'. We populate this plane on the input
coaddTempExps and the final coadd where

    i. difference imaging suggests that there is an outlier and
    ii. this outlier appears on only one or two images.

Such regions will not contribute to the final coadd. Furthermore, any
routine to determine the coadd PSF can now be cognizant of clipped regions.
Note that the algorithm implemented by this task is preliminary and works
correctly for HSC data. Parameter modifications and or considerable
redesigning of the algorithm is likley required for other surveys.

``SafeClipAssembleCoaddTask`` uses a ``SourceDetectionTask``
"clipDetection" subtask and also sub-classes ``AssembleCoaddTask``.
You can retarget the ``SourceDetectionTask`` "clipDetection" subtask
if you wish.


.. _lsst.pipe.tasks.assembleCoadd.SafeClipAssembleCoaddTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.pipe.tasks.assembleCoadd.SafeClipAssembleCoaddTask

.. _lsst.pipe.tasks.assembleCoadd.SafeClipAssembleCoaddTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.pipe.tasks.assembleCoadd.SafeClipAssembleCoaddTask

.. _lsst.pipe.tasks.assembleCoadd.SafeClipAssembleCoaddTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.pipe.tasks.assembleCoadd.SafeClipAssembleCoaddTask

.. _lsst.pipe.tasks.assembleCoadd.AssembleCoaddTask-examples:

Examples
========

`SafeClipAssembleCoaddTask` assembles a set of warped ``coaddTempExp``
images into a coadded image. The `SafeClipAssembleCoaddTask` is invoked by
running assembleCoadd.py *without* the flag '--legacyCoadd'.

Usage of ``assembleCoadd.py`` expects a data reference to the tract patch
and filter to be coadded (specified using
'--id = [KEY=VALUE1[^VALUE2[^VALUE3...] [KEY=VALUE1[^VALUE2[^VALUE3...] ...]]')
along with a list of coaddTempExps to attempt to coadd (specified using
'--selectId [KEY=VALUE1[^VALUE2[^VALUE3...] [KEY=VALUE1[^VALUE2[^VALUE3...] ...]]').
Only the coaddTempExps that cover the specified tract and patch will be
coadded. A list of the available optional arguments can be obtained by
calling assembleCoadd.py with the --help command line argument:

.. code-block:: none

   assembleCoadd.py --help

To demonstrate usage of the `SafeClipAssembleCoaddTask` in the larger
context of multi-band processing, we will generate the HSC-I & -R band
coadds from HSC engineering test data provided in the ci_hsc package.
To begin, assuming that the lsst stack has been already set up, we must
set up the obs_subaru and ci_hsc packages. This defines the environment
variable $CI_HSC_DIR and points at the location of the package. The raw
HSC data live in the ``$CI_HSC_DIR/raw`` directory. To begin assembling
the coadds, we must first

- ``processCcd``
    process the individual ccds in $CI_HSC_RAW to produce calibrated exposures
- ``makeSkyMap``
    create a skymap that covers the area of the sky present in the raw exposures
- ``makeCoaddTempExp``
    warp the individual calibrated exposures to the tangent plane of the coadd</DD>

We can perform all of these steps by running

.. code-block:: none

   $CI_HSC_DIR scons warp-903986 warp-904014 warp-903990 warp-904010 warp-903988

This will produce warped coaddTempExps for each visit. To coadd the
warped data, we call ``assembleCoadd.py`` as follows:

.. code-block:: none

   assembleCoadd.py $CI_HSC_DIR/DATA --id patch=5,4 tract=0 filter=HSC-I \
   --selectId visit=903986 ccd=16 --selectId visit=903986 ccd=22 --selectId visit=903986 ccd=23 \
   --selectId visit=903986 ccd=100--selectId visit=904014 ccd=1 --selectId visit=904014 ccd=6 \
   --selectId visit=904014 ccd=12 --selectId visit=903990 ccd=18 --selectId visit=903990 ccd=25 \
   --selectId visit=904010 ccd=4 --selectId visit=904010 ccd=10 --selectId visit=904010 ccd=100 \
   --selectId visit=903988 ccd=16 --selectId visit=903988 ccd=17 --selectId visit=903988 ccd=23 \
   --selectId visit=903988 ccd=24

This will process the HSC-I band data. The results are written in
``$CI_HSC_DIR/DATA/deepCoadd-results/HSC-I``.

You may also choose to run:

.. code-block:: none

   scons warp-903334 warp-903336 warp-903338 warp-903342 warp-903344 warp-903346 nnn
   assembleCoadd.py $CI_HSC_DIR/DATA --id patch=5,4 tract=0 filter=HSC-R --selectId visit=903334 ccd=16 \
   --selectId visit=903334 ccd=22 --selectId visit=903334 ccd=23 --selectId visit=903334 ccd=100 \
   --selectId visit=903336 ccd=17 --selectId visit=903336 ccd=24 --selectId visit=903338 ccd=18 \
   --selectId visit=903338 ccd=25 --selectId visit=903342 ccd=4 --selectId visit=903342 ccd=10 \
   --selectId visit=903342 ccd=100 --selectId visit=903344 ccd=0 --selectId visit=903344 ccd=5 \
   --selectId visit=903344 ccd=11 --selectId visit=903346 ccd=1 --selectId visit=903346 ccd=6 \
   --selectId visit=903346 ccd=12

to generate the coadd for the HSC-R band if you are interested in following
multiBand Coadd processing as discussed in ``pipeTasks_multiBand``.


.. _lsst.pipe.tasks.assembleCoadd.SafeClipAssembleCoaddTask-debug:

Debugging
=========

The `lsst.pipe.base.cmdLineTask.CmdLineTask` interface supports a
flag ``-d`` to import ``debug.py`` from your ``PYTHONPATH``;
see `baseDebug` for more about ``debug.py`` files.
`SafeClipAssembleCoaddTask` has no debug variables of its own.
The ``SourceDetectionTask`` "clipDetection" subtasks may support debug
variables. See the documetation for `SourceDetectionTask` "clipDetection"
for further information.
