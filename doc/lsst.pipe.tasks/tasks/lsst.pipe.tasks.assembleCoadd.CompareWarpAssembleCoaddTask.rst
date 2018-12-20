.. lsst-task-topic:: lsst.pipe.tasks.assembleCoadd.CompareWarpAssembleCoaddTask

############################
CompareWarpAssembleCoaddTask
############################

.. _lsst.pipe.tasks.assembleCoadd.CompareWarpAssembleCoaddTask-summary:

Processing summary
==================

In ``AssembleCoaddTask``, we compute the coadd as an clipped mean (i.e.,
we clip outliers). The problem with doing this is that when computing the
coadd PSF at a given location, individual visit PSFs from visits with
outlier pixels contribute to the coadd PSF and cannot be treated correctly.
In this task, we correct for this behavior by creating a new badMaskPlane
'CLIPPED' which marks pixels in the individual warps suspected to contain
an artifact. We populate this plane on the input warps by comparing
PSF-matched warps with a PSF-matched median coadd which serves as a
model of the static sky. Any group of pixels that deviates from the
PSF-matched template coadd by more than config.detect.threshold sigma,
is an artifact candidate. The candidates are then filtered to remove
variable sources and sources that are difficult to subtract such as
bright stars. This filter is configured using the config parameters
``temporalThreshold`` and ``spatialThreshold``. The temporalThreshold is
the maximum fraction of epochs that the deviation can appear in and still
be considered an artifact. The spatialThreshold is the maximum fraction of
pixels in the footprint of the deviation that appear in other epochs
(where other epochs is defined by the temporalThreshold). If the deviant
region meets this criteria of having a significant percentage of pixels
that deviate in only a few epochs, these pixels have the 'CLIPPED' bit
set in the mask. These regions will not contribute to the final coadd.
Furthermore, any routine to determine the coadd PSF can now be cognizant
of clipped regions.

Note that the algorithm implemented by this task is
preliminary and works correctly for HSC data. Parameter modifications and
or considerable redesigning of the algorithm is likely required for other
surveys.

``CompareWarpAssembleCoaddTask`` sub-classes
``AssembleCoaddTask`` and instantiates ``AssembleCoaddTask``
as a subtask to generate the TemplateCoadd (the model of the static sky).

.. _lsst.pipe.tasks.assembleCoadd.CompareWarpAssembleCoaddTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.pipe.tasks.assembleCoadd.CompareWarpAssembleCoaddTask

.. _lsst.pipe.tasks.assembleCoadd.CompareWarpAssembleCoaddTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.pipe.tasks.assembleCoadd.CompareWarpAssembleCoaddTask

.. _lsst.pipe.tasks.assembleCoadd.CompareWarpAssembleCoaddTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.pipe.tasks.assembleCoadd.CompareWarpAssembleCoaddTask

.. _lsst.pipe.tasks.assembleCoadd.CompareWarpAssembleCoaddTask-examples:

Examples
========

``CompareWarpAssembleCoaddTask`` assembles a set of warped images into a
coadded image. The ``CompareWarpAssembleCoaddTask`` is invoked by running
``assembleCoadd.py`` with the flag ``--compareWarpCoadd``.
Usage of ``assembleCoadd.py`` expects a data reference to the tract patch
and filter to be coadded (specified using
'--id = [KEY=VALUE1[^VALUE2[^VALUE3...] [KEY=VALUE1[^VALUE2[^VALUE3...] ...]]')
along with a list of coaddTempExps to attempt to coadd (specified using
'--selectId [KEY=VALUE1[^VALUE2[^VALUE3...] [KEY=VALUE1[^VALUE2[^VALUE3...] ...]]').
Only the warps that cover the specified tract and patch will be coadded.
A list of the available optional arguments can be obtained by calling
``assembleCoadd.py`` with the ``--help`` command line argument:

.. code-block:: none

   assembleCoadd.py --help

To demonstrate usage of the ``CompareWarpAssembleCoaddTask`` in the larger
context of multi-band processing, we will generate the HSC-I & -R band
oadds from HSC engineering test data provided in the ``ci_hsc`` package.
To begin, assuming that the lsst stack has been already set up, we must
set up the ``obs_subaru`` and ``ci_hsc`` packages.
This defines the environment variable ``$CI_HSC_DIR`` and points at the
location of the package. The raw HSC data live in the ``$CI_HSC_DIR/raw``
directory. To begin assembling the coadds, we must first

  - processCcd
    process the individual ccds in $CI_HSC_RAW to produce calibrated exposures
  - makeSkyMap
    create a skymap that covers the area of the sky present in the raw exposures
  - makeCoaddTempExp
    warp the individual calibrated exposures to the tangent plane of the coadd

We can perform all of these steps by running

.. code-block:: none

   $CI_HSC_DIR scons warp-903986 warp-904014 warp-903990 warp-904010 warp-903988

This will produce warped ``coaddTempExps`` for each visit. To coadd the
warped data, we call ``assembleCoadd.py`` as follows:

.. code-block:: none

   assembleCoadd.py --compareWarpCoadd $CI_HSC_DIR/DATA --id patch=5,4 tract=0 filter=HSC-I \
   --selectId visit=903986 ccd=16 --selectId visit=903986 ccd=22 --selectId visit=903986 ccd=23 \
   --selectId visit=903986 ccd=100 --selectId visit=904014 ccd=1 --selectId visit=904014 ccd=6 \
   --selectId visit=904014 ccd=12 --selectId visit=903990 ccd=18 --selectId visit=903990 ccd=25 \
   --selectId visit=904010 ccd=4 --selectId visit=904010 ccd=10 --selectId visit=904010 ccd=100 \
   --selectId visit=903988 ccd=16 --selectId visit=903988 ccd=17 --selectId visit=903988 ccd=23 \
   --selectId visit=903988 ccd=24

This will process the HSC-I band data. The results are written in
``$CI_HSC_DIR/DATA/deepCoadd-results/HSC-I``.

.. _lsst.pipe.tasks.assembleCoadd.CompareWarpAssembleCoaddTask-debug:

Debugging
=========

The `lsst.pipe.base.cmdLineTask.CmdLineTask` interface supports a
flag ``-d`` to import ``debug.py`` from your ``PYTHONPATH``; see
``baseDebug`` for more about ``debug.py`` files.

This task supports the following debug variables:

- ``saveCountIm``
    If True then save the Epoch Count Image as a fits file in the `figPath`
- ``figPath``
    Path to save the debug fits images and figures

For example, put something like:

.. code-block:: python

   import lsstDebug
   def DebugInfo(name):
       di = lsstDebug.getInfo(name)
       if name == "lsst.pipe.tasks.assembleCoadd":
           di.saveCountIm = True
           di.figPath = "/desired/path/to/debugging/output/images"
       return di
   lsstDebug.Info = DebugInfo

into your ``debug.py`` file and run ``assemebleCoadd.py`` with the
``--debug`` flag. Some subtasks may have their own debug variables;
see individual Task documentation.
