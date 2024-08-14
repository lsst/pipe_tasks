.. lsst-task-topic:: lsst.pipe.tasks.make_psf_matched_warp.MakePsfMatchedWarpTask

######################
MakePsfMatchedWarpTask
######################

Convolve a direct warp by a kernel to produce a PSF-matched warp, whose PSF matches a desired target PSF.

This task separates the warp into a set of non-overlapping polygons corresponding to each detector.
The PSF-matching is done on each detector separately.
The subtask `~lsst.ip.diffim.modelPsfMatch.ModelPsfMatchTask` is responsible for the PSF-Matching, and its config is accessed via `config.psfMatch`.

The optimal configuration depends on aspects of dataset: the pixel scale, average PSF FWHM and dimensions of the PSF kernel.
These configs include the requested model PSF, the matching kernel size, padding of the science PSF thumbnail and spatial sampling frequency of the PSF.

.. _lsst.pipe.tasks.make_psf_matched_warp.MakePsfMatchedWarpTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.pipe.tasks.make_psf_matched_warp.MakePsfMatchedWarpTask

.. _lsst.pipe.tasks.make_psf_matched_warp.MakePsfMatchedWarpTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.pipe.tasks.make_psf_matched_warp.MakePsfMatchedWarpTask

.. _lsst.pipe.tasks.make_psf_matched_warp.MakePsfMatchedWarpTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.pipe.tasks.make_psf_matched_warp.MakePsfMatchedWarpTask

In Depth
========

Config Guidelines
*****************

The user must specify the size of the model PSF to which to match by setting `config.modelPsf.defaultFwhm` in units of pixels.
The appropriate values depend on science case.
In general, for a set of input images, this config should equal the FWHM of the visit with the worst seeing.
The smallest it should be set to is the median FWHM.
The defaults of the other config options offer a reasonable starting point.

The following list presents the most common problems that arise from a misconfigured `~lsst.ip.diffim.modelPsfMatch.ModelPsfMatchTask` and corresponding solutions.
All assume the default Alard-Lupton kernel, with configs accessed via `config.psfMatch.kernel['AL']`.
Each item in the list is formatted as Problem, Explanation. *Solution*.

Troubleshooting PSF-Matching Configuration
******************************************

Matched PSFs look boxy
-----------------------

The matching kernel is too small.

Solution
^^^^^^^^

Increase the matching kernel size. For example:

.. code-block:: python

    config.psfMatch.kernel['AL'].kernelSize=27
    # default 21

Note that increasing the kernel size also increases runtime.

Matched PSFs look ugly (dipoles, quadrupoles, donuts)
-----------------------------------------------------

Unable to find good solution for matching kernel.

Solution
^^^^^^^^

Provide the matcher with more data by either increasing the spatial sampling by decreasing the spatial cell size.

.. code-block:: python

    config.psfMatch.kernel['AL'].sizeCellX = 64
    # default 128
    config.psfMatch.kernel['AL'].sizeCellY = 64
    # default 128

- or increasing the padding around the Science PSF, for example:

.. code-block:: python

    config.psfMatch.autoPadPsfTo=1.6  # default 1.4

Increasing `autoPadPsfTo` increases the minimum ratio of input PSF dimensions to the matching kernel dimensions, thus increasing the number of pixels available to fit after convolving the PSF with the matching kernel.
Optionally, for debugging the effects of padding, the level of padding may be manually controlled by setting turning off the automatic padding and setting the number of pixels by which to pad the PSF:

.. code-block:: python

    config.psfMatch.doAutoPadPsf = False
    # default True
    config.psfMatch.padPsfBy = 6
    # pixels. default 0

Ripple Noise Pattern
--------------------

 Matching a large PSF to a smaller PSF produces a telltale noise pattern which looks like ripples or a brain.

Solution
^^^^^^^^

Increase the size of the requested model PSF. For example:

.. code-block:: python

    config.modelPsf.defaultFwhm = 11  # Gaussian sigma in units of pixels.

High frequency (sometimes checkered) noise
------------------------------------------

The matching basis functions are too small.

Solution
^^^^^^^^

Increase the width of the Gaussian basis functions.
For example:

.. code-block:: python

    config.psfMatch.kernel['AL'].alardSigGauss= [1.5, 3.0, 6.0]  # from default [0.7, 1.5, 3.0]
