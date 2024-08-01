.. lsst-task-topic:: lsst.pipe.tasks.make_direct_warp.MakeDirectWarpTask

##################
MakeDirectWarpTask
##################

Warp single visit images (calexps or PVIs) onto a common projection by
performing the following operations:

- Group the single-visit images by visit/run
- For each visit, generate a Warp by calling method @ref run.

.. _lsst.pipe.tasks.make_direct_warp.MakeDirectWarpTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.pipe.tasks.make_direct_warp.MakeDirectWarpTask

.. _lsst.pipe.tasks.make_direct_warp.MakeDirectWarpTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.pipe.tasks.make_direct_warp.MakeDirectWarpTask

.. _lsst.pipe.tasks.make_direct_warp.MakeDirectWarpTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.pipe.tasks.make_direct_warp.MakeDirectWarpTask

In Depth
========

Config Guidelines
*****************

The user must specify the size of the model PSF to
which to match by setting `config.modelPsf.defaultFwhm` in units of pixels.
The appropriate values depends on science case. In general, for a set of
input images, this config should equal the FWHM of the visit with the worst
seeing. The smallest it should be set to is the median FWHM. The defaults
of the other config options offer a reasonable starting point.

The following list presents the most common problems that arise from a
misconfigured `~ip.diffim.modelPsfMatch.ModelPsfMatchTask`
and corresponding solutions. All assume the default Alard-Lupton kernel,
with configs accessed via
`config.warpAndPsfMatch.psfMatch.kernel['AL']`. Each item in the list
is formatted as:
Problem: Explanation. *Solution*

Troublshooting PSF-Matching Configuration
*****************************************

Matched PSFs look boxy
**********************

The matching kernel is too small.

Solution
********

Increase the matching kernel size. For example:

.. code-block:: python

    config.warpAndPsfMatch.psfMatch.kernel['AL'].kernelSize=27
    # default 21

Note that increasing the kernel size also increases runtime.

Matched PSFs look ugly (dipoles, quadropoles, donuts)
*****************************************************

Unable to find good solution for matching kernel.

Solution
********

Provide the matcher with more data by either increasing the spatial sampling by decreasing the spatial cell size.

.. code-block:: python

    config.warpAndPsfMatch.psfMatch.kernel['AL'].sizeCellX = 64
    # default 128
    config.warpAndPsfMatch.psfMatch.kernel['AL'].sizeCellY = 64
    # default 128

- or increasing the padding around the Science PSF, for example:

.. code-block:: python

    config.warpAndPsfMatch.psfMatch.autoPadPsfTo=1.6  # default 1.4

Increasing `autoPadPsfTo` increases the minimum ratio of input PSF
dimensions to the matching kernel dimensions, thus increasing the
number of pixels available to fit after convolving the PSF with the
matching kernel. Optionally, for debugging the effects of padding, the
level of padding may be manually controlled by setting turning off the
automatic padding and setting the number of pixels by which to pad the
PSF:

.. code-block:: python

    config.warpAndPsfMatch.psfMatch.doAutoPadPsf = False
    # default True
    config.warpAndPsfMatch.psfMatch.padPsfBy = 6
    # pixels. default 0

Ripple Noise Pattern
********************

 Matching a large PSF to a smaller PSF produces a telltale noise pattern which looks like ripples or a brain.

Solution
********

Increase the size of the requested model PSF. For example:

.. code-block:: python

    config.modelPsf.defaultFwhm = 11  # Gaussian sigma in units of pixels.

High frequency (sometimes checkered) noise
******************************************

The matching basis functions are too small.

Solution
********

Increase the width of the Gaussian basis functions. For example:

.. code-block:: python

    config.warpAndPsfMatch.psfMatch.kernel['AL'].alardSigGauss=
    [1.5, 3.0, 6.0]  # from default [0.7, 1.5, 3.0]
