.. lsst-task-topic:: lsst.pipe.tasks.healSparseMapping.HealSparsePropertyMapTask

#########################
HealSparsePropertyMapTask
#########################

``HealSparsePropertyMapTask`` generates a set of survey property maps in `healsparse <https://healsparse.readthedocs.io/en/stable>`_ format.
One map is created per map type per tract per band.

The resolution of the maps is set by the resolution configured for :doc:`lsst.pipe.tasks.healSparseMapping.HealSparseInputMapTask` and run during the coadd assembly.

The maps to be run are configured through the "property map registry", and new maps can be defined via the ``lsst.pipe.tasks.healSparseMappingProperties.register_property_map`` decorator.
Maps can do computations with any values that are available via the visit summary datasets.

Each map type can be configured to compute the minimum value at each position (``do_min``, dataset type ``{name}_map_min``); the maximum value (``do_max``, dataset type ``{name}_map_max``); the mean value (``do_mean``, dataset type ``{name}_map_mean``); the weighted mean, using the coadd weights (``do_weighted_mean``, dataset type ``{name}_map_weighted_mean``); and the sum (``do_sum``, dataset type ``{name}_map_sum``).
In each case ``{name}`` refers to the registered name of the map.

Note that the output maps cover the full coadd tract, and are not truncated to the inner tract region.
Truncation to the inner region is performed when tract maps are consoldated in :doc:`lsst.pipe.tasks.healSparseMapping.ConsolidateHealSparsePropertyMapTask`.

Supported Map Types
===================
The following map types are supported, and it is possible to add more by subclassing ``lsst.pipe.tasks.healSparseMappingProperties.BasePropertyMap``.
All values are sampled at the center of each map pixel.
All PSF properties are estimated by realizing the PSF model over a grid of points on each detector and approximating the variation with a second-order Chebyshev polynomial.

- ``exposure_time``: The input visit exposure time, usually with ``do_sum=True``.
- ``psf_size``: The size of the PSF as computed from the determinant radius (pixels).
- ``psf_e1``: The PSF e1 ellipticity.
- ``psf_e2``: The PSF e2 ellipticity.
- ``n_exposure``: The number of exposures, usually with ``do_sum=True``.
- ``psf_maglim``: The magnitude limit for PSF magnitudes.  Can only be used with ``do_weighted_mean=True``.
- ``sky_background`` : The sky background, scaled to nJy.
- ``sky_noise``: The sky noise, scaled to nJy.
- ``dcr_dra``: The shift in the RA position of sources due to differential chromatic refraction (DCR) will be an empirically determined constant multiplied by the object color multiplied by this map.
- ``dcr_ddec``: The shift in the Dec position of the sources due to DCR.
- ``dcr_e1``: The shift in the ellipticity e1 due to DCR will be an empirically determined constant multipled by the object color multiplied by this map.
- ``dcr_e2``: The shift in the ellipticity e2 due to DCR.

.. _lsst.pipe.tasks.healSparseMapping.HealSparsePropertyMapTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.pipe.tasks.healSparseMapping.HealSparsePropertyMapTask

.. _lsst.pipe.tasks.healSparseMapping.HealSparsePropertyMapTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.pipe.tasks.healSparseMapping.HealSparsePropertyMapTask

.. _lsst.pipe.tasks.healSparseMapping.HealSparsePropertyMapTask-fields:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.pipe.tasks.healSparseMapping.HealSparsePropertyMapTask
