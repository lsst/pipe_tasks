.. lsst-task-topic:: lsst.pipe.tasks.calibrateImage.CalibrateImageTask

##################
CalibrateImageTask
##################

`~lsst.pipe.tasks.calibrateImage.CalibrateImageTask` performs "single frame processing" on one (single *visit*) or two (two *snap* visit) post- :ref:`Instrument Signature Removal <lsst.ip.isr>` single detector exposure (``postISRCCD``).
This involves merging two *snaps* (if provided) into one *visit* exposure, repairing cosmic rays and defects, detecting and measuring sources on the exposure to make an initial estimation of the point spread function (PSF), re-doing detection and measurement with that fitted PSF to compute the astrometric and photometric calibrations, and computing summary statistics of the exposure and measured catalog.
Its primary outputs are a calibrated exposure (``initial_pvi``, pixel values in nanojansky) and catalog (``initial_stars_detector``) of bright (S/N >= 10) isolated point-like sources that were used as inputs to calibration and that are suitable for downstream use (for example as kernel candidates in difference imaging).
This task replaces the two older tasks `~lsst.pipe.tasks.characterizeImage.CharacterizeImageTask` (roughly repair/estimate PSF/aperture correct) and `~lsst.pipe.tasks.calibrate.CalibrateTask` (roughly detect/measure/astrometry/photometry).

.. _lsst.pipe.tasks.calibrateImage.CalibrateImageTask-summary:

Processing summary
==================

`~lsst.pipe.tasks.calibrateImage.CalibrateImageTask` runs this sequence of operations:

#. If two *snap* exposures are input, :py:class:`combine <lsst.pipe.tasks.snapCombine.SnapCombineTask>` them into one *visit* exposure.

#. Find stars to estimate the PSF, via two passes of repair/detect/measure/estimate PSF:

   #. Install a simple Gaussian PSF in the exposure and subtract an initial estimate of the image background.

   #. Perform cosmic ray :py:class:`repair <lsst.pipe.tasks.repair.RepairTask>`, source :py:class:`detection <lsst.meas.algorithms.SourceDetectionTask>` to :math:`S/N >= 50`, and a minimal list of :py:class:`measurement plugins <lsst.meas.base.sfm.SingleFrameMeasurementTask>` to compute a first :py:class:`estimate of the PSF <lsst.pipe.tasks.measurePsf.MeasurePsfTask>`.

   #. Install an updated Gaussian PSF representation of that first PSF estimate (to reduce noise and help with convergence) and re-run repair/detect/measure/estimate PSF as above.

   #. Use that final fitted PSF to redo repair and measurement (hopefully with all cosmic rays now removed), resulting in the optional ``initial_psf_stars_detector`` and ``initial_psf_stars_footprints_detector`` output catalogs. Note that these catalogs do not have sky coordinates or calibrated fluxes.

#. Compute an :py:class:`aperture correction <lsst.meas.algorithms.MeasureApCorrTask>` for the exposure using the final catalog measured after the PSF fit.

#. Find stars to use as potential calibration sources:

   #. Detect sources with a peak :math:`S/N >= 10`.

   #. :py:class:`Find potential streaks <lsst.pipe.tasks.maskStreaks.MaskStreaksTask>` on the image from the detection `Mask`_ computed above, and `Mask`_ those streaks on the exposure.

   #. For the detected sources, :py:class:`deblend <lsst.meas.deblender.SourceDeblendTask>`, :py:class:`measure <lsst.meas.base.sfm.SingleFrameMeasurementTask>`, aperture correct, and set flags based on blendedness, footprint size, and other properties.

   #. Select non-"bad" flagged, unresolved, :math:`S/N >= 10`, isolated sources to pass to the subsequent calibration steps and to be saved as the ``initial_stars_detector`` and ``initial_stars_footprints_detector`` output catalogs. Note that these catalogs do not have sky coordinates or calibrated fluxes: those are computed at a later step.

#. Match the list of stars from the two steps above, to propagate flags (e.g. ``calib_psf_candidate``, ``calib_psf_used``) from the psf stars catalog into the second, primary output catalog.

#. The steps above perform several rounds of background fitting, which together are saved as the ``initial_pvi_background`` output.

#. Fit the :py:class:`astrometry <lsst.meas.astrom.AstrometryTask>` to a reference catalog using an :py:class:`affine WCS fitter <lsst.meas.astrom.FitAffineWcsTask>` that requires a reasonable model of the :ref:`camera geometry <section_CameraGeom_Overview>`, to produce a `SkyWcs`_ for the exposure and compute on-sky coordinates for the catalog of stars. The star/refcat matches used in the astrometric fit is saved as the optional ``initial_astrometry_match_detector`` catalog.

#. Fit the :py:class:`photometry <lsst.pipe.tasks.photoCal.PhotoCalTask>` to a reference catalog, to produce a `PhotoCalib`_ for the exposure and calibrate both the image and catalog of stars to have pixels and fluxes respectively in nanojansky. Note that this results in the output exposure having a `PhotoCalib`_ identically 1; the applied `PhotoCalib`_ is saved as the ``initial_photoCalib_detector`` output. The star/refcat matches used in the photometric fit is saved as the optional ``initial_photometry_match_detector`` catalog.

#. Finally, the measurements and fits performed above are combined into a variety of summary statistics which are attached to the exposure, which is saved as the ``initial_pvi`` output.

.. _lsst.pipe.tasks.calibrateImage.CalibrateImageTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.pipe.tasks.calibrateImage.CalibrateImageTask

.. _lsst.pipe.tasks.calibrateImage.CalibrateImageTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.pipe.tasks.calibrateImage.CalibrateImageTask

.. _lsst.pipe.tasks.calibrateImage.CalibrateImageTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.pipe.tasks.calibrateImage.CalibrateImageTask

.. _Mask: http://doxygen.lsst.codes/stack/doxygen/x_masterDoxyDoc/classlsst_1_1afw_1_1image_1_1_mask.html#details
.. _SkyWcs: http://doxygen.lsst.codes/stack/doxygen/x_masterDoxyDoc/classlsst_1_1afw_1_1geom_1_1_sky_wcs.html#details
.. _PhotoCalib: http://doxygen.lsst.codes/stack/doxygen/x_masterDoxyDoc/classlsst_1_1afw_1_1image_1_1_photo_calib.html#details
