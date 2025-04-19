.. lsst-task-topic:: lsst.pipe.tasks.calibrateImage.CalibrateImageTask

##################
CalibrateImageTask
##################

`~lsst.pipe.tasks.calibrateImage.CalibrateImageTask` performs "single frame processing" on one (single *visit*) or two (two *snap* visit) post- :ref:`Instrument Signature Removal <lsst.ip.isr>` single detector exposure (``post_isr_image``).
This involves merging two *snaps* (if provided) into one *visit* exposure, repairing cosmic rays and defects, detecting and measuring sources on the exposure to make an initial estimation of the point spread function (PSF), using that same catalog of psf stars to compute the astrometric calibration, then re-doing detection and measurement with the fitted PSF to compute the photometric calibrations, and computing summary statistics of the exposure and measured catalog.
Its primary outputs are a calibrated, background-subtracted exposure (``preliminary_visit_image``, pixel values in nanojansky) and catalog (``single_visit_star_unstandardized``) of bright, well-measured point-like sources that were used as inputs to calibration and that are suitable for downstream use (for example as kernel candidates in difference imaging).
This task replaces the two older tasks `~lsst.pipe.tasks.characterizeImage.CharacterizeImageTask` (roughly: repair/estimate PSF/aperture correct) and `~lsst.pipe.tasks.calibrate.CalibrateTask` (roughly: detect/measure/astrometry/photometry).

.. _lsst.pipe.tasks.calibrateImage.CalibrateImageTask-summary:

Processing summary
==================

`~lsst.pipe.tasks.calibrateImage.CalibrateImageTask` runs this sequence of operations:

#. If two *snap* exposures are input, :py:class:`combine <lsst.pipe.tasks.snapCombine.SnapCombineTask>` them into one *visit* exposure.

#. Find stars to estimate the PSF, via two passes of repair/detect/measure/estimate PSF:

   #. Install a simple Gaussian PSF in the exposure and subtract an initial estimate of the image background.

   #. Perform cosmic ray :py:class:`repair <lsst.pipe.tasks.repair.RepairTask>`, source :py:class:`detection <lsst.meas.algorithms.SourceDetectionTask>` to :math:`S/N >= 50`, and a minimal list of :py:class:`measurement plugins <lsst.meas.base.sfm.SingleFrameMeasurementTask>` to compute a first :py:class:`estimate of the PSF <lsst.pipe.tasks.measurePsf.MeasurePsfTask>`.

   #. Install an updated Gaussian PSF representation of that first PSF estimate (to reduce noise and help with convergence) and re-run repair/detect/measure/estimate PSF as above.

   #. Use that final fitted PSF to redo repair and measurement (hopefully with all cosmic rays now removed), resulting in the optional ``single_visit_psf_star`` and ``single_visit_psf_star_footprints`` output catalogs. Note that these catalogs do not have sky coordinates or calibrated fluxes.

#. Compute an :py:class:`aperture correction <lsst.meas.algorithms.MeasureApCorrTask>` for the exposure using the final catalog measured after the PSF fit.

#. Perform astrometric fit

   #. Use sources flagged as ``calib_psf_candidate`` from the PSF model catalog above, ``single_visit_psf_star_footprints``, in the astrometric calibration.

   #. Fit the :py:class:`astrometry <lsst.meas.astrom.AstrometryTask>` to a reference catalog using an :py:class:`affine WCS fitter <lsst.meas.astrom.FitAffineWcsTask>` that requires a reasonable model of the :ref:`camera geometry <section_CameraGeom_Overview>`, to produce a `SkyWcs`_ for the exposure and compute on-sky coordinates for the catalog of stars. The star/refcat matches used in the astrometric fit is saved as the optional ``initial_astrometry_match_detector`` catalog.

#. Find stars to use as potential photometric calibration sources:

   #. Detect sources with a peak :math:`S/N >= 10`.

   #. For the detected sources, :py:class:`deblend <lsst.meas.deblender.SourceDeblendTask>`, :py:class:`measure <lsst.meas.base.sfm.SingleFrameMeasurementTask>`, aperture correct, and set flags based on blendedness, footprint size, and other properties.

   #. Select non-"bad" flagged, unresolved, :math:`S/N >= 10` sources to pass to the subsequent calibration steps and to be saved as the ``single_visit_star_unstandardized`` and ``single_visit_psf_star_footprints`` output catalogs. Note that these catalogs do not have sky coordinates or calibrated fluxes: those are computed at a later step.

#. Match the list of stars from the two steps above, to propagate flags (e.g. ``calib_psf_candidate``, ``calib_psf_used``, and ``calib_astrometry_used``) from the psf/astrometry stars catalog into the second, primary output catalog.

#. The steps above perform several rounds of background fitting, which together are saved as the ``preliminary_visit_image_background`` output; this saved background has been calibrated to be in the same nJy units as the ``preliminary_visit_image`` output exposure.

#. Fit the :py:class:`photometry <lsst.pipe.tasks.photoCal.PhotoCalTask>` to a reference catalog, to produce a `PhotoCalib`_ for the exposure and calibrate both the image and catalog of stars to have pixels and fluxes respectively in nanojansky. Note that this results in the output exposure having a `PhotoCalib`_ identically 1; the applied `PhotoCalib`_ is saved as the ``initial_photoCalib_detector`` output. The star/refcat matches used in the photometric fit is saved as the optional ``initial_photometry_match_detector`` catalog.

#. Finally, the measurements and fits performed above are combined into a variety of summary statistics which are attached to the exposure, which is saved as the ``preliminary_visit_image`` output.

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

.. _lsst.pipe.tasks.calibrateImage.CalibrateImageTask-indepth:

In Depth
========

.. _lsst.pipe.tasks.calibrateImage.CalibrateImageTask-psf-crossmatch:

Catalog cross-matching
----------------------

The catalog of calibrated stars (``single_visit_star_unstandardized``) produced by this task has different source ids than the catalog of stars that were detected for PSF determination (``single_visit_psf_star``), because those subtasks used different detection configurations.
The stars catalog contains a ``psf_id`` field, which if non-zero is the source id of the corresponding record in the psf stars catalog.
This also applies to the reference/source match catalogs for astrometry (``initial_astrometry_match_detector``) and photometry (``initial_photometry_match_detector``).
We use the psf star catalog for the astrometry fit, so the ``src_id`` values in the astrometry match catalog refer to the psf stars, not the calibrated stars.

For how to find the matching objects in the respective `astropy Table`_ output catalogs, see this example:

.. code-block:: python
    :name: psf-crossmatch-example

    import esutil

    matches = esutil.numpy_util.match(psf_stars["id"], stars["psf_id"])
    # psf_stars[matches[0]] and stars[matched[1]] are the matching objects.

    matches = esutil.numpy_util.match(astrometry_matches["src_id"], photometry_matches["src_psf_id"])
    # astrometry_matches[matches[0]] and photometry_matches[matched[1]] are the matching objects.

.. warning::
    Only boolean index arrays are supported on `lsst.afw.table` Catalogs, so you cannot use the matched index arrays shown in the examples above with the ``astrometry_matches`` or ``photometry_matches`` catalogs directly.
    You can instead access by column first, or convert the table to astropy:

    .. code-block:: python
        :name: psf-crossmatch-warning

        # raises error
        astrometry_matches[matches[0]]

        # column-first access
        ra = astrometry_matches["src_ra"][matches[0]]
        dec = astrometry_matches["src_dec"][matches[0]]

        # astropy conversion
        astrometry_matches_astropy = astrometry_matches.asAstropy()
        astrometry_matches_astropy[matches[0]]  # all columns

.. _Mask: http://doxygen.lsst.codes/stack/doxygen/x_masterDoxyDoc/classlsst_1_1afw_1_1image_1_1_mask.html#details
.. _SkyWcs: http://doxygen.lsst.codes/stack/doxygen/x_masterDoxyDoc/classlsst_1_1afw_1_1geom_1_1_sky_wcs.html#details
.. _PhotoCalib: http://doxygen.lsst.codes/stack/doxygen/x_masterDoxyDoc/classlsst_1_1afw_1_1image_1_1_photo_calib.html#details
.. _astropy Table: https://docs.astropy.org/en/latest/table/index.html
