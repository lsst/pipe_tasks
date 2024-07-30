# This file is part of pipe_tasks.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np
from lsst.afw.geom import makeWcsPairTransform
from lsst.afw.image import ExposureF, Mask
from lsst.afw.math import Warper
from lsst.coadd.utils import copyGoodPixels
from lsst.geom import Box2D
from lsst.meas.algorithms import CoaddPsf, CoaddPsfConfig, WarpedPsf
from lsst.meas.algorithms.cloughTocher2DInterpolator import (
    CloughTocher2DInterpolateTask,
)
from lsst.meas.base import DetectorVisitIdGeneratorConfig
from lsst.pex.config import (
    ConfigField,
    ConfigurableField,
    Field,
    RangeField,
)
from lsst.pipe.base import (
    NoWorkFound,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.pipe.base.connectionTypes import Input, Output
from lsst.pipe.tasks.coaddBase import makeSkyInfo
from lsst.pipe.tasks.selectImages import PsfWcsSelectImagesTask
from lsst.skymap import BaseSkyMap

from .coaddInputRecorder import CoaddInputRecorderTask

if TYPE_CHECKING:
    from lsst.afw.image import MaskedImage
    from lsst.afw.table import Exposure, ExposureCatalog


__all__ = (
    "MakeDirectWarpConfig",
    "MakeDirectWarpTask",
)


class MakeDirectWarpConnections(
    PipelineTaskConnections,
    dimensions=("tract", "patch", "skymap", "instrument", "visit"),
    defaultTemplates={
        "coaddName": "deep",
        "calexpType": "",
    },
):
    """Connections for MakeWarpTask"""

    calexp_list = Input(
        doc="Input exposures to be interpolated and resampled onto a SkyMap "
            "projection/patch.",
        name="{calexpType}initial_pvi",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
        multiple=True,
        deferLoad=True,
    )
    background_revert_list = Input(
        doc="Background to be reverted (i.e., added back to the calexp). "
        "This connection is used only if doRevertOldBackground=False.",
        name="initial_pvi_background",
        storageClass="Background",
        dimensions=("instrument", "visit", "detector"),
        multiple=True,
    )
    background_apply_list = Input(
        doc="Background to be applied (subtracted from the calexp). "
        "This is used only if doApplyNewBackground=True.",
        name="skyCorr",
        storageClass="Background",
        dimensions=("instrument", "visit", "detector"),
        multiple=True,
    )
    visit_summary = Input(
        doc="Input visit-summary catalog with updated calibration objects.",
        name="finalVisitSummary",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit"),
    )
    sky_map = Input(
        doc="Input definition of geometry/bbox and projection/wcs for warps.",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    # Declare all possible outputs (except noise, which is configurable)
    warp = Output(
        doc="Output direct warped exposure produced by resampling calexps "
            "onto the skyMap patch geometry.",
        name="{coaddName}Coadd_directWarp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "instrument", "visit"),
    )
    masked_fraction_warp = Output(
        doc="Output masked fraction warped exposure.",
        name="{coaddName}Coadd_directWarp_maskedFraction",
        storageClass="ImageF",
        dimensions=("tract", "patch", "skymap", "instrument", "visit"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if not config:
            return

        if not config.doRevertOldBackground:
            del self.background_revert_list
        if not config.doApplyNewBackground:
            del self.background_apply_list

        if not config.doWarpMaskedFraction:
            del self.masked_fraction_warp

        # Dynamically set output connections for noise images, depending on the
        # number of noise realization specified in the config.
        for n in range(config.numberOfNoiseRealizations):
            noise_warp = Output(
                doc=f"Output direct warped noise exposure ({n})",
                name=f"{config.connections.coaddName}Coadd_directWarp_noise{n}",
                # Store it as a MaskedImage to preserve the variance plane.
                storageClass="MaskedImageF",
                dimensions=("tract", "patch", "skymap", "instrument", "visit"),
            )
            setattr(self, f"noise_warp{n}", noise_warp)


class MakeDirectWarpConfig(
    PipelineTaskConfig,
    pipelineConnections=MakeDirectWarpConnections,
):
    """Configuration for the MakeDirectWarpTask.

    The config fields are as similar as possible to the corresponding fields in
    MakeWarpConfig.

    Notes
    -----
    The config fields are in camelCase to match the fields in the earlier
    version of the makeWarp task as closely as possible.
    """

    MAX_NUMBER_OF_NOISE_REALIZATIONS = 3
    """
    numberOfNoiseRealizations is defined as a RangeField to prevent from
    making multiple output connections and blowing up the memory usage by
    accident. An upper bound of 3 is based on the best guess of the maximum
    number of noise realizations that will be used for metadetection.
    """

    numberOfNoiseRealizations = RangeField[int](
        doc="Number of noise realizations to simulate and persist.",
        default=0,
        min=0,
        max=MAX_NUMBER_OF_NOISE_REALIZATIONS,
        inclusiveMax=True,
    )
    seedOffset = Field[int](
        doc="Offset to the seed used for the noise realization. This can be "
            "used to create a different noise realization if the default ones "
            "are catastrophic, or for testing sensitivity to the noise.",
        default=0,
    )
    useMedianVariance = Field[bool](
        doc="Use the median of variance plane in the input calexp to generate "
            "noise realizations? If False, per-pixel variance will be used.",
        default=True,
    )
    doRevertOldBackground = Field[bool](
        doc="Revert the old backgrounds from the `background_revert_list` "
            "connection?",
        default=False,
    )
    doApplyNewBackground = Field[bool](
        doc="Apply the new backgrounds from the `background_apply_list` "
            "connection?",
        default=False,
    )
    useVisitSummaryPsf = Field[bool](
        doc="If True, use the PSF model and aperture corrections from the "
            "'visit_summary' connection to make the warp. If False, use the "
            "PSF model and aperture corrections from the 'calexp' connection.",
        default=True,
    )
    doSelectPreWarp = Field[bool](
        doc="Select ccds before warping?",
        default=True,
    )
    select = ConfigurableField(
        doc="Image selection subtask.",
        target=PsfWcsSelectImagesTask,
    )
    doPreWarpInterpolation = Field[bool](
        doc="Interpolate over bad pixels before warping?",
        default=False,
    )
    preWarpInterpolation = ConfigurableField(
        doc="Interpolation task to use for pre-warping interpolation",
        target=CloughTocher2DInterpolateTask,
    )
    inputRecorder = ConfigurableField(
        doc="Subtask that helps fill CoaddInputs catalogs added to the final "
            "coadd",
        target=CoaddInputRecorderTask,
    )
    includeCalibVar = Field[bool](
        doc="Add photometric calibration variance to warp variance plane?",
        default=False,
    )
    border = Field[int](
        doc="Pad the patch boundary of the warp by these many pixels, so as to allow for PSF-matching later",
        default=256,
    )
    warper = ConfigField(
        doc="Configuration for the warper that warps the image and noise",
        dtype=Warper.ConfigClass,
    )
    doWarpMaskedFraction = Field[bool](
        doc="Warp the masked fraction image?",
        default=False,
    )
    maskedFractionWarper = ConfigField(
        doc="Configuration for the warp that warps the mask fraction image",
        dtype=Warper.ConfigClass,
    )
    coaddPsf = ConfigField(
        doc="Configuration for CoaddPsf",
        dtype=CoaddPsfConfig,
    )
    idGenerator = DetectorVisitIdGeneratorConfig.make_field()

    # Use bgSubtracted and doApplySkyCorr to match the old MakeWarpConfig,
    # but as properties instead of config fields.
    @property
    def bgSubtracted(self) -> bool:
        return not self.doRevertOldBackground

    @bgSubtracted.setter
    def bgSubtracted(self, value: bool) -> None:
        self.doRevertOldBackground = ~value

    @property
    def doApplySkyCorr(self) -> bool:
        return self.doApplyNewBackground

    @doApplySkyCorr.setter
    def doApplySkyCorr(self, value: bool) -> None:
        self.doApplyNewBackground = value

    def setDefaults(self) -> None:
        super().setDefaults()
        self.warper.warpingKernelName = "lanczos3"
        self.warper.cacheSize = 0
        self.maskedFractionWarper.warpingKernelName = "bilinear"


class MakeDirectWarpTask(PipelineTask):
    """Warp single-detector images onto a common projection.

    This task iterates over multiple images (corresponding to different
    detectors) from a single visit that overlap the target patch. Pixels that
    receive no input from any detector are set to NaN in the output image, and
    NO_DATA bit is set in the mask plane.

    This differs from the standard `MakeWarp` Task in the following
    ways:

    1. No selection on ccds at the time of warping. This is done later during
       the coaddition stage.
    2. Interpolate over a set of masked pixels before warping.
    3. Generate an image where each pixel denotes how much of the pixel is
       masked.
    4. Generate multiple noise warps with the same interpolation applied.
    5. No option to produce a PSF-matched warp.
    """

    ConfigClass = MakeDirectWarpConfig
    _DefaultName = "makeDirectWarp"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("inputRecorder")
        self.makeSubtask("preWarpInterpolation")
        if self.config.doSelectPreWarp:
            self.makeSubtask("select")

        self.warper = Warper.fromConfig(self.config.warper)
        if self.config.doWarpMaskedFraction:
            self.maskedFractionWarper = Warper.fromConfig(self.config.maskedFractionWarper)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Docstring inherited.

        # Read in all inputs.
        inputs = butlerQC.get(inputRefs)

        if not inputs["calexp_list"]:
            raise NoWorkFound("No input warps provided for co-addition")

        sky_map = inputs.pop("sky_map")

        quantumDataId = butlerQC.quantum.dataId
        sky_info = makeSkyInfo(
            sky_map,
            tractId=quantumDataId["tract"],
            patchId=quantumDataId["patch"],
        )

        results = self.run(inputs, sky_info)
        butlerQC.put(results, outputRefs)

    def _preselect_inputs(self, inputs, sky_info):
        dataIdList = [ref.dataId for ref in inputs["calexp_list"]]
        visit_summary = inputs["visit_summary"]

        bboxList, wcsList = [], []
        for dataId in dataIdList:
            row = visit_summary.find(dataId["detector"])
            if row is None:
                raise RuntimeError(f"Unexpectedly incomplete visit_summary: {dataId=} is missing.")
            bboxList.append(row.getBBox())
            wcsList.append(row.getWcs())

        cornerPosList = Box2D(sky_info.bbox).getCorners()
        coordList = [sky_info.wcs.pixelToSky(pos) for pos in cornerPosList]

        goodIndices = self.select.run(
            **inputs,
            bboxList=bboxList,
            wcsList=wcsList,
            visitSummary=visit_summary,
            coordList=coordList,
            dataIds=dataIdList,
        )
        inputs = self._filterInputs(indices=goodIndices, inputs=inputs)

        return inputs

    def run(self, inputs, sky_info, **kwargs):
        """Create a Warp dataset from inputs.

        Parameters
        ----------
        inputs : `Mapping`
            Dictionary of input datasets. It must have a list of input calexps
            under the key "calexp_list". Other supported keys are
            "background_revert_list" and "background_apply_list", corresponding
            to the old and the new backgrounds to be reverted and applied to
            the calexps. They must be in the same order as the calexps.
        sky_info : `~lsst.pipe.base.Struct`
            A Struct object containing wcs, bounding box, and other information
            about the patches within the tract.
        visit_summary : `~lsst.afw.table.ExposureCatalog` | None
            Table of visit summary information.  If provided, the visit summary
            information will be used to update the calibration of the input
            exposures.  If None, the input exposures will be used as-is.

        Returns
        -------
        results : `~lsst.pipe.base.Struct`
            A Struct object containing the warped exposure, noise exposure(s),
            and masked fraction image.
        """

        if self.config.doSelectPreWarp:
            inputs = self._preselect_inputs(inputs, sky_info)
            if not inputs["calexp_list"]:
                raise NoWorkFound("No input warps remain after selection for co-addition")

        sky_info.bbox.grow(self.config.border)
        target_bbox, target_wcs = sky_info.bbox, sky_info.wcs

        visit_summary = inputs["visit_summary"] if self.config.useVisitSummaryPsf else None

        # Initialize the objects that will hold the warp.
        final_warp = ExposureF(target_bbox, target_wcs)

        exposures = inputs["calexp_list"]
        background_revert_list = inputs.get("background_revert_list", [None] * len(exposures))
        background_apply_list = inputs.get("background_apply_list", [None] * len(exposures))

        visit_id = exposures[0].dataId["visit"]

        # The warpExposure routine is expensive, and we do not want to call
        # it twice (i.e., a second time for PSF-matched warps). We do not
        # want to hold all the warped exposures in memory at once either.
        # So we create empty exposure(s) to accumulate the warps of each type,
        # and then process each detector serially.
        final_warp = self._prepareEmptyExposure(sky_info)
        final_masked_fraction_warp = self._prepareEmptyExposure(sky_info)
        final_noise_warps = {
            n_noise: self._prepareEmptyExposure(sky_info)
            for n_noise in range(self.config.numberOfNoiseRealizations)
        }

        # We need a few bookkeeping variables only for the science coadd.
        totalGoodPixels = 0
        inputRecorder = self.inputRecorder.makeCoaddTempExpRecorder(
            visit_id,
            len(exposures),
        )

        for index, (calexp_ref, old_background, new_background) in enumerate(
            zip(exposures, background_revert_list, background_apply_list, strict=True)
        ):
            dataId = calexp_ref.dataId
            self.log.debug(
                "Warping exposure %d/%d for id=%s",
                index + 1,
                len(exposures),
                dataId,
            )
            calexp = calexp_ref.get()

            # Generate noise image(s) in-situ.
            seed = self.get_seed_from_data_id(dataId)
            rng = np.random.RandomState(seed + self.config.seedOffset)

            # Generate noise images in-situ.
            noise_calexps = self.make_noise_exposures(calexp, rng)

            # Warp the PSF before processing nad overwriting exposure.
            xyTransform = makeWcsPairTransform(calexp.getWcs(), target_wcs)
            psfWarped = WarpedPsf(calexp.getPsf(), xyTransform)

            warpedExposure = self.process(
                calexp,
                target_wcs,
                self.warper,
                old_background,
                new_background,
                visit_summary,
                destBBox=target_bbox,
            )
            warpedExposure.setPsf(psfWarped)

            if final_warp.photoCalib is not None:
                ratio = (
                    final_warp.photoCalib.getInstFluxAtZeroMagnitude()
                    / warpedExposure.photoCalib.getInstFluxAtZeroMagnitude()
                )
            else:
                ratio = 1

            self.log.debug("Scaling exposure %s by %f", dataId, ratio)
            warpedExposure.maskedImage *= ratio

            # Accumulate the partial warps in an online fashion.
            nGood = copyGoodPixels(
                final_warp.maskedImage,
                warpedExposure.maskedImage,
                final_warp.mask.getPlaneBitMask(["NO_DATA"]),
            )

            if final_warp.photoCalib is None and nGood > 0:
                final_warp.setPhotoCalib(warpedExposure.photoCalib)

            ccdId = self.config.idGenerator.apply(dataId).catalog_id
            inputRecorder.addCalExp(calexp, ccdId, nGood)
            totalGoodPixels += nGood

            if self.config.doWarpMaskedFraction:
                # Obtain the masked fraction exposure and warp it.
                if self.config.doPreWarpInterpolation:
                    badMaskPlanes = self.preWarpInterpolation.config.badMaskPlanes
                else:
                    badMaskPlanes = []
                masked_fraction_exp = self._get_bad_mask(calexp, badMaskPlanes)

                masked_fraction_warp = self.maskedFractionWarper.warpExposure(
                    target_wcs, masked_fraction_exp, destBBox=target_bbox
                )

                copyGoodPixels(
                    final_masked_fraction_warp.maskedImage,
                    masked_fraction_warp.maskedImage,
                    final_masked_fraction_warp.mask.getPlaneBitMask(["NO_DATA"]),
                )

            # Process and accumulate noise images.
            for n_noise in range(self.config.numberOfNoiseRealizations):
                noise_calexp = noise_calexps[n_noise]
                warpedNoise = self.process(
                    noise_calexp,
                    target_wcs,
                    self.warper,
                    old_background,
                    new_background,
                    visit_summary,
                    destBBox=target_bbox,
                )

                warpedNoise.maskedImage *= ratio

                copyGoodPixels(
                    final_noise_warps[n_noise].maskedImage,
                    warpedNoise.maskedImage,
                    final_noise_warps[n_noise].mask.getPlaneBitMask(["NO_DATA"]),
                )

        # If there are no good pixels, return a Struct filled with None.
        if totalGoodPixels == 0:
            results = Struct(
                warp=None,
                masked_fraction_warp=None,
            )
            for noise_index in range(self.config.numberOfNoiseRealizations):
                setattr(results, f"noise_warp{noise_index}", None)

            return results

        # Finish the inputRecorder and add the coaddPsf to the final warp.
        inputRecorder.finish(final_warp, totalGoodPixels)

        coaddPsf = CoaddPsf(
            inputRecorder.coaddInputs.ccds,
            sky_info.wcs,
            self.config.coaddPsf.makeControl(),
        )

        final_warp.setPsf(coaddPsf)
        final_warp.setFilter(calexp.getFilter())
        final_warp.getInfo().setVisitInfo(calexp.getInfo().getVisitInfo())

        results = Struct(
            warp=final_warp,
        )

        if self.config.doWarpMaskedFraction:
            results.masked_fraction_warp = final_masked_fraction_warp.image

        for noise_index, noise_exposure in final_noise_warps.items():
            setattr(results, f"noise_warp{noise_index}", noise_exposure.maskedImage)

        return results

    def process(
        self,
        exposure,
        target_wcs,
        warper,
        old_background=None,
        new_background=None,
        visit_summary=None,
        maxBBox=None,
        destBBox=None,
    ):
        """Process an exposure.

        There are three processing steps that are applied to the input:

            1. Interpolate over bad pixels before warping.
            2. Apply all calibrations from visit_summary to the exposure.
            3. Warp the exposure to the target coordinate system.

        Parameters
        ----------
        exposure : `~lsst.afw.image.Exposure`
            The input exposure to be processed.
        target_wcs : `~lsst.afw.geom.SkyWcs`
            The WCS of the target patch.
        warper : `~lsst.afw.math.Warper`
            The warper to use for warping the input exposure.
        old_background : `~lsst.afw.image.Background` | None
            The old background to be added back into the calexp.
        new_background : `~lsst.afw.image.Background` | None
            The new background to be subtracted from the calexp.
        visit_summary : `~lsst.afw.table.ExposureCatalog` | None
            Table of visit summary information.  If not None, the visit_summary
            information will be used to update the calibration of the input
            exposures. Otherwise, the input exposures will be used as-is.
        maxBBox : `~lsst.geom.Box2I` | None
            Maximum bounding box of the warped exposure. If None, this is
            determined automatically.
        destBBox : `~lsst.geom.Box2I` | None
            Exact bounding box of the warped exposure. If None, this is
            determined automatically.

        Returns
        -------
        warped_exposure : `~lsst.afw.image.Exposure`
            The processed and warped exposure.
        """

        if self.config.doPreWarpInterpolation:
            self.preWarpInterpolation.run(exposure.maskedImage)

        self._apply_all_calibrations(
            exposure,
            old_background,
            new_background,
            logger=self.log,
            visit_summary=visit_summary,
            includeScaleUncertainty=self.config.includeCalibVar,
        )
        with self.timer("warp"):
            warped_exposure = warper.warpExposure(
                target_wcs,
                exposure,
                maxBBox=maxBBox,
                destBBox=destBBox,
            )

        # Potentially a post-warp interpolation here? Relies on DM-38630.

        return warped_exposure

    @staticmethod
    def _filterInputs(indices, inputs):
        """Filter inputs by their indices.

        This method down-selects the list entries in ``inputs`` dictionary by
        keeping only those items in the lists that correspond to ``indices``.
        This is intended to select input visits that go into the warps.

        Parameters
        ----------
        indices : `list` [`int`]
        inputs : `dict`
            A dictionary of input connections to be passed to run.

        Returns
        -------
        inputs : `dict`
            Task inputs with their lists filtered by indices.
        """
        for key in inputs.keys():
            # Only down-select on list inputs
            if isinstance(inputs[key], list):
                inputs[key] = [inputs[key][ind] for ind in indices]

        return inputs

    def _apply_all_calibrations(
        self,
        exp: Exposure,
        old_background,
        new_background,
        logger,
        visit_summary: ExposureCatalog | None = None,
        includeScaleUncertainty: bool = False,
    ) -> None:
        """Apply all of the calibrations from visit_summary to the exposure.

        Specifically, this method updates the following (if available) to the
        input exposure ``exp`` in place from ``visit_summary``:

        - Aperture correction map
        - Photometric calibration
        - PSF
        - WCS

        Parameters
        ----------
        exp : `~lsst.afw.image.Exposure`
            Exposure to be updated.
        old_background : `~lsst.afw.image.Exposure`
            Exposure corresponding to the old background, to be added back.
        new_background : `~lsst.afw.image.Exposure`
            Exposure corresponding to the new background, to be subtracted.
        logger : `logging.Logger`
            Logger object from the caller Task to write the logs onto.
        visit_summary : `~lsst.afw.table.ExposureCatalog` | None
            Table of visit summary information.  If not None, the visit summary
            information will be used to update the calibration of the input
            exposures. Otherwise, the input exposures will be used as-is.
        includeScaleUncertainty : bool
            Whether to include the uncertainty on the calibration in the
            resulting variance? Passed onto the `calibrateImage` method of the
            PhotoCalib object attached to ``exp``.

        Raises
        ------
        RuntimeError
            Raised if ``visit_summary`` is provided but does not contain a
            record corresponding to ``exp``.
        """
        if old_background:
            exp.maskedImage += old_background.getImage()

        if self.config.useVisitSummaryPsf:
            detector = exp.info.getDetector().getId()
            row = visit_summary.find(detector)

            if row is None:
                raise RuntimeError(f"Unexpectedly incomplete visit_summary: {detector=} is missing.")

            if photo_calib := row.getPhotoCalib():
                exp.setPhotoCalib(photo_calib)
            else:
                logger.warning(
                    "No photometric calibration found in visit summary for detector = %s.",
                    detector,
                )

            if wcs := row.getWcs():
                exp.setWcs(wcs)
            else:
                logger.warning("No WCS found in visit summary for detector = %s.", detector)

            if psf := row.getPsf():
                exp.setPsf(psf)
            else:
                logger.warning("No PSF found in visit summary for detector = %s.", detector)

            if apcorr_map := row.getApCorrMap():
                exp.setApCorrMap(apcorr_map)
            else:
                logger.warning(
                    "No aperture correction map found in visit summary for detector = %s.",
                    detector,
                )

        if new_background:
            exp.maskedImage -= new_background.getImage()

        # Calibrate the (masked) image.
        # This should likely happen even if visit_summary is None.
        photo_calib = exp.photoCalib
        exp.maskedImage = photo_calib.calibrateImage(
            exp.maskedImage, includeScaleUncertainty=includeScaleUncertainty
        )
        exp.maskedImage /= photo_calib.getCalibrationMean()

    # This method is copied from makeWarp.py
    @classmethod
    def _prepareEmptyExposure(cls, sky_info):
        """Produce an empty exposure for a given patch.

        Parameters
        ----------
        sky_info : `lsst.pipe.base.Struct`
            Struct from `~lsst.pipe.base.coaddBase.makeSkyInfo` with
            geometric information about the patch.

        Returns
        -------
        exp : `lsst.afw.image.exposure.ExposureF`
            An empty exposure for a given patch.
        """
        exp = ExposureF(sky_info.bbox, sky_info.wcs)
        exp.getMaskedImage().set(np.nan, Mask.getPlaneBitMask("NO_DATA"), np.inf)
        return exp

    @staticmethod
    def compute_median_variance(mi: MaskedImage) -> float:
        """Compute the median variance across the good pixels of a MaskedImage.

        Parameters
        ----------
        mi : `~lsst.afw.image.MaskedImage`
            The input image on which to compute the median variance.

        Returns
        -------
        median_variance : `float`
            Median variance of the input calexp.
        """
        # Shouldn't this exclude pixels that are masked, to be safe?
        # This is implemented as it was in descwl_coadd.
        return np.median(mi.variance.array[np.isfinite(mi.variance.array) & np.isfinite(mi.image.array)])

    def get_seed_from_data_id(self, data_id) -> int:
        """Get a seed value given a data_id.

        This method generates a unique, reproducible pseudo-random number for
        a data id. This is not affected by ordering of the input, or what
        set of visits, ccds etc. are given.

        This is implemented as a public method, so that simulations that
        don't necessary deal with the middleware can mock up a ``data_id``
        instance, or override this method with a different one to obtain a
        seed value consistent with the pipeline task.

        Parameters
        ----------
        data_id : `~lsst.daf.butler.DataCoordinate`
            Data identifier dictionary.

        Returns
        -------
        seed : `int`
            A unique seed for this data_id to seed a random number generator.
        """
        return self.config.idGenerator.apply(data_id).catalog_id

    def make_noise_exposures(self, calexp: ExposureF, rng) -> dict[int, ExposureF]:
        """Make pure noise realizations based on ``calexp``.

        Parameters
        ----------
        calexp : `~lsst.afw.image.ExposureF`
            The input exposure on which to base the noise realizations.
        rng : `np.random.RandomState`
            Random number generator to use for the noise realizations.

        Returns
        -------
        noise_calexps : `dict` [`int`, `~lsst.afw.image.ExposureF`]
            A mapping of integers ranging from 0 up to
            config.numberOfNoiseRealizations to the corresponding
            noise realization exposures.
        """
        noise_calexps = {}

        # If no noise exposures are requested, return the empty dictionary
        # without any further computations.
        if self.config.numberOfNoiseRealizations == 0:
            return noise_calexps

        if self.config.useMedianVariance:
            variance = self.compute_median_variance(calexp.maskedImage)
        else:
            variance = calexp.variance.array

        for n_noise in range(self.config.numberOfNoiseRealizations):
            noise_calexp = calexp.clone()
            noise_calexp.image.array[:, :] = rng.normal(
                scale=np.sqrt(variance),
                size=noise_calexp.image.array.shape,
            )
            noise_calexp.variance.array[:, :] = variance
            noise_calexps[n_noise] = noise_calexp

        return noise_calexps

    @classmethod
    def _get_bad_mask(cls, exp: ExposureF, badMaskPlanes: Iterable[str]) -> ExposureF:
        """Get an Exposure of bad mask

        Parameters
        ----------
        exp: `lsst.afw.image.Exposure`
            The exposure data.
        badMaskPlanes: `list` [`str`]
            List of mask planes to be considered as bad.

        Returns
        -------
        bad_mask: `~lsst.afw.image.Exposure`
            An Exposure with boolean array with True if inverse variance <= 0
            or if any of the badMaskPlanes bits are set, and False otherwise.
        """

        bad_mask = exp.clone()

        var = exp.variance.array
        mask = exp.mask.array

        bitMask = exp.mask.getPlaneBitMask(badMaskPlanes)

        bad_mask.image.array[:, :] = (var < 0) | np.isinf(var) | ((mask & bitMask) != 0)

        bad_mask.variance.array *= 0.0

        return bad_mask
