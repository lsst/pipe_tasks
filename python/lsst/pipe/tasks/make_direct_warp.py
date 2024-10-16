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

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING, Iterable

import numpy as np
from lsst.afw.geom import makeWcsPairTransform
from lsst.afw.image import ExposureF, Mask, PhotoCalib
from lsst.afw.math import BackgroundList, Warper
from lsst.coadd.utils import copyGoodPixels
from lsst.daf.butler import DataCoordinate, DeferredDatasetHandle
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
    from lsst.afw.table import ExposureCatalog


__all__ = (
    "MakeDirectWarpConfig",
    "MakeDirectWarpTask",
)


@dataclasses.dataclass
class WarpDetectorInputs:
    """Inputs passed to `MakeDirectWarpTask.run` for a single detector.
    """

    exposure_or_handle: ExposureF | DeferredDatasetHandle
    """Detector image with initial calibration objects, or a deferred-load
    handle for one.
    """

    data_id: DataCoordinate
    """Butler data ID for this detector."""

    background_revert: BackgroundList | TrivialBackgroundList | None = None
    """Background model to restore in (i.e. add to) the image."""

    background_apply: BackgroundList | TrivialBackgroundList | None = None
    """Background model to apply to (i.e. subtract from) the image."""

    initial_photo_calib: PhotoCalib | None = None
    """Initial photometric calibration that was already applied to the image."""

    @property
    def exposure(self) -> ExposureF:
        if isinstance(self.exposure_or_handle, DeferredDatasetHandle):
            self.exposure_or_handle = self.exposure_or_handle.get()
        return self.exposure_or_handle


class TrivialBackgroundList:
    """Trivial class that quacks like BackgroundList and gives 0 as image."""

    @classmethod
    def getImage(cls) -> float:
        return 0.0


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
    initial_photo_calib_list = Input(
        doc="Initial photometric calibration that was already applied to "
            "calexp_list images, to be removed prior to applying the final "
            "photometric calibration in visit_summary.",
        name="initial_photoCalib_detector",
        storageClass="PhotoCalib",
        dimensions=("instrument", "visit", "detector"),
        multiple=True,
        deferLoad=False,
        minimum=0,
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

        # initial photo calib connection is not conditional on configs; instead
        # we look for them regardless, and check that this is consistent with
        # configs in adjustQuantum, so we have a better chance of warning when
        # the data repo looks inconsistent with the configuration.

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

    def adjustQuantum(self, inputs, outputs, label, data_id):
        # Delegate to super first so it can raise NoWorkFound when appropriate.
        results = super().adjustQuantum(inputs, outputs, label, data_id)
        if self.config.removeInitialPhotoCalib and not inputs["initial_photo_calib_list"]:
            self.log.warning(
                "Dropping %s quantum %s because initial photo calibs are needed and none were present; "
                "this may be an upstream partial-outputs error on much of a entire visit (which is why this "
                "is not an error), but it may mean that 'config.removeInitialPhotoCalib' should be False.",
                label, data_id,
            )
            raise NoWorkFound("No initial photo calibs.")
        elif not self.config.removeInitialPhotoCalib and inputs["initial_photo_calib_list"]:
            self.log.warning(
                "Input collections have initial photo calib datasets but "
                "'config.removeInitialPhotoCalib=False'.  This is either a very unusual collection "
                "search path or (more likely) a bad configuration.  Not that this config option should "
                "be true when using images produced by CalibrateImageTask."
            )
        return results


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
    removeInitialPhotoCalib = Field[bool](
        doc=("Expect an initial photo calib input to be present, and use it to restore the image ",
             "to instrumental units before applying the final photometric calibration."),
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
        self.doRevertOldBackground = not value

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

        inputs: Mapping[int, WarpDetectorInputs] = {}  # Detector ID -> WarpDetectorInputs
        for handle in butlerQC.get(inputRefs.calexp_list):
            inputs[handle.dataId["detector"]] = WarpDetectorInputs(handle, data_id=handle.dataId)
        if not inputs:
            raise NoWorkFound("No input warps provided for co-addition")
        # Add backgrounds to the inputs struct, being careful not to assume
        # they're present for the same detectors we got image handles for, in
        # case of upstream errors.
        for ref in getattr(inputRefs, "background_revert_list", []):
            inputs[ref.dataId["detector"]].background_revert = butlerQC.get(ref)
        for ref in getattr(inputRefs, "background_apply_list", []):
            inputs[ref.dataId["detector"]].background_apply = butlerQC.get(ref)
        for ref in getattr(inputRefs, "initial_photo_calib_list", []):
            inputs[ref.dataId["detector"]].initial_photo_calib = butlerQC.get(ref)

        visit_summary = butlerQC.get(inputRefs.visit_summary)

        sky_map = butlerQC.get(inputRefs.sky_map)

        quantumDataId = butlerQC.quantum.dataId
        sky_info = makeSkyInfo(
            sky_map,
            tractId=quantumDataId["tract"],
            patchId=quantumDataId["patch"],
        )

        results = self.run(inputs, sky_info, visit_summary)
        butlerQC.put(results, outputRefs)

    def _preselect_inputs(self, inputs, sky_info, visit_summary):
        """Filter the list of inputs via a 'select' subtask.

        Parameters
        ----------
        inputs : ``~collections.abc.Mapping` [ `int`, `WarpDetectorInputs` ]
            Per-detector input structs.
        sky_info : `lsst.pipe.base.Struct`
            Structure with information about the tract and patch.
        visit_summary : `lsst.afw.table.ExposureRecord`
            Record with updated calibratin information for the full visit.

        Returns
        -------
        filtered_inputs : `dict` [ `int`, `WarpDetectorInputs` ]
            Like ``inputs``, with rejected detectors dropped.
        """
        data_id_list, bbox_list, wcs_list = [], [], []
        for detector_id, detector_inputs in inputs.items():
            row = visit_summary.find(detector_id)
            if row is None:
                raise RuntimeError(
                    f"Unexpectedly incomplete visit_summary: {detector_inputs.data_id} is missing."
                )
            data_id_list.append(detector_inputs.data_id)
            bbox_list.append(row.getBBox())
            wcs_list.append(row.getWcs())

        cornerPosList = Box2D(sky_info.bbox).getCorners()
        coordList = [sky_info.wcs.pixelToSky(pos) for pos in cornerPosList]

        good_indices = self.select.run(
            wcsList=wcs_list,
            bboxList=bbox_list,
            coordList=coordList,
            visitSummary=visit_summary,
            dataIds=data_id_list,
        )
        detector_ids = list(inputs.keys())
        good_detector_ids = [detector_ids[i] for i in good_indices]
        return {detector_id: inputs[detector_id] for detector_id in good_detector_ids}

    def run(self, inputs: Mapping[int, WarpDetectorInputs], sky_info, visit_summary):
        """Create a Warp dataset from inputs.

        Parameters
        ----------
        inputs : `~collections.abc.Mapping` [ `int`, `WarpDetectorInputs` ]
            Dictionary of per-detector input structs.
        sky_info : `lsst.pipe.base.Struct`
            Structure with information about the tract and patch.
        visit_summary : `lsst.afw.table.ExposureRecord`
            Record with updated calibratin information for the full visit.

        Returns
        -------
        results : `~lsst.pipe.base.Struct`
            A Struct object containing the warped exposure, noise exposure(s),
            and masked fraction image.
        """

        if self.config.doSelectPreWarp:
            inputs = self._preselect_inputs(inputs, sky_info, visit_summary)
            if not inputs:
                raise NoWorkFound("No input warps remain after selection for co-addition")

        sky_info.bbox.grow(self.config.border)
        target_bbox, target_wcs = sky_info.bbox, sky_info.wcs

        # Initialize the objects that will hold the warp.
        final_warp = ExposureF(target_bbox, target_wcs)
        for _, warp_detector_input in inputs.items():
            visit_id = warp_detector_input.data_id["visit"]
            break

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
            len(inputs),
        )

        for index, detector_inputs in enumerate(inputs.values()):
            self.log.debug(
                "Warping exposure %d/%d for id=%s",
                index + 1,
                len(inputs),
                detector_inputs.data_id,
            )

            # Generate noise image(s) in-situ.
            seed = self.get_seed_from_data_id(detector_inputs.data_id)
            rng = np.random.RandomState(seed + self.config.seedOffset)

            # Generate noise images in-situ.
            noise_calexps = self.make_noise_exposures(detector_inputs.exposure, rng)

            warpedExposure = self.process(
                detector_inputs,
                target_wcs,
                self.warper,
                visit_summary,
                destBBox=target_bbox,
            )

            if warpedExposure is None:
                self.log.debug("Skipping exposure %s because it could not be warped.", detector_inputs.data_id)
                continue

            if final_warp.photoCalib is not None:
                ratio = (
                    final_warp.photoCalib.getInstFluxAtZeroMagnitude()
                    / warpedExposure.photoCalib.getInstFluxAtZeroMagnitude()
                )
            else:
                ratio = 1

            self.log.debug("Scaling exposure %s by %f", detector_inputs.data_id, ratio)
            warpedExposure.maskedImage *= ratio

            # Accumulate the partial warps in an online fashion.
            nGood = copyGoodPixels(
                final_warp.maskedImage,
                warpedExposure.maskedImage,
                final_warp.mask.getPlaneBitMask(["NO_DATA"]),
            )

            if final_warp.photoCalib is None and nGood > 0:
                final_warp.setPhotoCalib(warpedExposure.photoCalib)

            ccdId = self.config.idGenerator.apply(detector_inputs.data_id).catalog_id
            inputRecorder.addCalExp(detector_inputs.exposure, ccdId, nGood)
            totalGoodPixels += nGood

            if self.config.doWarpMaskedFraction:
                # Obtain the masked fraction exposure and warp it.
                if self.config.doPreWarpInterpolation:
                    badMaskPlanes = self.preWarpInterpolation.config.badMaskPlanes
                else:
                    badMaskPlanes = []
                masked_fraction_exp = self._get_bad_mask(detector_inputs.exposure, badMaskPlanes)

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
                noise_pseudo_inputs = dataclasses.replace(
                    detector_inputs,
                    exposure_or_handle=noise_calexp,
                    background_revert=TrivialBackgroundList(),
                    background_apply=TrivialBackgroundList(),
                    initial_photo_calib=PhotoCalib(calibrationMean=1.0),  # TODO: is this correct?
                )
                warpedNoise = self.process(
                    noise_pseudo_inputs,
                    target_wcs,
                    self.warper,
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

        for _, warp_detector_input in inputs.items():
            final_warp.setFilter(warp_detector_input.exposure.getFilter())
            final_warp.getInfo().setVisitInfo(warp_detector_input.exposure.getInfo().getVisitInfo())
            break

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
        detector_inputs,
        target_wcs,
        warper,
        visit_summary=None,
        maxBBox=None,
        destBBox=None,
    ) -> ExposureF | None:
        """Process an exposure.

        There are three processing steps that are applied to the input:

            1. Interpolate over bad pixels before warping.
            2. Apply all calibrations from visit_summary to the exposure.
            3. Warp the exposure to the target coordinate system.

        Parameters
        ----------
        detector_inputs : `WarpDetectorInputs`
            The input exposure to be processed, along with any other
            per-detector modifications.
        target_wcs : `~lsst.afw.geom.SkyWcs`
            The WCS of the target patch.
        warper : `~lsst.afw.math.Warper`
            The warper to use for warping the input exposure.
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
        warped_exposure : `~lsst.afw.image.Exposure` | None
            The processed and warped exposure, if all the calibrations could be
            applied successfully. Otherwise, None.
        """

        if self.config.doPreWarpInterpolation:
            self.preWarpInterpolation.run(detector_inputs.exposure.maskedImage)

        success = self._apply_all_calibrations(
            detector_inputs,
            visit_summary=visit_summary,
            includeScaleUncertainty=self.config.includeCalibVar,
        )

        if not success:
            return None

        with self.timer("warp"):
            warped_exposure = warper.warpExposure(
                target_wcs,
                detector_inputs.exposure,
                maxBBox=maxBBox,
                destBBox=destBBox,
            )

        # Potentially a post-warp interpolation here? Relies on DM-38630.

        return warped_exposure

    def _apply_all_calibrations(
        self,
        detector_inputs: WarpDetectorInputs,
        *,
        visit_summary: ExposureCatalog | None = None,
        includeScaleUncertainty: bool = False,
    ) -> bool:
        """Apply all of the calibrations from visit_summary to the exposure.

        Specifically, this method updates the following (if available) to the
        input exposure in place from ``visit_summary``:

        - Aperture correction map
        - Photometric calibration
        - PSF
        - WCS

        It also reverts and applies backgrounds in ``detector_inputs``.

        Parameters
        ----------
        detector_inputs : `WarpDetectorInputs`
            The input exposure to be processed, along with any other
            per-detector modifications.
        visit_summary : `~lsst.afw.table.ExposureCatalog` | None
            Table of visit summary information.  If not None, the visit summary
            information will be used to update the calibration of the input
            exposures. Otherwise, the input exposures will be used as-is.
        includeScaleUncertainty : bool
            Whether to include the uncertainty on the calibration in the
            resulting variance? Passed onto the `calibrateImage` method of the
            PhotoCalib object attached to ``exp``.

        Returns
        -------
        success : `bool`
            True if all calibrations were successfully applied, False otherwise.

        Raises
        ------
        NotImplementedError
            Raised if ``initial_photo_calib_list`` is provided and if any of
            the elements in it is not constant (see DM-46720).
        RuntimeError
            Raised if ``visit_summary`` is provided but does not contain a
            record corresponding to ``detector_inputs``, or if one of the
            background fields in ``detector_inputs`` is inconsistent with the
            task configuration.
        """
        if self.config.doRevertOldBackground:
            if detector_inputs.background_revert is None:
                raise RuntimeError(
                    f"doRevertOldBackground is True, but no background_revert for {detector_inputs.data_id}."
                )
            detector_inputs.exposure.maskedImage += detector_inputs.background_revert.getImage()
        elif detector_inputs.background_revert is not None:
            raise RuntimeError(
                f"doRevertOldBackground is False, but {detector_inputs.data_id} has a background_revert."
            )

        if visit_summary is not None:
            detector = detector_inputs.exposure.info.getDetector().getId()
            row = visit_summary.find(detector)

            if row is None:
                self.log.info(
                    "Unexpectedly incomplete visit_summary: detector = %s is missing. Skipping it.",
                    detector,
                )
                return False

            if photo_calib := row.getPhotoCalib():
                detector_inputs.exposure.setPhotoCalib(photo_calib)
            else:
                self.log.info(
                    "No photometric calibration found in visit summary for detector = %s. Skipping it.",
                    detector,
                )
                return False

            if wcs := row.getWcs():
                detector_inputs.exposure.setWcs(wcs)
            else:
                self.log.info(
                    "No WCS found in visit summary for detector = %s. Skipping it.",
                    detector,
                )
                return False

            if self.config.useVisitSummaryPsf:
                if psf := row.getPsf():
                    detector_inputs.exposure.setPsf(psf)
                else:
                    self.log.info(
                        "No PSF found in visit summary for detector = %s. Skipping it.",
                        detector,
                    )
                    return False

                if apcorr_map := row.getApCorrMap():
                    detector_inputs.exposure.setApCorrMap(apcorr_map)
                else:
                    self.log.info(
                        "No aperture correction map found in visit summary for detector = %s. Skipping it.",
                    )
                    return False

        elif self.config.useVisitSummaryPsf:
            # We can only get here by calling `run`, not `runQuantum`.
            raise RuntimeError("useVisitSummaryPsf=True but no visit summary provided.")

        if self.config.doApplyNewBackground:
            if detector_inputs.background_apply is None:
                raise RuntimeError(
                    f"doRevertOldBackground is True, but no background_apply for {detector_inputs.data_id}."
                )
            detector_inputs.exposure.maskedImage -= detector_inputs.background_apply.getImage()

        elif detector_inputs.background_apply is not None:
            raise RuntimeError(
                f"doRevertOldBackground is False, but {detector_inputs.data_id} has a background_apply."
            )

        if self.config.removeInitialPhotoCalib:
            if (initial_photo_calib := detector_inputs.initial_photo_calib) is None:
                self.log.warning(
                    "Detector id %d for visit %d has no initialPhotoCalib and will "
                    "not be used in the warp", detector, row["visit"],
                )
                return False

            if not initial_photo_calib._isConstant:
                # TODO DM-46720: remove this limitation and usage of private (why?!) property.
                raise NotImplementedError(
                    "removeInitialPhotoCalib=True can only work when the initialPhotoCalib is constant."
                )

            detector_inputs.exposure.maskedImage /= initial_photo_calib.getCalibrationMean()

        # Calibrate the (masked) image.
        # This should happen even if visit_summary is None.
        photo_calib = detector_inputs.exposure.photoCalib
        detector_inputs.exposure.maskedImage = photo_calib.calibrateImage(
            detector_inputs.exposure.maskedImage, includeScaleUncertainty=includeScaleUncertainty
        )
        detector_inputs.exposure.maskedImage /= photo_calib.getCalibrationMean()

        return True

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
